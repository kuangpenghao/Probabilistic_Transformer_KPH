import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention, 
    LlamaDecoderLayer, 
    LlamaModel, 
    LlamaForCausalLM,
    apply_rotary_pos_emb, 
    repeat_kv
)
from transformers.cache_utils import Cache

from .configuration_llama_v4 import MethodCbaseConfig_v4


class v4mcbase_ModifiedScalingComputation(nn.Module):
    """
    封装修改后的注意力缩放计算逻辑
    为每一层的注意力模块提供一个可学习的权重列向量，长度为seq_len
    计算Softmax(QK^T/√d_k)时，将这个列向量显式广播，与QK^T逐元素相乘
    然后再加上一个seq_len维的bias向量，广播成seq_len×seq_len的形状
    """
    def __init__(self, head_dim: int, num_layers: int):
        super().__init__()
        self.head_dim = head_dim
        self.num_layers = num_layers
        
        # 记录每层是否已初始化权重向量和bias向量
        self.layer_initialized = [False] * num_layers
        
        # 为每一层创建可学习权重向量的容器
        self.layer_weight_vectors = nn.ModuleList([nn.ParameterList() for _ in range(num_layers)])
        
        # 为每一层创建可学习bias向量的容器 (seq_len维)
        self.layer_bias_vectors = nn.ModuleList([nn.ParameterList() for _ in range(num_layers)])
    
    def _initialize_layer_weights(self, layer_idx: int, seq_len: int, device, dtype):
        """
        初始化指定层的权重列向量和bias向量
        
        Args:
            layer_idx: 层索引
            seq_len: 序列长度
            device: 设备
            dtype: 数据类型
        """
        if self.layer_initialized[layer_idx]:
            return
        
        # 清空当前层的权重列向量ParameterList
        self.layer_weight_vectors[layer_idx] = nn.ParameterList()
        # 清空当前层的bias向量ParameterList
        self.layer_bias_vectors[layer_idx] = nn.ParameterList()
        
        # 创建权重列向量，长度为seq_len，初始化为0（经过exp后为1）
        weight_vector = nn.Parameter(torch.zeros(seq_len, device=device, dtype=dtype))
        # 直接添加到ParameterList中，自动注册参数
        self.layer_weight_vectors[layer_idx].append(weight_vector)
        
        # 创建bias向量，长度为seq_len，初始化为0
        bias_vector = nn.Parameter(torch.zeros(seq_len, device=device, dtype=dtype))
        # 直接添加到ParameterList中，自动注册参数
        self.layer_bias_vectors[layer_idx].append(bias_vector)
        
        # 标记该层已初始化
        self.layer_initialized[layer_idx] = True
    
    def compute_modified_scaling(self, qk_matrix: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        计算修改后的注意力缩放
        
        Args:
            qk_matrix: 当前层的QK^T矩阵
            layer_idx: 当前层索引
            
        Returns:
            修改后的注意力权重
        """
        # 获取QK^T矩阵的序列长度
        seq_len = qk_matrix.shape[2]
        
        # 初始化该层的权重列向量（如果还没有初始化）
        self._initialize_layer_weights(layer_idx, seq_len, qk_matrix.device, qk_matrix.dtype)
        
        # 获取权重列向量
        weight_vector = self.layer_weight_vectors[layer_idx][0]  # [seq_len]
        
        # 使用exp确保权重为正
        exp_weight_vector = torch.exp(weight_vector)  # [seq_len]
        
        # 计算标准的缩放
        scaling_value = 1.0 / math.sqrt(self.head_dim)
        scaled_qk = qk_matrix * scaling_value  # [batch_size, num_heads, seq_len, seq_len]
        
        # 显式列向量广播：[seq_len] -> [1, 1, seq_len, 1]
        exp_weight_broadcast = exp_weight_vector.view(1, 1, seq_len, 1)

        exp_weight_broadcast=exp_weight_broadcast + 1
        
        # 与QK^T矩阵逐元素相乘
        weighted_qk = scaled_qk * exp_weight_broadcast
        
        # 获取bias向量并广播
        bias_vector = self.layer_bias_vectors[layer_idx][0]  # [seq_len]
        # 将bias向量广播成seq_len×seq_len的形状: [seq_len] -> [1, 1, seq_len, seq_len]
        # 每一行都是相同的bias_vector
        bias_broadcast = bias_vector.view(1, 1, seq_len, 1).expand(-1, -1, -1, seq_len)
        
        # 加上bias
        final_qk = weighted_qk + bias_broadcast
        
        return final_qk


class MethodCbaseLlamaAttention_v4(LlamaAttention):
    """
    基于原始LlamaAttention的自定义Attention类，使用可学习权重列向量
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # 添加修改后的缩放计算模块
        self.modified_scaling = v4mcbase_ModifiedScalingComputation(
            self.head_dim, config.num_hidden_layers)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        """
        重写forward方法，使用修改后的缩放计算
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        past_key_value = past_key_value if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 计算QK^T矩阵 (未经过缩放)
        current_qk_matrix = torch.matmul(query_states, key_states.transpose(2, 3))
        
        # 使用修改后的缩放计算
        attn_weights = self.modified_scaling.compute_modified_scaling(
            current_qk_matrix, self.layer_idx)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MethodCbaseDecoderLayer_v4(LlamaDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # 使用自定义的Attention类
        self.self_attn = MethodCbaseLlamaAttention_v4(config=config, layer_idx=layer_idx)


class MethodCbaseLlamaModel_v4(LlamaModel):
    config_class = MethodCbaseConfig_v4

    def __init__(self, config: MethodCbaseConfig_v4):
        super().__init__(config)
        # 替换所有的decoder layer为新的实现
        self.layers = nn.ModuleList(
            [MethodCbaseDecoderLayer_v4(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # 重新初始化权重
        self.post_init()


class MethodCbaseLlamaForCausalLM_v4(LlamaForCausalLM):
    config_class = MethodCbaseConfig_v4

    def __init__(self, config: MethodCbaseConfig_v4):
        super().__init__(config)
        self.model = MethodCbaseLlamaModel_v4(config)
        
        # 重新初始化权重
        self.post_init()
