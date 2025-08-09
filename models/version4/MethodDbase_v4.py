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

from .configuration_llama_v4 import MethodDbaseConfig_v4


class v4mdbase_ModifiedScalingComputation(nn.Module):
    """
    封装修改后的注意力缩放计算逻辑
    为每一层的注意力模块提供一个通过MLP动态生成的权重列向量
    计算Softmax(QK^T/√d_k)时，将这个列向量显式广播，与QK^T逐元素相乘
    权重列向量通过 A_i = GELU(RMSNorm(X_i)W_1)W_2 + bias 动态生成
    """
    def __init__(self, hidden_size: int, head_dim: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_layers = num_layers
        
        # 为每一层创建MLP组件
        self.layer_mlps = nn.ModuleList()
        self.layer_biases = nn.ParameterList()
        self.layer_norms = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            # RMSNorm
            layer_norm = nn.RMSNorm(hidden_size)
            
            # MLP: hidden_size -> 1 -> 1
            # W_1: hidden_size * 1, W_2: 1 * 1
            mlp = nn.Sequential(
                nn.Linear(hidden_size, 1, bias=False),  # W_1
                nn.GELU(),
                nn.Linear(1, 1, bias=False)  # W_2
            )
            
            # bias向量，规模为 1
            bias = nn.Parameter(torch.zeros(1))
            
            self.layer_mlps.append(mlp)
            self.layer_biases.append(bias)
            self.layer_norms.append(layer_norm)
    
    def compute_modified_scaling(self, qk_matrix: torch.Tensor, layer_idx: int, 
                               input_embedding: torch.Tensor) -> torch.Tensor:
        """
        计算修改后的注意力缩放
        
        Args:
            qk_matrix: 当前层的QK^T矩阵，形状 [batch_size, num_heads, seq_len, seq_len]
            layer_idx: 当前层索引
            input_embedding: 当前层的输入嵌入 X_i，形状 [batch_size, seq_len, hidden_size]
            
        Returns:
            修改后的注意力权重
        """
        # input_embedding形状: [batch_size, seq_len, hidden_size]
        normed_input = self.layer_norms[layer_idx](input_embedding)  # RMSNorm
        mlp_output = self.layer_mlps[layer_idx](normed_input)  # [batch_size, seq_len, 1]
        
        # 添加bias（广播相加）
        weight_column_A = mlp_output + self.layer_biases[layer_idx]  # [batch_size, seq_len, 1]
        
        # 压缩最后一个维度得到列向量
        weight_column = weight_column_A.squeeze(-1)  # [batch_size, seq_len]
        
        # 使用exp确保权重为正
        exp_weight_column = torch.exp(weight_column)  # [batch_size, seq_len]
        
        # 计算标准的缩放
        scaling_value = 1.0 / math.sqrt(self.head_dim)
        scaled_qk = qk_matrix * scaling_value  # [batch_size, num_heads, seq_len, seq_len]
        
        # 显式列向量广播：[batch_size, seq_len] -> [batch_size, 1, seq_len, 1]
        exp_weight_broadcast = exp_weight_column.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, seq_len, 1]
        
        # 与QK^T矩阵逐元素相乘
        weighted_qk = scaled_qk * exp_weight_broadcast
        
        return weighted_qk


class MethodDbaseLlamaAttention_v4(LlamaAttention):
    """
    基于原始LlamaAttention的自定义Attention类，使用MLP动态生成权重列向量
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # 添加修改后的缩放计算模块
        self.modified_scaling = v4mdbase_ModifiedScalingComputation(
            config.hidden_size, self.head_dim, config.num_hidden_layers)
    
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
        重写forward方法，使用修改后的缩放计算，传入输入嵌入
        """
        # 保存输入嵌入以传递给scaling computation
        input_embedding = hidden_states  # [batch_size, seq_len, hidden_size]
        
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
        
        # 使用修改后的缩放计算（传入输入嵌入）
        attn_weights = self.modified_scaling.compute_modified_scaling(
            current_qk_matrix, self.layer_idx, input_embedding)

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


class MethodDbaseDecoderLayer_v4(LlamaDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # 使用自定义的Attention类
        self.self_attn = MethodDbaseLlamaAttention_v4(config=config, layer_idx=layer_idx)


class MethodDbaseLlamaModel_v4(LlamaModel):
    config_class = MethodDbaseConfig_v4

    def __init__(self, config: MethodDbaseConfig_v4):
        super().__init__(config)
        # 替换所有的decoder layer为新的实现
        self.layers = nn.ModuleList(
            [MethodDbaseDecoderLayer_v4(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # 重新初始化权重
        self.post_init()


class MethodDbaseLlamaForCausalLM_v4(LlamaForCausalLM):
    config_class = MethodDbaseConfig_v4

    def __init__(self, config: MethodDbaseConfig_v4):
        super().__init__(config)
        self.model = MethodDbaseLlamaModel_v4(config)
        
        # 重新初始化权重
        self.post_init()
