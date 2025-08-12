import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache

from .configuration_llama_v4 import Method1EConfig_v4
from .Method1_v4 import (
    Method1LlamaAttention_v4, 
    Method1DecoderLayer_v4, 
    Method1LlamaModel_v4, 
    Method1LlamaForCausalLM_v4
)


class v4m1E_ModifiedScailingComputation(nn.Module):
    """
    封装修改后的注意力缩放计算逻辑
    将原始的sqrt(d_k)广播成向量，与之前所有层QK^T组成的向量进行点乘
    每个QK^T矩阵都有对应的可学习权重行向量，但这些行向量不是直接学习的，
    而是由输入嵌入X_i通过MLP动态生成：A_i = GELU(RMSNorm(X_i)W_1)W_2 + bias
    
    第i层Transformer拥有i+1个权重行向量（对应前面i+1层的QK^T矩阵）
    这些行向量组成矩阵A_i，规模为(i+1)*T，由输入嵌入动态生成
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
            # 该层需要生成 (layer_idx + 1) 个行向量
            output_dim = layer_idx + 1
            
            # RMSNorm
            layer_norm = nn.RMSNorm(hidden_size)
            
            # MLP: hidden_size -> output_dim -> output_dim
            # 生成的输出将被重塑为 [batch_size, output_dim, seq_len]，每一行是一个权重行向量
            mlp = nn.Sequential(
                nn.Linear(hidden_size, output_dim, bias=False),
                nn.GELU(),
                nn.Linear(output_dim, output_dim, bias=False)
            )
            
            # bias向量，规模为 (layer_idx + 1)
            bias = nn.Parameter(torch.zeros(output_dim))
            
            self.layer_mlps.append(mlp)
            self.layer_biases.append(bias)
            self.layer_norms.append(layer_norm)
    
    def compute_modified_scaling(self, qk_matrices: List[torch.Tensor], layer_idx: int, 
                               input_embedding: torch.Tensor) -> torch.Tensor:
        """
        计算修改后的注意力缩放
        
        Args:
            qk_matrices: 前面所有层(包括当前层)的QK^T矩阵列表
            layer_idx: 当前层索引
            input_embedding: 当前层的输入嵌入 X_i，形状 [batch_size, seq_len, hidden_size]
            
        Returns:
            修改后的注意力权重
        """
        # 将1/sqrt(d_k)广播成向量，每个元素都是1/sqrt(d_k)  
        scaling_value = 1.0 / math.sqrt(self.head_dim)
        scaling_vector = torch.full((len(qk_matrices),), scaling_value, 
                                   device=qk_matrices[0].device, dtype=qk_matrices[0].dtype)
        
        # input_embedding形状: [batch_size, seq_len, hidden_size]
        normed_input = self.layer_norms[layer_idx](input_embedding)  # RMSNorm
        mlp_output = self.layer_mlps[layer_idx](normed_input)  # [batch_size, seq_len, layer_idx+1]
        
        # 添加bias（广播相加）
        weight_matrix_A = mlp_output + self.layer_biases[layer_idx]  # [batch_size, seq_len, layer_idx+1]
        
        # 对所有QK^T矩阵进行加权求和，每个矩阵乘以其对应的动态生成的权重行向量
        weighted_qk = torch.zeros_like(qk_matrices[-1])
        for i, qk_matrix in enumerate(qk_matrices):
            # 获取该层第i个动态生成的权重行向量
            # weight_matrix_A形状: [batch_size, seq_len, layer_idx+1]
            # 取第i列: [batch_size, seq_len]
            weight_row = weight_matrix_A[:, :, i]  # [batch_size, seq_len]
            
            # 使用exp确保权重为正
            exp_weight_row = torch.exp(weight_row)  # [batch_size, seq_len]
            
            # 显式行向量广播：[batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            # 然后与qk_matrix相乘: [batch_size, num_heads, seq_len, seq_len]
            exp_weight_broadcast = exp_weight_row.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            
            weighted_qk += scaling_vector[i] * (qk_matrix * exp_weight_broadcast)
        
        return weighted_qk


class Method1ELlamaAttention_v4(Method1LlamaAttention_v4):
    """
    自定义Attention类，使用动态生成权重行向量的ModifiedScailingComputation
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # 替换为新的缩放计算模块
        self.modified_scaling = v4m1E_ModifiedScailingComputation(
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
        previous_qk_matrices: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ):
        """
        重写forward方法，将输入嵌入传递给modified_scaling
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
        
        # 收集所有QK^T矩阵
        all_qk_matrices = []
        if previous_qk_matrices is not None:
            all_qk_matrices.extend(previous_qk_matrices)
        all_qk_matrices.append(current_qk_matrix)
        
        # 使用修改后的缩放计算（传入输入嵌入）
        attn_weights = self.modified_scaling.compute_modified_scaling(
            all_qk_matrices, self.layer_idx, input_embedding)

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

        return attn_output, attn_weights, past_key_value, current_qk_matrix


class Method1EDecoderLayer_v4(Method1DecoderLayer_v4):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # 使用自定义的Attention类
        self.self_attn = Method1ELlamaAttention_v4(config=config, layer_idx=layer_idx)


class Method1ELlamaModel_v4(Method1LlamaModel_v4):
    config_class = Method1EConfig_v4

    def __init__(self, config: Method1EConfig_v4):
        super().__init__(config)
        # 替换所有的decoder layer为新的实现
        self.layers = nn.ModuleList(
            [Method1EDecoderLayer_v4(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # 重新初始化权重
        self.post_init()


class Method1ELlamaForCausalLM_v4(Method1LlamaForCausalLM_v4):
    config_class = Method1EConfig_v4

    def __init__(self, config: Method1EConfig_v4):
        super().__init__(config)
        self.model = Method1ELlamaModel_v4(config)
        
        # 重新初始化权重
        self.post_init()
