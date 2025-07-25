import math
from typing import List, Optional

import torch
from torch import nn

from .configuration_llama_v4 import Method2Config_v4
from .Method1_v4 import (
    Method1LlamaAttention_v4, 
    Method1DecoderLayer_v4, 
    Method1LlamaModel_v4, 
    Method1LlamaForCausalLM_v4
)


class v4m2_ModifiedScailingComputation(nn.Module):
    """
    封装修改后的注意力缩放计算逻辑
    使用1/sqrt(d_k*m)进行缩放，其中m是矩阵数量
    """
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
    
    def compute_modified_scaling(self, qk_matrices: List[torch.Tensor], layer_idx: int) -> torch.Tensor:
        """
        计算修改后的注意力缩放
        
        Args:
            qk_matrices: 前面所有层(包括当前层)的QK^T矩阵列表
            layer_idx: 当前层索引
            
        Returns:
            修改后的注意力权重
        """
        # 计算1/sqrt(d_k*m)的倒数，其中m是矩阵数量
        m = len(qk_matrices)
        scaling_value = 1.0 / math.sqrt(self.head_dim * m)
        scaling_vector = torch.full((len(qk_matrices),), scaling_value, 
                                   device=qk_matrices[0].device, dtype=qk_matrices[0].dtype)
        
        # 对所有QK^T矩阵进行加权求和
        weighted_qk = torch.zeros_like(qk_matrices[-1])
        for i, qk_matrix in enumerate(qk_matrices):
            weighted_qk += scaling_vector[i] * qk_matrix
        
        return weighted_qk


class Method2LlamaAttention_v4(Method1LlamaAttention_v4):
    """
    继承Method1的Attention类，只修改ModifiedScalingComputation
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # 替换为Method2的缩放计算
        self.modified_scaling = v4m2_ModifiedScailingComputation(self.head_dim)


class Method2DecoderLayer_v4(Method1DecoderLayer_v4):
    """
    继承Method1的DecoderLayer类，只修改Attention模块
    """
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # 替换为Method2的Attention类
        self.self_attn = Method2LlamaAttention_v4(config=config, layer_idx=layer_idx)


class Method2LlamaModel_v4(Method1LlamaModel_v4):
    """
    继承Method1的Model类，只修改config和decoder layers
    """
    config_class = Method2Config_v4

    def __init__(self, config: Method2Config_v4):
        super().__init__(config)
        # 替换为Method2的decoder layers
        self.layers = nn.ModuleList(
            [Method2DecoderLayer_v4(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class Method2LlamaForCausalLM_v4(Method1LlamaForCausalLM_v4):
    """
    继承Method1的CausalLM类，只修改config和model
    """
    config_class = Method2Config_v4

    def __init__(self, config: Method2Config_v4):
        # 先调用父类初始化，然后替换model
        super().__init__(config)
        self.model = Method2LlamaModel_v4(config)
