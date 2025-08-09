import math
from typing import List, Optional

import torch
from torch import nn

from .configuration_llama_v4 import Method1BConfig_v4
from .Method1_v4 import (
    Method1LlamaAttention_v4, 
    Method1DecoderLayer_v4, 
    Method1LlamaModel_v4, 
    Method1LlamaForCausalLM_v4
)


class v4m1B_ModifiedScailingComputation(nn.Module):
    """
    封装修改后的注意力缩放计算逻辑
    将原始的sqrt(d_k)广播成向量，与之前所有层QK^T组成的向量进行点乘
    每个QK^T矩阵都有对应的可学习权重行向量，广播成矩阵后进行逐元素相乘
    
    第i层Transformer拥有i+1个可学习权重行向量（对应前面i+1层的QK^T矩阵）
    使用ModuleList[ParameterList]存储每层的权重行向量列表
    """
    def __init__(self, head_dim: int, num_layers: int):
        super().__init__()
        self.head_dim = head_dim
        self.num_layers = num_layers
        # 为每一层创建权重行向量ParameterList，第i层有i+1个权重行向量
        self.layer_weight_vectors = nn.ModuleList([nn.ParameterList() for _ in range(num_layers)])
        # 记录每层是否已经初始化
        self.layer_initialized = [False] * num_layers
    
    def _initialize_layer_weights(self, layer_idx: int, qk_shapes: List[torch.Size], device: torch.device, dtype: torch.dtype):
        """
        为指定层初始化权重行向量
        
        Args:
            layer_idx: 当前层索引
            qk_shapes: 该层所有QK^T矩阵的形状列表
            device: 设备
            dtype: 数据类型
        """
        if self.layer_initialized[layer_idx]:
            return
        
        # 清空当前层的权重行向量ParameterList
        self.layer_weight_vectors[layer_idx] = nn.ParameterList()
        
        # 为该层的每个QK^T矩阵创建对应的权重行向量
        for qk_shape in qk_shapes:
            # 创建权重行向量，长度为QK^T矩阵的列数，初始化为0（经过exp后为1）
            weight_vector = nn.Parameter(torch.zeros(qk_shape[3], device=device, dtype=dtype))
            # 直接添加到ParameterList中，自动注册参数
            self.layer_weight_vectors[layer_idx].append(weight_vector)
        
        # 标记该层已初始化
        self.layer_initialized[layer_idx] = True
    
    def compute_modified_scaling(self, qk_matrices: List[torch.Tensor], layer_idx: int) -> torch.Tensor:
        """
        计算修改后的注意力缩放
        
        Args:
            qk_matrices: 前面所有层(包括当前层)的QK^T矩阵列表
            layer_idx: 当前层索引
            
        Returns:
            修改后的注意力权重
        """
        # 获取所有QK^T矩阵的形状
        qk_shapes = [qk.shape for qk in qk_matrices]
        
        # 初始化该层的权重行向量（如果还没有初始化）
        self._initialize_layer_weights(layer_idx, qk_shapes, qk_matrices[0].device, qk_matrices[0].dtype)
        
        # 将1/sqrt(d_k)广播成向量，每个元素都是1/sqrt(d_k)  
        scaling_value = 1.0 / math.sqrt(self.head_dim)
        scaling_vector = torch.full((len(qk_matrices),), scaling_value, 
                                   device=qk_matrices[0].device, dtype=qk_matrices[0].dtype)
        
        # 对所有QK^T矩阵进行加权求和，每个矩阵乘以其对应的可学习权重行向量
        weighted_qk = torch.zeros_like(qk_matrices[-1])
        for i, qk_matrix in enumerate(qk_matrices):
            # 获取该层第i个权重行向量
            weight_vector = self.layer_weight_vectors[layer_idx][i]
            
            # 使用exp确保权重为正，初始值exp(0)=1
            exp_weight_vector = torch.exp(weight_vector)
            exp_weight_row = exp_weight_vector.unsqueeze(0)  # [seq_len] -> [1, seq_len]
            weighted_qk += scaling_vector[i] * (qk_matrix * exp_weight_row)
        
        return weighted_qk


class Method1BLlamaAttention_v4(Method1LlamaAttention_v4):
    """
    自定义Attention类，使用带可学习权重行向量的ModifiedScailingComputation
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # 替换为新的缩放计算模块
        self.modified_scaling = v4m1B_ModifiedScailingComputation(self.head_dim, config.num_hidden_layers)


class Method1BDecoderLayer_v4(Method1DecoderLayer_v4):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # 使用自定义的Attention类
        self.self_attn = Method1BLlamaAttention_v4(config=config, layer_idx=layer_idx)


class Method1BLlamaModel_v4(Method1LlamaModel_v4):
    config_class = Method1BConfig_v4

    def __init__(self, config: Method1BConfig_v4):
        super().__init__(config)
        # 替换所有的decoder layer为新的实现
        self.layers = nn.ModuleList(
            [Method1BDecoderLayer_v4(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # 重新初始化权重
        self.post_init()


class Method1BLlamaForCausalLM_v4(Method1LlamaForCausalLM_v4):
    config_class = Method1BConfig_v4

    def __init__(self, config: Method1BConfig_v4):
        super().__init__(config)
        self.model = Method1BLlamaModel_v4(config)
