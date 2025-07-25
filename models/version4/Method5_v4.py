import math
from typing import List, Optional

import torch
from torch import nn

from .configuration_llama_v4 import Method5Config_v4
from .Method1_v4 import (
    Method1LlamaAttention_v4, 
    Method1DecoderLayer_v4, 
    Method1LlamaModel_v4, 
    Method1LlamaForCausalLM_v4
)


class v4m5_ModifiedScailingComputation(nn.Module):
    """
    封装修改后的注意力缩放计算逻辑
    学习一个向量，向量的每一项为a_i*sqrt(d_k)，其中a_i是可学习参数
    每个层都有自己独立的参数向量，长度等于该层的索引+1
    """
    def __init__(self, head_dim: int, layer_idx: int):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        
        # 初始化可学习参数向量a_i，长度为layer_idx+1，需要保证为正值
        vector_length = layer_idx + 1
        self.log_a_params = nn.Parameter(torch.zeros(vector_length))  # 使用log确保正值
    
    def compute_modified_scaling(self, qk_matrices: List[torch.Tensor], layer_idx: int) -> torch.Tensor:
        """
        计算修改后的注意力缩放
        
        Args:
            qk_matrices: 前面所有层(包括当前层)的QK^T矩阵列表
            layer_idx: 当前层索引
            
        Returns:
            修改后的注意力权重
        """
        # 构建缩放向量，每一项为1/(a_i*sqrt(d_k))的倒数，a_i需要是正值
        num_matrices = len(qk_matrices)
        scaling_vector = torch.zeros(num_matrices, device=qk_matrices[0].device, dtype=qk_matrices[0].dtype)
        
        for i in range(num_matrices):
            # 每层都有自己的参数向量，长度等于层索引+1
            a_i = torch.exp(self.log_a_params[i])  # 确保a_i为正值
            
            # 计算1/(a_i*sqrt(d_k))
            scaling_vector[i] = 1.0 / (a_i * math.sqrt(self.head_dim))
        
        # 对所有QK^T矩阵进行加权求和
        weighted_qk = torch.zeros_like(qk_matrices[-1])
        for i, qk_matrix in enumerate(qk_matrices):
            weighted_qk += scaling_vector[i] * qk_matrix
        
        return weighted_qk


class Method5LlamaAttention_v4(Method1LlamaAttention_v4):
    """
    继承Method1的Attention类，只修改ModifiedScalingComputation
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # 替换为Method5的缩放计算，每层都有自己独立的参数向量
        self.modified_scaling = v4m5_ModifiedScailingComputation(self.head_dim, layer_idx)


class Method5DecoderLayer_v4(Method1DecoderLayer_v4):
    """
    继承Method1的DecoderLayer类，只修改Attention模块
    """
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # 替换为Method5的Attention类
        self.self_attn = Method5LlamaAttention_v4(config=config, layer_idx=layer_idx)


class Method5LlamaModel_v4(Method1LlamaModel_v4):
    """
    继承Method1的Model类，只修改config和decoder layers
    """
    config_class = Method5Config_v4

    def __init__(self, config: Method5Config_v4):
        super().__init__(config)
        # 替换为Method5的decoder layers
        self.layers = nn.ModuleList(
            [Method5DecoderLayer_v4(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class Method5LlamaForCausalLM_v4(Method1LlamaForCausalLM_v4):
    """
    继承Method1的CausalLM类，只修改config和model
    """
    config_class = Method5Config_v4

    def __init__(self, config: Method5Config_v4):
        super().__init__(config)
        self.model = Method5LlamaModel_v4(config)
    
    def get_all_layer_weights(self):
        """
        获取所有层的可学习参数向量
        返回格式: List[torch.Tensor] 每层的参数向量，长度为layer_idx+1
        """
        all_weights = []
        
        for layer_idx, layer in enumerate(self.model.layers):
            layer_weights = torch.empty(0)  # 默认空tensor
            
            # 获取该层attention中的可学习参数向量
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'modified_scaling'):
                scaling_module = layer.self_attn.modified_scaling
                if hasattr(scaling_module, 'log_a_params'):
                    # 获取实际的a参数（对log_a_params取exp）
                    a_params = torch.exp(scaling_module.log_a_params).detach().cpu()
                    layer_weights = a_params
            
            all_weights.append(layer_weights)
            
        return all_weights
    
    def save_learned_parameters(self, output_dir: str):
        """
        保存可学习参数到指定目录
        """
        import json
        import os
        
        # 获取所有层的权重
        all_weights = self.get_all_layer_weights()
        
        # 转换为可序列化格式
        weights_data = {}
        for layer_idx, layer_weights in enumerate(all_weights):
            if len(layer_weights) > 0:  # 如果该层有参数
                weights_data[f"layer_{layer_idx}"] = layer_weights.numpy().tolist()
        
        # 保存到文件
        weights_file = os.path.join(output_dir, "method5_v4_learned_parameters.json")
        with open(weights_file, 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        # 保存统计信息
        stats_file = weights_file.replace('.json', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write("Method5_v4 Learned Parameters\n")
            f.write("="*40 + "\n")
            f.write(f"Total layers: {len(weights_data)}\n")
            f.write(f"Parameters: Vector a_i for scaling 1/(a_i*sqrt(d_k))\n\n")
            
            for layer_idx, layer_weights in enumerate(all_weights):
                if len(layer_weights) > 0:
                    weights_np = layer_weights.numpy()
                    f.write(f"Layer {layer_idx} (vector length: {len(weights_np)}):\n")
                    f.write(f"  Parameters: {weights_np}\n")
                    f.write(f"  Mean: {weights_np.mean():.6f}\n")
                    f.write(f"  Std: {weights_np.std():.6f}\n")
                    f.write(f"  Min: {weights_np.min():.6f}\n")
                    f.write(f"  Max: {weights_np.max():.6f}\n")
                    f.write(f"  Scaling formula: weighted sum with 1/(a_i*sqrt(d_k))\n")
                    f.write("-" * 30 + "\n")
        
        return weights_file
