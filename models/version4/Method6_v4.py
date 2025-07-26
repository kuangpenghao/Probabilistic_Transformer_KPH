import math
from typing import List, Optional

import torch
from torch import nn

from .configuration_llama_v4 import Method6Config_v4
from .Method1_v4 import (
    Method1LlamaAttention_v4, 
    Method1DecoderLayer_v4, 
    Method1LlamaModel_v4, 
    Method1LlamaForCausalLM_v4
)


class v4m6_ModifiedScailingComputation(nn.Module):
    """
    封装修改后的注意力缩放计算逻辑
    学习一个向量，向量的每一项为a_i * d_k^{b_i}，其中a_i和b_i都是可学习参数
    每个层都有自己独立的参数向量，长度等于该层的索引+1
    """
    def __init__(self, head_dim: int, layer_idx: int):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        
        # 初始化可学习参数向量：a_i，b_i正负性随意
        vector_length = layer_idx + 1
        self.a_params = nn.Parameter(torch.ones(vector_length)) 
        self.b_params = nn.Parameter(torch.ones(vector_length) * 0.5)  # b_i正负性随意
    
    def compute_modified_scaling(self, qk_matrices: List[torch.Tensor], layer_idx: int) -> torch.Tensor:
        """
        计算修改后的注意力缩放
        
        Args:
            qk_matrices: 前面所有层(包括当前层)的QK^T矩阵列表
            layer_idx: 当前层索引
            
        Returns:
            修改后的注意力权重
        """
        # 构建缩放向量，每一项为1/(a_i * d_k^{b_i})的倒数
        num_matrices = len(qk_matrices)
        scaling_vector = torch.zeros(num_matrices, device=qk_matrices[0].device, dtype=qk_matrices[0].dtype)
        
        for i in range(num_matrices):
            # 每层都有自己的参数向量，长度等于层索引+1
            a_i = self.a_params[i]
            b_i = self.b_params[i]  # b_i正负性随意
            
            # 计算1/(a_i * d_k^{b_i})
            scaling_vector[i] = 1.0 / (a_i * (self.head_dim ** b_i))
        
        # 对所有QK^T矩阵进行加权求和
        weighted_qk = torch.zeros_like(qk_matrices[-1])
        for i, qk_matrix in enumerate(qk_matrices):
            weighted_qk += scaling_vector[i] * qk_matrix
        
        return weighted_qk


class Method6LlamaAttention_v4(Method1LlamaAttention_v4):
    """
    继承Method1的Attention类，只修改ModifiedScalingComputation
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # 替换为Method6的缩放计算，每层都有自己独立的参数向量
        self.modified_scaling = v4m6_ModifiedScailingComputation(self.head_dim, layer_idx)


class Method6DecoderLayer_v4(Method1DecoderLayer_v4):
    """
    继承Method1的DecoderLayer类，只修改Attention模块
    """
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # 替换为Method6的Attention类
        self.self_attn = Method6LlamaAttention_v4(config=config, layer_idx=layer_idx)


class Method6LlamaModel_v4(Method1LlamaModel_v4):
    """
    继承Method1的Model类，只修改config和decoder layers
    """
    config_class = Method6Config_v4

    def __init__(self, config: Method6Config_v4):
        super().__init__(config)
        # 替换为Method6的decoder layers
        self.layers = nn.ModuleList(
            [Method6DecoderLayer_v4(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class Method6LlamaForCausalLM_v4(Method1LlamaForCausalLM_v4):
    """
    继承Method1的CausalLM类，只修改config和model
    """
    config_class = Method6Config_v4

    def __init__(self, config: Method6Config_v4):
        super().__init__(config)
        self.model = Method6LlamaModel_v4(config)
    
    def get_all_layer_weights(self):
        """
        获取所有层的可学习参数向量 (a_i, b_i)
        返回格式: List[Dict] 每层包含两个向量 a_params 和 b_params
        """
        all_weights = []
        
        for layer_idx, layer in enumerate(self.model.layers):
            layer_weights = {}
            
            # 获取该层attention中的可学习参数向量
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'modified_scaling'):
                scaling_module = layer.self_attn.modified_scaling
                if hasattr(scaling_module, 'log_a_params') and hasattr(scaling_module, 'b_params'):
                    # 获取实际的a参数（对log_a_params取exp）
                    a_params = torch.exp(scaling_module.log_a_params).detach().cpu()
                    b_params = scaling_module.b_params.detach().cpu()
                    layer_weights['a_params'] = a_params
                    layer_weights['b_params'] = b_params
            
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
            if layer_weights:  # 如果该层有参数
                layer_data = {}
                for param_name, param_tensor in layer_weights.items():
                    layer_data[param_name] = param_tensor.numpy().tolist()
                weights_data[f"layer_{layer_idx}"] = layer_data
        
        # 保存到文件
        weights_file = os.path.join(output_dir, "method6_v4_learned_parameters.json")
        with open(weights_file, 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        # 保存统计信息
        stats_file = weights_file.replace('.json', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write("Method6_v4 Learned Parameters\n")
            f.write("="*40 + "\n")
            f.write(f"Total layers: {len(weights_data)}\n")
            f.write(f"Parameters: Vectors a_i and b_i for scaling 1/(a_i * d_k^b_i)\n\n")
            
            for layer_idx, layer_weights in enumerate(all_weights):
                if layer_weights:
                    a_params = layer_weights['a_params'].numpy()
                    b_params = layer_weights['b_params'].numpy()
                    f.write(f"Layer {layer_idx} (vector length: {len(a_params)}):\n")
                    f.write(f"  a_params: {a_params}\n")
                    f.write(f"  a_mean: {a_params.mean():.6f}, a_std: {a_params.std():.6f}\n")
                    f.write(f"  b_params: {b_params}\n")
                    f.write(f"  b_mean: {b_params.mean():.6f}, b_std: {b_params.std():.6f}\n")
                    f.write(f"  Scaling formula: weighted sum with 1/(a_i * d_k^b_i)\n")
                    f.write("-" * 30 + "\n")
        
        return weights_file
