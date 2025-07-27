import math
from typing import List, Optional

import torch
from torch import nn

from .configuration_llama_v4 import Method8Config_v4
from .Method1_v4 import (
    Method1LlamaAttention_v4, 
    Method1DecoderLayer_v4, 
    Method1LlamaModel_v4, 
    Method1LlamaForCausalLM_v4
)


class v4m8_ModifiedScailingComputation(nn.Module):
    """
    封装修改后的注意力缩放计算逻辑
    """
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        
        # 学习参数：a初始化为0.5(匹配1/sqrt(d_k))
        self.a = nn.Parameter(torch.tensor(0.5))
    
    def compute_modified_scaling(self, qk_matrices: List[torch.Tensor], layer_idx: int) -> torch.Tensor:
        """
        计算修改后的注意力缩放
        
        Args:
            qk_matrices: 前面所有层(包括当前层)的QK^T矩阵列表
            layer_idx: 当前层索引
            
        Returns:
            修改后的注意力权重
        """
        # 计算1/(d_k^a)的倒数，其中m是矩阵数量，a和b可为正负
        m = len(qk_matrices)
        scaling_value = 1.0 / (self.head_dim ** self.a)
        # 确保scaling_value在正确的设备和数据类型上，同时保持梯度计算图
        device = qk_matrices[0].device
        dtype = qk_matrices[0].dtype
        scaling_value = scaling_value.to(device=device, dtype=dtype)
        scaling_vector = scaling_value.expand(len(qk_matrices))
        
        # 对所有QK^T矩阵进行加权求和
        weighted_qk = torch.zeros_like(qk_matrices[-1])
        for i, qk_matrix in enumerate(qk_matrices):
            weighted_qk += scaling_vector[i] * qk_matrix
        
        return weighted_qk


class Method8LlamaAttention_v4(Method1LlamaAttention_v4):
    """
    继承Method1的Attention类，只修改ModifiedScalingComputation
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # 替换为Method8的缩放计算
        self.modified_scaling = v4m8_ModifiedScailingComputation(self.head_dim)


class Method8DecoderLayer_v4(Method1DecoderLayer_v4):
    """
    继承Method1的DecoderLayer类，只修改Attention模块
    """
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # 替换为Method8的Attention类
        self.self_attn = Method8LlamaAttention_v4(config=config, layer_idx=layer_idx)


class Method8LlamaModel_v4(Method1LlamaModel_v4):
    """
    继承Method1的Model类，只修改config和decoder layers
    """
    config_class = Method8Config_v4

    def __init__(self, config: Method8Config_v4):
        super().__init__(config)
        # 替换为Method8的decoder layers
        self.layers = nn.ModuleList(
            [Method8DecoderLayer_v4(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class Method8LlamaForCausalLM_v4(Method1LlamaForCausalLM_v4):
    """
    继承Method1的CausalLM类，只修改config和model
    """
    config_class = Method8Config_v4

    def __init__(self, config: Method8Config_v4):
        super().__init__(config)
        self.model = Method8LlamaModel_v4(config)
    
    def get_all_layer_weights(self):
        """
        获取所有层的可学习参数a和b
        返回格式: List[Dict] 每层一个字典，包含该层的a和b参数
        """
        all_weights = []
        
        for layer_idx, layer in enumerate(self.model.layers):
            layer_weights = {}
            
            # 获取该层attention中的可学习参数
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'modified_scaling'):
                scaling_module = layer.self_attn.modified_scaling
                if hasattr(scaling_module, 'a') and hasattr(scaling_module, 'b'):
                    layer_weights['a'] = scaling_module.a.detach().cpu()
                    layer_weights['b'] = scaling_module.b.detach().cpu()
            
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
                    layer_data[param_name] = param_tensor.item()  # 标量参数直接转换
                weights_data[f"layer_{layer_idx}"] = layer_data
        
        # 保存到文件
        weights_file = os.path.join(output_dir, "Method8_v4_learned_parameters.json")
        with open(weights_file, 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        # 保存统计信息
        stats_file = weights_file.replace('.json', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write("Method8_v4 Learned Parameters\n")
            f.write("="*40 + "\n")
            f.write(f"Total layers: {len(weights_data)}\n")
            f.write(f"Parameters per layer: a (power for d_k)\n\n")
            
            for layer_idx, layer_weights in enumerate(all_weights):
                if layer_weights:
                    f.write(f"Layer {layer_idx}:\n")
                    a_val = layer_weights['a'].item()
                    f.write(f"  a = {a_val:.6f} (d_k^{a_val:.3f})\n")
                    f.write("-" * 30 + "\n")
        
        return weights_file
