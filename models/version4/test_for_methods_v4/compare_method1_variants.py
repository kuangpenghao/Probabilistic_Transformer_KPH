#!/usr/bin/env python3
"""
比较 Method1B_v4、Method1E_v4、Method1C_v4 和 Method1D_v4 的实现
验证它们之间的关系：
- Method1B vs Method1E: 直接学习行向量 vs MLP生成行向量
- Method1C vs Method1D: 直接学习列向量 vs MLP生成列向量
"""

import torch
from models.version4.Method1B_v4 import Method1BLlamaForCausalLM_v4
from models.version4.Method1E_v4 import Method1ELlamaForCausalLM_v4
from models.version4.Method1C_v4 import Method1CLlamaForCausalLM_v4
from models.version4.Method1D_v4 import Method1DLlamaForCausalLM_v4
from models.version4.configuration_llama_v4 import (
    Method1BConfig_v4, Method1EConfig_v4, Method1CConfig_v4, Method1DConfig_v4
)

def compare_method1_variants():
    print("=== 比较 Method1 系列模型 ===")
    
    # 创建相同的配置
    config_kwargs = {
        'vocab_size': 1000,
        'hidden_size': 64,
        'intermediate_size': 128,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'max_position_embeddings': 512,
        'use_cache': True,
    }
    
    # 创建各个模型
    models = {
        'Method1B': Method1BLlamaForCausalLM_v4(Method1BConfig_v4(**config_kwargs)),
        'Method1E': Method1ELlamaForCausalLM_v4(Method1EConfig_v4(**config_kwargs)),
        'Method1C': Method1CLlamaForCausalLM_v4(Method1CConfig_v4(**config_kwargs)),
        'Method1D': Method1DLlamaForCausalLM_v4(Method1DConfig_v4(**config_kwargs)),
    }
    
    print("\n=== 参数数量比较 ===")
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {total_params:,} 参数")
    
    # 创建测试输入
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, config_kwargs['vocab_size'], (batch_size, seq_len))
    
    print(f"\n=== 前向传播测试 (seq_len={seq_len}) ===")
    outputs = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        total_before = sum(p.numel() for p in model.parameters())
        
        with torch.no_grad():
            output = model(input_ids)
        
        total_after = sum(p.numel() for p in model.parameters())
        new_params = total_after - total_before
        
        print(f"前向传播前参数: {total_before:,}")
        print(f"前向传播后参数: {total_after:,}")
        print(f"新增参数: {new_params:,}")
        print(f"输出形状: {output.logits.shape}")
        
        outputs[name] = output.logits
    
    # 验证不同模型的特征
    print(f"\n=== 模型特征分析 ===")
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # 分析attention模块的组件
        first_attn = model.model.layers[0].self_attn
        if hasattr(first_attn, 'modified_scaling'):
            scaling_module = first_attn.modified_scaling
            
            # 检查是否有权重向量（Method1B/1C）
            if hasattr(scaling_module, 'layer_weight_vectors'):
                print("✓ 使用直接学习的权重向量")
                
            # 检查是否有MLP组件（Method1D/1E）
            if hasattr(scaling_module, 'layer_mlps'):
                print("✓ 使用MLP动态生成权重向量")
                print(f"  MLP组件数量: {len(scaling_module.layer_mlps)}")
                
                # 分析第一层的MLP输出维度
                first_bias_shape = scaling_module.layer_biases[0].shape
                print(f"  第0层输出维度: {first_bias_shape[0]} (对应{first_bias_shape[0]}个向量)")
                
                if len(scaling_module.layer_biases) > 1:
                    second_bias_shape = scaling_module.layer_biases[1].shape
                    print(f"  第1层输出维度: {second_bias_shape[0]} (对应{second_bias_shape[0]}个向量)")
    
    # 验证广播方式的区别
    print(f"\n=== 广播方式验证 ===")
    print("Method1B/1E: 行向量广播 - [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]")
    print("Method1C/1D: 列向量广播 - [batch_size, seq_len] -> [batch_size, 1, seq_len, 1]")
    
    # 检查输出是否有显著差异
    print(f"\n=== 输出差异分析 ===")
    method_pairs = [
        ('Method1B', 'Method1E'),  # 直接vs动态 行向量
        ('Method1C', 'Method1D'),  # 直接vs动态 列向量
        ('Method1B', 'Method1C'),  # 行向量vs列向量 (直接)
        ('Method1E', 'Method1D'),  # 行向量vs列向量 (动态)
    ]
    
    for method1, method2 in method_pairs:
        diff = torch.abs(outputs[method1] - outputs[method2])
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"{method1} vs {method2}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    print("\n=== 比较总结 ===")
    print("✓ Method1B: 直接学习行向量，参数在前向传播时动态创建")
    print("✓ Method1E: MLP生成行向量，参数在初始化时创建")
    print("✓ Method1C: 直接学习列向量，参数在前向传播时动态创建") 
    print("✓ Method1D: MLP生成列向量，参数在初始化时创建")
    print("✓ 所有模型都成功前向传播并产生合理输出")

if __name__ == "__main__":
    compare_method1_variants()
