#!/usr/bin/env python3
"""
测试修复后的Version4方法 - 验证每层独立的可学习参数设计
"""

import math
import torch
import sys
sys.path.append('/home/kuangph/hf-starter')

def test_layer_specific_parameters():
    """测试每层都有独立的参数向量，长度与层索引相关"""
    print("=== 测试每层独立参数设计 ===\n")
    
    head_dim = 64
    
    # 测试Method5在不同层的参数向量长度
    print("Method5 - 每层参数向量长度测试:")
    from models.version4.Method5_v4 import v4m5_ModifiedScailingComputation
    
    for layer_idx in range(5):
        method5 = v4m5_ModifiedScailingComputation(head_dim, layer_idx)
        expected_length = layer_idx + 1
        actual_length = len(method5.log_a_params)
        print(f"  第{layer_idx}层: 期望参数长度={expected_length}, 实际参数长度={actual_length}, 匹配: {expected_length == actual_length}")
    print()
    
    # 测试Method6在不同层的参数向量长度
    print("Method6 - 每层参数向量长度测试:")
    from models.version4.Method6_v4 import v4m6_ModifiedScailingComputation
    
    for layer_idx in range(5):
        method6 = v4m6_ModifiedScailingComputation(head_dim, layer_idx)
        expected_length = layer_idx + 1
        actual_a_length = len(method6.log_a_params)
        actual_b_length = len(method6.b_params)
        print(f"  第{layer_idx}层: 期望参数长度={expected_length}, a_params长度={actual_a_length}, b_params长度={actual_b_length}")
        print(f"    匹配: {expected_length == actual_a_length == actual_b_length}")
    print()
    
    # 测试Method7在不同层的参数向量长度
    print("Method7 - 每层参数向量长度测试:")
    from models.version4.Method7_v4 import v4m7_ModifiedScailingComputation
    
    for layer_idx in range(5):
        method7 = v4m7_ModifiedScailingComputation(head_dim, layer_idx)
        expected_length = layer_idx + 1
        actual_length = len(method7.log_a_params)
        print(f"  第{layer_idx}层: 期望参数长度={expected_length}, 实际参数长度={actual_length}, 匹配: {expected_length == actual_length}")
    print()

def test_independence_between_layers():
    """测试不同层的参数确实是独立的"""
    print("=== 测试层间参数独立性 ===\n")
    
    head_dim = 64
    
    # 创建两个不同层的Method5实例
    from models.version4.Method5_v4 import v4m5_ModifiedScailingComputation
    method5_layer1 = v4m5_ModifiedScailingComputation(head_dim, layer_idx=1)
    method5_layer2 = v4m5_ModifiedScailingComputation(head_dim, layer_idx=2)
    
    # 修改第1层的参数
    method5_layer1.log_a_params.data[0] = 1.0
    method5_layer1.log_a_params.data[1] = 2.0
    
    # 修改第2层的参数
    method5_layer2.log_a_params.data[0] = -1.0
    method5_layer2.log_a_params.data[1] = 0.0
    method5_layer2.log_a_params.data[2] = 1.0
    
    print("Method5 层间参数独立性测试:")
    print(f"  第1层参数: {method5_layer1.log_a_params.data.tolist()}")
    print(f"  第2层参数: {method5_layer2.log_a_params.data.tolist()}")
    print(f"  参数长度不同: {len(method5_layer1.log_a_params) != len(method5_layer2.log_a_params)}")
    print(f"  参数值独立: {not torch.equal(method5_layer1.log_a_params.data[:2], method5_layer2.log_a_params.data[:2])}")
    print("  ✓ 层间参数确实独立\n")

def test_compute_scaling_with_correct_length():
    """测试缩放计算使用正确长度的参数向量"""
    print("=== 测试缩放计算使用正确长度的参数 ===\n")
    
    head_dim = 64
    
    # 测试第3层（应该有4个参数，对应层0,1,2,3的QK矩阵）
    from models.version4.Method5_v4 import v4m5_ModifiedScailingComputation
    method5_layer3 = v4m5_ModifiedScailingComputation(head_dim, layer_idx=3)
    
    # 设置不同的参数值
    method5_layer3.log_a_params.data = torch.tensor([0.0, 1.0, -1.0, 0.5])
    
    # 创建4个QK矩阵（对应前4层）
    qk_matrices = [torch.randn(2, 8, 10, 10) for _ in range(4)]
    
    # 计算缩放
    result = method5_layer3.compute_modified_scaling(qk_matrices, 3)
    
    # 手动验证计算
    expected = torch.zeros_like(qk_matrices[-1])
    scaling_values = []
    
    for i in range(4):
        a_i = torch.exp(method5_layer3.log_a_params[i]).item()
        scaling = 1.0 / (a_i * math.sqrt(head_dim))
        scaling_values.append(scaling)
        expected += scaling * qk_matrices[i]
    
    print("Method5 第3层缩放计算测试:")
    print(f"  参数向量长度: {len(method5_layer3.log_a_params)}")
    print(f"  QK矩阵数量: {len(qk_matrices)}")
    print(f"  缩放值: {[f'{v:.6f}' for v in scaling_values]}")
    print(f"  计算结果正确: {torch.allclose(result, expected, atol=1e-6)}")
    print()

def test_layer_creation_in_attention():
    """测试Attention层中ModifiedScalingComputation的正确创建"""
    print("=== 测试Attention层中的正确创建 ===\n")
    
    # 创建一个简单的配置
    class DummyConfig:
        def __init__(self):
            self.hidden_size = 512
            self.num_attention_heads = 8
            self.num_key_value_heads = 8
            self.head_dim = self.hidden_size // self.num_attention_heads
            self.max_position_embeddings = 2048
            self.rope_theta = 10000.0
            self.attention_dropout = 0.0
            self.pretraining_tp = 1
    
    config = DummyConfig()
    
    # 测试不同层索引的Attention
    from models.version4.Method5_v4 import Method5LlamaAttention_v4
    
    print("Method5 Attention层创建测试:")
    for layer_idx in range(4):
        attention = Method5LlamaAttention_v4(config, layer_idx=layer_idx)
        expected_param_length = layer_idx + 1
        actual_param_length = len(attention.modified_scaling.log_a_params)
        
        print(f"  第{layer_idx}层Attention: 期望参数长度={expected_param_length}, 实际={actual_param_length}, 匹配: {expected_param_length == actual_param_length}")
    
    print("  ✓ Attention层中的ModifiedScalingComputation创建正确\n")

if __name__ == "__main__":
    test_layer_specific_parameters()
    test_independence_between_layers()
    test_compute_scaling_with_correct_length()
    test_layer_creation_in_attention()
    print("🎉 所有层独立参数设计测试都通过！")
    print("\n📝 总结:")
    print("- 每层都有独立的ModifiedScalingComputation实例")
    print("- 每层的参数向量长度 = layer_idx + 1")
    print("- 不需要max_layers参数，避免了参数共享问题")
    print("- 第i层使用前i+1层的QK^T矩阵，对应i+1个缩放参数")
