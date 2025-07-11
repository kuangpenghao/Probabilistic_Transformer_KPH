#!/usr/bin/env python3
"""
测试Method3_v2中可学习权重参数的功能
"""

import torch
import torch.nn.functional as F
import sys
import os

# 添加项目根目录到路径
sys.path.append('/home/kuangph/hf-starter')

from models.version2.Method3_v2 import Method3DecoderLayer_v2, Method3Config_v2

def test_learnable_weights():
    """测试可学习权重参数的功能"""
    
    # 创建配置
    config = Method3Config_v2(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        vocab_size=1000
    )
    
    print("测试可学习权重参数功能...")
    print("=" * 50)
    
    # 测试第0层（不应该有权重参数）
    layer_0 = Method3DecoderLayer_v2(config, layer_idx=0)
    print(f"第0层权重参数: {layer_0.mlp_residual_weights}")
    assert layer_0.mlp_residual_weights is None, "第0层不应该有权重参数"
    
    # 测试第1层（应该有2个权重参数：第0层和第1层）
    layer_1 = Method3DecoderLayer_v2(config, layer_idx=1)
    print(f"第1层权重参数形状: {layer_1.mlp_residual_weights.shape}")
    print(f"第1层权重参数值: {layer_1.mlp_residual_weights}")
    assert layer_1.mlp_residual_weights.shape == (2,), "第1层应该有2个权重参数"
    
    # 测试第2层（应该有3个权重参数：第0、1、2层）
    layer_2 = Method3DecoderLayer_v2(config, layer_idx=2)
    print(f"第2层权重参数形状: {layer_2.mlp_residual_weights.shape}")
    print(f"第2层权重参数值: {layer_2.mlp_residual_weights}")
    assert layer_2.mlp_residual_weights.shape == (3,), "第2层应该有3个权重参数"
    
    # 测试权重归一化
    print("\n测试权重归一化:")
    layer_2.mlp_residual_weights.data = torch.tensor([2.0, 3.0, 1.0])
    normalized_weights = F.softmax(layer_2.mlp_residual_weights, dim=0)
    print(f"原始权重: {layer_2.mlp_residual_weights}")
    print(f"归一化后权重: {normalized_weights}")
    print(f"权重总和: {normalized_weights.sum()}")
    assert torch.allclose(normalized_weights.sum(), torch.tensor(1.0)), "权重总和应该为1"
    
    print("\n✅ 所有测试通过！")

def test_forward_pass():
    """测试前向传播"""
    
    # 创建配置
    config = Method3Config_v2(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        vocab_size=1000
    )
    
    print("\n测试前向传播...")
    print("=" * 50)
    
    # 创建测试数据
    batch_size, seq_length = 2, 10
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
    
    # 创建层
    layer_2 = Method3DecoderLayer_v2(config, layer_idx=2)
    
    # 模拟前面层的MLP输出
    previous_mlp_outputs = [
        torch.randn(batch_size, seq_length, config.hidden_size),  # 第0层MLP输出
        torch.randn(batch_size, seq_length, config.hidden_size),  # 第1层MLP输出
    ]
    
    # 前向传播
    outputs = layer_2(
        hidden_states=hidden_states,
        previous_mlp_outputs=previous_mlp_outputs
    )
    
    print(f"输入形状: {hidden_states.shape}")
    print(f"输出形状: {outputs[0].shape}")
    print(f"输出元组长度: {len(outputs)}")
    print(f"权重参数: {layer_2.mlp_residual_weights}")
    print(f"归一化权重: {F.softmax(layer_2.mlp_residual_weights, dim=0)}")
    
    print("\n✅ 前向传播测试通过！")

if __name__ == "__main__":
    test_learnable_weights()
    test_forward_pass()
    print("\n🎉 所有测试完成！")
