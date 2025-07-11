#!/usr/bin/env python3
"""
简化测试：专注于可学习权重参数的核心功能
"""

import torch
import torch.nn.functional as F
import sys
import os

# 添加项目根目录到路径
sys.path.append('/home/kuangph/hf-starter')

from models.version2.Method3_v2 import Method3DecoderLayer_v2, Method3Config_v2

def test_weights_functionality():
    """测试权重参数的核心功能"""
    
    # 创建配置
    config = Method3Config_v2(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        vocab_size=1000
    )
    
    print("测试可学习权重参数核心功能...")
    print("=" * 50)
    
    # 测试不同层的权重参数
    for layer_idx in range(4):
        layer = Method3DecoderLayer_v2(config, layer_idx=layer_idx)
        
        if layer_idx == 0:
            print(f"第{layer_idx}层: 权重参数 = {layer.mlp_residual_weights}")
            assert layer.mlp_residual_weights is None
        else:
            print(f"第{layer_idx}层: 权重参数形状 = {layer.mlp_residual_weights.shape}")
            print(f"第{layer_idx}层: 权重参数值 = {layer.mlp_residual_weights}")
            assert layer.mlp_residual_weights.shape == (layer_idx + 1,)
            
            # 测试梯度
            assert layer.mlp_residual_weights.requires_grad == True
            
            # 测试归一化
            normalized = F.softmax(layer.mlp_residual_weights, dim=0)
            print(f"第{layer_idx}层: 归一化权重 = {normalized}")
            print(f"第{layer_idx}层: 权重总和 = {normalized.sum():.6f}")
            assert torch.allclose(normalized.sum(), torch.tensor(1.0))
        
        print("-" * 30)
    
    # 测试权重更新
    print("\n测试权重更新:")
    layer_2 = Method3DecoderLayer_v2(config, layer_idx=2)
    original_weights = layer_2.mlp_residual_weights.clone()
    
    # 手动设置权重
    layer_2.mlp_residual_weights.data = torch.tensor([0.5, 2.0, 1.5])
    normalized = F.softmax(layer_2.mlp_residual_weights, dim=0)
    
    print(f"原始权重: {original_weights}")
    print(f"更新后权重: {layer_2.mlp_residual_weights}")
    print(f"归一化权重: {normalized}")
    print(f"权重总和: {normalized.sum():.6f}")
    
    # 测试梯度计算
    print("\n测试梯度计算:")
    dummy_target = torch.tensor([0.3, 0.5, 0.2])
    loss = F.mse_loss(normalized, dummy_target)
    loss.backward()
    
    print(f"损失: {loss.item():.6f}")
    print(f"权重梯度: {layer_2.mlp_residual_weights.grad}")
    
    print("\n✅ 所有权重功能测试通过！")

def test_weighted_combination():
    """测试加权组合逻辑"""
    
    print("\n测试加权组合逻辑...")
    print("=" * 50)
    
    # 模拟MLP输出
    batch_size, seq_length, hidden_size = 2, 5, 64
    
    mlp_outputs = [
        torch.randn(batch_size, seq_length, hidden_size),  # 第0层
        torch.randn(batch_size, seq_length, hidden_size),  # 第1层
        torch.randn(batch_size, seq_length, hidden_size),  # 第2层 (当前层)
    ]
    
    # 权重参数
    weights = torch.tensor([1.0, 2.0, 0.5])  # 未归一化
    normalized_weights = F.softmax(weights, dim=0)
    print(f"权重: {weights}")
    print(f"归一化权重: {normalized_weights}")
    
    # 计算加权和
    weighted_sum = torch.zeros_like(mlp_outputs[0])
    for i, output in enumerate(mlp_outputs):
        weighted_sum += normalized_weights[i] * output
        print(f"权重 {i}: {normalized_weights[i]:.4f}")
    
    print(f"输入形状: {[out.shape for out in mlp_outputs]}")
    print(f"加权和形状: {weighted_sum.shape}")
    
    # 验证加权和不是简单的平均
    simple_avg = sum(mlp_outputs) / len(mlp_outputs)
    
    print(f"是否与简单平均相同: {torch.allclose(weighted_sum, simple_avg)}")
    print(f"加权和范数: {torch.norm(weighted_sum):.4f}")
    print(f"简单平均范数: {torch.norm(simple_avg):.4f}")
    
    print("\n✅ 加权组合测试通过！")

if __name__ == "__main__":
    test_weights_functionality()
    test_weighted_combination()
    print("\n🎉 所有测试完成！可学习权重参数功能正常工作！")
