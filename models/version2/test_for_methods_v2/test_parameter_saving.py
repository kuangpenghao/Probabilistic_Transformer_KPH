#!/usr/bin/env python3
"""
测试Method3_v2权重参数保存功能
"""

import torch
import torch.nn.functional as F
import sys
import os

# 添加项目根目录到路径
sys.path.append('/home/kuangph/hf-starter')

from models.version2.Method3_v2 import Method3LlamaForCausalLM_v2, Method3Config_v2

def test_parameter_saving():
    """测试权重参数保存功能"""
    
    print("测试权重参数保存功能...")
    print("=" * 60)
    
    # 创建配置
    config = Method3Config_v2(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        vocab_size=1000
    )
    
    # 创建模型
    model = Method3LlamaForCausalLM_v2(config)
    
    # 模拟训练后的权重（手动设置一些不同的权重值）
    print("设置模拟的训练后权重...")
    
    # 设置第1层的权重
    if hasattr(model.model.layers[1], 'mlp_residual_weights'):
        model.model.layers[1].mlp_residual_weights.data = torch.tensor([1.5, 0.8])
        print(f"第1层权重: {model.model.layers[1].mlp_residual_weights.data}")
    
    # 设置第2层的权重
    if hasattr(model.model.layers[2], 'mlp_residual_weights'):
        model.model.layers[2].mlp_residual_weights.data = torch.tensor([2.0, 1.0, 0.5])
        print(f"第2层权重: {model.model.layers[2].mlp_residual_weights.data}")
    
    # 设置第3层的权重
    if hasattr(model.model.layers[3], 'mlp_residual_weights'):
        model.model.layers[3].mlp_residual_weights.data = torch.tensor([0.5, 1.5, 2.5, 1.2])
        print(f"第3层权重: {model.model.layers[3].mlp_residual_weights.data}")
    
    print("\n保存权重参数...")
    
    # 保存权重参数到当前目录
    save_path = model.save_learned_parameters('/home/kuangph/hf-starter/models/version2')
    
    print(f"权重参数已保存到: {save_path}")
    
    # 验证文件是否创建成功
    if os.path.exists(save_path):
        print("✅ 文件创建成功!")
        
        # 读取并显示文件内容的前几行
        print("\n文件内容预览:")
        print("-" * 40)
        with open(save_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:20]):  # 显示前20行
                print(f"{i+1:2d}: {line.rstrip()}")
            if len(lines) > 20:
                print(f"... (文件共{len(lines)}行)")
    else:
        print("❌ 文件创建失败!")
    
    return save_path

def test_weight_analysis():
    """测试权重分析功能"""
    
    print("\n\n测试权重分析功能...")
    print("=" * 60)
    
    # 创建配置
    config = Method3Config_v2(
        hidden_size=32,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        vocab_size=500
    )
    
    # 创建模型
    model = Method3LlamaForCausalLM_v2(config)
    
    # 设置一些有趣的权重模式进行分析
    print("设置不同的权重模式...")
    
    # 第1层：比较均匀的权重
    if hasattr(model.model.layers[1], 'mlp_residual_weights'):
        model.model.layers[1].mlp_residual_weights.data = torch.tensor([1.1, 0.9])
        print(f"第1层 (均匀): {model.model.layers[1].mlp_residual_weights.data}")
    
    # 第2层：有明显偏好的权重
    if hasattr(model.model.layers[2], 'mlp_residual_weights'):
        model.model.layers[2].mlp_residual_weights.data = torch.tensor([0.2, 3.0, 0.5])
        print(f"第2层 (偏好): {model.model.layers[2].mlp_residual_weights.data}")
    
    # 保存并分析
    save_path = model.save_learned_parameters('/home/kuangph/hf-starter/models/version2')
    
    print(f"分析结果已保存到: {save_path}")
    
    return save_path

if __name__ == "__main__":
    # 测试基本保存功能
    save_path1 = test_parameter_saving()
    
    # 测试权重分析功能
    save_path2 = test_weight_analysis()
    
    print("\n🎉 所有测试完成!")
    print(f"可以查看保存的文件: {save_path1}")
    print(f"可以查看分析文件: {save_path2}")
