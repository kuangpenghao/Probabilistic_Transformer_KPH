#!/usr/bin/env python3
"""
Method3_v2 训练后权重保存示例
这个脚本展示了如何在模型训练完成后保存可学习的权重参数
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('/home/kuangph/hf-starter')

from models.version2.Method3_v2 import Method3LlamaForCausalLM_v2, Method3Config_v2

def simulate_training_and_save_weights():
    """模拟训练过程并保存权重参数"""
    
    print("🚀 开始模拟训练过程...")
    print("=" * 60)
    
    # 创建配置
    config = Method3Config_v2(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        vocab_size=1000,
        max_position_embeddings=512
    )
    
    # 创建模型
    print("📊 创建模型...")
    model = Method3LlamaForCausalLM_v2(config)
    
    # 显示初始权重
    print("\n📈 初始权重状态:")
    for layer_idx in range(config.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        if hasattr(layer, 'mlp_residual_weights') and layer.mlp_residual_weights is not None:
            print(f"  第{layer_idx}层: {layer.mlp_residual_weights.data.tolist()}")
    
    # 模拟训练过程 - 手动调整权重来模拟学习过程
    print("\n🎯 模拟训练过程...")
    
    # 模拟训练轮次
    for epoch in range(3):
        print(f"  训练轮次 {epoch + 1}/3")
        
        # 为不同层设置不同的学习模式
        if hasattr(model.model.layers[1], 'mlp_residual_weights'):
            # 第1层：逐渐偏向第二个位置
            model.model.layers[1].mlp_residual_weights.data = torch.tensor([1.0 - epoch*0.3, 1.0 + epoch*0.5])
        
        if hasattr(model.model.layers[2], 'mlp_residual_weights'):
            # 第2层：偏向中间位置
            model.model.layers[2].mlp_residual_weights.data = torch.tensor([1.0 + epoch*0.2, 1.0 + epoch*0.8, 1.0 + epoch*0.3])
        
        if hasattr(model.model.layers[3], 'mlp_residual_weights'):
            # 第3层：偏向最后位置
            model.model.layers[3].mlp_residual_weights.data = torch.tensor([1.0, 1.0 + epoch*0.1, 1.0 + epoch*0.2, 1.0 + epoch*1.0])
    
    # 显示训练后权重
    print("\n📊 训练后权重状态:")
    for layer_idx in range(config.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        if hasattr(layer, 'mlp_residual_weights') and layer.mlp_residual_weights is not None:
            raw_weights = layer.mlp_residual_weights.data
            normalized = torch.nn.functional.softmax(raw_weights, dim=0)
            print(f"  第{layer_idx}层: 原始={raw_weights.tolist()}")
            print(f"           归一化={normalized.tolist()}")
    
    # 训练完成，保存权重参数
    print("\n💾 训练完成，保存可学习权重参数...")
    save_path = model.save_learned_parameters('/home/kuangph/hf-starter/models/version2')
    
    print(f"✅ 权重参数已保存到: {save_path}")
    
    return save_path

def analyze_saved_weights(save_path):
    """分析保存的权重文件"""
    
    print("\n🔍 分析保存的权重文件...")
    print("=" * 60)
    
    if not os.path.exists(save_path):
        print("❌ 文件不存在!")
        return
    
    with open(save_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📄 文件大小: {len(content)} 字符")
    print(f"📄 文件行数: {len(content.split(chr(10)))}")
    
    # 提取关键信息
    lines = content.split('\n')
    
    print("\n🎯 关键信息提取:")
    
    # 找到权重学习趋势分析
    in_trend_analysis = False
    for line in lines:
        if "权重学习趋势分析:" in line:
            in_trend_analysis = True
            print("  权重学习趋势:")
            continue
        elif in_trend_analysis and line.strip().startswith("第"):
            print(f"    {line.strip()}")
        elif in_trend_analysis and line.strip() == "":
            break
    
    print("\n📊 文件内容预览:")
    print("-" * 40)
    for i, line in enumerate(lines[:15]):
        print(f"{i+1:2d}: {line}")
    print("...")

if __name__ == "__main__":
    print("Method3_v2 权重保存功能演示")
    print("=" * 80)
    
    # 模拟训练并保存权重
    save_path = simulate_training_and_save_weights()
    
    # 分析保存的权重
    analyze_saved_weights(save_path)
    
    print("\n🎉 演示完成!")
    print(f"💡 您可以查看完整的权重参数文件: {save_path}")
    print("💡 在实际训练中，只需在训练完成后调用 model.save_learned_parameters() 即可")
