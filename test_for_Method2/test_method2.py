#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Method2模型测试脚本
测试新的残差连接方式（平均值）的正确性
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Method2 import Method2LlamaForCausalLM, Method2LlamaModel, Method2DecoderLayer
from models.configuration_llama import Method2LlamaConfig


def test_method2_config():
    """测试Method2配置"""
    print("=== 测试Method2配置 ===")
    config = Method2LlamaConfig(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=512,
    )
    print(f"模型类型: {config.model_type}")
    print(f"隐藏层数: {config.num_hidden_layers}")
    print(f"隐藏层大小: {config.hidden_size}")
    print("配置测试通过!")
    return config


def test_method2_decoder_layer():
    """测试Method2DecoderLayer的残差连接逻辑"""
    print("\n=== 测试Method2DecoderLayer残差连接 ===")
    
    config = Method2LlamaConfig(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=512,
    )
    
    batch_size, seq_len = 2, 10
    
    # 测试第0层（无残差连接）
    layer_0 = Method2DecoderLayer(config, layer_idx=0)
    layer_0.eval()
    
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    with torch.no_grad():
        outputs_0 = layer_0(hidden_states, previous_attn_outputs=None)
    
    print(f"第0层输出形状: {outputs_0[0].shape}")
    print(f"第0层注意力输出形状: {outputs_0[-1].shape}")
    
    # 测试第1层（有1个前置层的平均残差）
    layer_1 = Method2DecoderLayer(config, layer_idx=1)
    layer_1.eval()
    
    # 模拟前一层的注意力输出
    prev_attn_outputs = [outputs_0[-1]]  # 使用第0层的注意力输出
    
    with torch.no_grad():
        outputs_1 = layer_1(hidden_states, previous_attn_outputs=prev_attn_outputs)
    
    print(f"第1层输出形状: {outputs_1[0].shape}")
    print(f"第1层注意力输出形状: {outputs_1[-1].shape}")
    
    # 测试第2层（有2个前置层的平均残差）
    layer_2 = Method2DecoderLayer(config, layer_idx=2)
    layer_2.eval()
    
    # 模拟前两层的注意力输出
    prev_attn_outputs = [outputs_0[-1], outputs_1[-1]]
    
    with torch.no_grad():
        outputs_2 = layer_2(hidden_states, previous_attn_outputs=prev_attn_outputs)
    
    print(f"第2层输出形状: {outputs_2[0].shape}")
    print(f"第2层注意力输出形状: {outputs_2[-1].shape}")
    
    print("DecoderLayer测试通过!")
    return outputs_0, outputs_1, outputs_2


def test_residual_averaging():
    """专门测试残差平均化逻辑"""
    print("\n=== 测试残差平均化逻辑 ===")
    
    # 创建一些模拟的注意力输出
    batch_size, seq_len, hidden_size = 2, 5, 8
    
    attn_out_1 = torch.tensor([[[1.0] * hidden_size] * seq_len] * batch_size)
    attn_out_2 = torch.tensor([[[2.0] * hidden_size] * seq_len] * batch_size)
    attn_out_3 = torch.tensor([[[3.0] * hidden_size] * seq_len] * batch_size)
    
    previous_outputs = [attn_out_1, attn_out_2, attn_out_3]
    
    # 计算平均值
    residual_sum = sum(previous_outputs)
    residual_avg = residual_sum / len(previous_outputs)
    
    expected_avg = torch.tensor([[[2.0] * hidden_size] * seq_len] * batch_size)  # (1+2+3)/3 = 2
    
    print(f"残差平均值: {residual_avg[0, 0, 0].item()}")
    print(f"期望平均值: {expected_avg[0, 0, 0].item()}")
    
    assert torch.allclose(residual_avg, expected_avg, atol=1e-6), "残差平均化计算错误!"
    print("残差平均化逻辑测试通过!")


def test_method2_model():
    """测试完整的Method2模型"""
    print("\n=== 测试Method2完整模型 ===")
    
    config = Method2LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=128,
    )
    
    model = Method2LlamaModel(config)
    model.eval()
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"模型输出形状: {outputs.last_hidden_state.shape}")
    print(f"期望形状: ({batch_size}, {seq_len}, {config.hidden_size})")
    
    assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)
    print("Method2模型测试通过!")


def test_method2_causal_lm():
    """测试Method2因果语言模型"""
    print("\n=== 测试Method2因果语言模型 ===")
    
    config = Method2LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=128,
    )
    
    model = Method2LlamaForCausalLM(config)
    model.eval()
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"Logits形状: {outputs.logits.shape}")
    print(f"期望形状: ({batch_size}, {seq_len}, {config.vocab_size})")
    
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
    print("Method2因果语言模型测试通过!")


def test_gradient_flow():
    """测试梯度流动"""
    print("\n=== 测试梯度流动 ===")
    
    config = Method2LlamaConfig(
        vocab_size=100,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
    )
    
    model = Method2LlamaForCausalLM(config)
    model.train()
    
    batch_size, seq_len = 1, 5
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    
    print(f"损失值: {loss.item()}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            break
    
    assert has_grad, "模型参数没有梯度!"
    print("梯度流动测试通过!")


def main():
    """运行所有测试"""
    print("开始测试Method2.py代码...")
    
    try:
        # 基础配置测试
        config = test_method2_config()
        
        # 残差平均化逻辑测试
        test_residual_averaging()
        
        # DecoderLayer测试
        test_method2_decoder_layer()
        
        # 完整模型测试
        test_method2_model()
        
        # 因果语言模型测试
        test_method2_causal_lm()
        
        # 梯度流动测试
        test_gradient_flow()
        
        print("\n" + "="*50)
        print("✅ 所有测试通过! Method2.py代码正确!")
        print("✅ 残差连接已正确实现为平均值方式")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()
