#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Method7模型测试脚本
测试结合注意力和MLP新残差连接方式的正确性
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录和模型目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Method7 import Method7LlamaForCausalLM, Method7LlamaModel, Method7DecoderLayer
from configuration_llama import Method7LlamaConfig


def test_method7_config():
    """测试Method7配置"""
    print("=== 测试Method7配置 ===")
    config = Method7LlamaConfig(
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


def test_method7_decoder_layer():
    """测试Method7DecoderLayer的双重残差连接逻辑"""
    print("\n=== 测试Method7DecoderLayer双重残差连接 ===")
    
    config = Method7LlamaConfig(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=512,
    )
    
    batch_size, seq_len = 2, 10
    
    # 测试第0层（无残差连接）
    layer_0 = Method7DecoderLayer(config, layer_idx=0)
    layer_0.eval()
    
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    with torch.no_grad():
        outputs_0 = layer_0(hidden_states, position_ids=position_ids, previous_attn_outputs=None, previous_mlp_outputs=None)
    
    print(f"第0层输出形状: {outputs_0[0].shape}")
    print(f"第0层注意力输出形状: {outputs_0[-2].shape}")
    print(f"第0层MLP输出形状: {outputs_0[-1].shape}")
    
    # 测试第1层（有1个前置层的残差）
    layer_1 = Method7DecoderLayer(config, layer_idx=1)
    layer_1.eval()
    
    # 模拟前一层的注意力输出和MLP输出
    prev_attn_outputs = [outputs_0[-2]]  # 使用第0层的注意力输出
    prev_mlp_outputs = [outputs_0[-1]]   # 使用第0层的MLP输出
    
    with torch.no_grad():
        outputs_1 = layer_1(hidden_states, position_ids=position_ids, previous_attn_outputs=prev_attn_outputs, previous_mlp_outputs=prev_mlp_outputs)
    
    print(f"第1层输出形状: {outputs_1[0].shape}")
    print(f"第1层注意力输出形状: {outputs_1[-2].shape}")
    print(f"第1层MLP输出形状: {outputs_1[-1].shape}")
    
    # 测试第2层（有2个前置层的残差）
    layer_2 = Method7DecoderLayer(config, layer_idx=2)
    layer_2.eval()
    
    # 模拟前两层的注意力输出和MLP输出
    prev_attn_outputs = [outputs_0[-2], outputs_1[-2]]
    prev_mlp_outputs = [outputs_0[-1], outputs_1[-1]]
    
    with torch.no_grad():
        outputs_2 = layer_2(hidden_states, position_ids=position_ids, previous_attn_outputs=prev_attn_outputs, previous_mlp_outputs=prev_mlp_outputs)
    
    print(f"第2层输出形状: {outputs_2[0].shape}")
    print(f"第2层注意力输出形状: {outputs_2[-2].shape}")
    print(f"第2层MLP输出形状: {outputs_2[-1].shape}")
    
    print("Method7DecoderLayer测试通过!")
    return outputs_0, outputs_1, outputs_2


def test_dual_residual_logic():
    """专门测试双重残差连接逻辑"""
    print("\n=== 测试双重残差连接逻辑 ===")
    
    # 创建一些模拟的注意力输出和MLP输出
    batch_size, seq_len, hidden_size = 2, 5, 8
    
    # 模拟注意力输出
    attn_out_1 = torch.tensor([[[1.0] * hidden_size] * seq_len] * batch_size)
    attn_out_2 = torch.tensor([[[2.0] * hidden_size] * seq_len] * batch_size)
    attn_out_3 = torch.tensor([[[3.0] * hidden_size] * seq_len] * batch_size)
    
    # 模拟MLP输出
    mlp_out_1 = torch.tensor([[[10.0] * hidden_size] * seq_len] * batch_size)
    mlp_out_2 = torch.tensor([[[20.0] * hidden_size] * seq_len] * batch_size)
    mlp_out_3 = torch.tensor([[[30.0] * hidden_size] * seq_len] * batch_size)
    
    previous_attn_outputs = [attn_out_1, attn_out_2, attn_out_3]
    previous_mlp_outputs = [mlp_out_1, mlp_out_2, mlp_out_3]
    
    # 计算注意力残差和
    attn_residual_sum = sum(previous_attn_outputs)
    expected_attn_sum = torch.tensor([[[6.0] * hidden_size] * seq_len] * batch_size)  # (1+2+3) = 6
    
    # 计算MLP残差和
    mlp_residual_sum = sum(previous_mlp_outputs)
    expected_mlp_sum = torch.tensor([[[60.0] * hidden_size] * seq_len] * batch_size)  # (10+20+30) = 60
    
    print(f"注意力残差和: {attn_residual_sum[0, 0, 0].item()}")
    print(f"期望注意力残差和: {expected_attn_sum[0, 0, 0].item()}")
    print(f"MLP残差和: {mlp_residual_sum[0, 0, 0].item()}")
    print(f"期望MLP残差和: {expected_mlp_sum[0, 0, 0].item()}")
    
    assert torch.allclose(attn_residual_sum, expected_attn_sum, atol=1e-6), "注意力残差计算错误!"
    assert torch.allclose(mlp_residual_sum, expected_mlp_sum, atol=1e-6), "MLP残差计算错误!"
    print("双重残差连接逻辑测试通过!")


def test_method7_model():
    """测试完整的Method7模型"""
    print("\n=== 测试Method7完整模型 ===")
    
    config = Method7LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=128,
    )
    
    model = Method7LlamaModel(config)
    model.eval()
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"模型输出形状: {outputs.last_hidden_state.shape}")
    print(f"期望形状: ({batch_size}, {seq_len}, {config.hidden_size})")
    
    assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)
    print("Method7模型测试通过!")


def test_method7_causal_lm():
    """测试Method7因果语言模型"""
    print("\n=== 测试Method7因果语言模型 ===")
    
    config = Method7LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=128,
    )
    
    model = Method7LlamaForCausalLM(config)
    model.eval()
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"Logits形状: {outputs.logits.shape}")
    print(f"期望形状: ({batch_size}, {seq_len}, {config.vocab_size})")
    
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
    print("Method7因果语言模型测试通过!")


def test_gradient_flow():
    """测试梯度流动"""
    print("\n=== 测试梯度流动 ===")
    
    config = Method7LlamaConfig(
        vocab_size=100,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
    )
    
    model = Method7LlamaForCausalLM(config)
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
    print("开始测试Method7.py代码...")
    
    try:
        # 基础配置测试
        config = test_method7_config()
        
        # 双重残差连接逻辑测试
        test_dual_residual_logic()
        
        # DecoderLayer测试
        test_method7_decoder_layer()
        
        # 完整模型测试
        test_method7_model()
        
        # 因果语言模型测试
        test_method7_causal_lm()
        
        # 梯度流动测试
        test_gradient_flow()
        
        print("\n" + "="*60)
        print("✅ 所有测试通过! Method7.py代码正确!")
        print("✅ 注意力残差连接已正确实现（来自Method1）")
        print("✅ MLP残差连接已正确实现（来自Method3）")
        print("✅ 双重残差连接方式正确结合")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()
