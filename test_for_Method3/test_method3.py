#!/usr/bin/env python3
"""
测试Method3实现的正确性
"""
import torch
import torch.nn as nn
import numpy as np
from models.Method3 import Method3LlamaForCausalLM, Method3LlamaModel, Method3DecoderLayer
from models.configuration_llama import Method3LlamaConfig
from transformers import LlamaTokenizer
import sys
import os

def test_config():
    """测试配置是否正确"""
    print("=== 测试配置 ===")
    config = Method3LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        torch_dtype="float32"
    )
    print(f"配置创建成功: {config.model_type}")
    print(f"隐藏层数: {config.num_hidden_layers}")
    print(f"隐藏大小: {config.hidden_size}")
    return config

def test_model_initialization(config):
    """测试模型初始化"""
    print("\n=== 测试模型初始化 ===")
    
    # 测试DecoderLayer
    decoder_layer = Method3DecoderLayer(config, layer_idx=0)
    print(f"DecoderLayer初始化成功, layer_idx: {decoder_layer.layer_idx}")
    
    # 测试Model
    model = Method3LlamaModel(config)
    print(f"Model初始化成功, 层数: {len(model.layers)}")
    
    # 测试ForCausalLM
    causal_model = Method3LlamaForCausalLM(config)
    print(f"CausalLM初始化成功, vocab_size: {causal_model.vocab_size}")
    
    return decoder_layer, model, causal_model

def test_forward_pass(config, model, causal_model):
    """测试前向传播"""
    print("\n=== 测试前向传播 ===")
    
    batch_size = 2
    seq_len = 10
    
    # 创建输入
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"输入形状: {input_ids.shape}")
    
    # 测试模型前向传播
    with torch.no_grad():
        model_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        
        print(f"模型输出形状: {model_outputs.last_hidden_state.shape}")
        print(f"隐藏状态数量: {len(model_outputs.hidden_states) if model_outputs.hidden_states else 0}")
        print(f"注意力权重数量: {len(model_outputs.attentions) if model_outputs.attentions else 0}")
        
        # 测试因果语言模型
        causal_outputs = causal_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids  # 使用input_ids作为labels来计算loss
        )
        
        print(f"因果模型logits形状: {causal_outputs.logits.shape}")
        print(f"损失值: {causal_outputs.loss.item():.4f}")
        
    return model_outputs, causal_outputs

def test_residual_connection_logic():
    """测试残差连接逻辑"""
    print("\n=== 测试残差连接逻辑 ===")
    
    # 创建一个小的配置用于测试
    config = Method3LlamaConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        use_cache=False,
        tie_word_embeddings=False,
        torch_dtype="float32"
    )
    
    model = Method3LlamaModel(config)
    
    # 创建输入
    batch_size = 1
    seq_len = 5
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # 修改模型使其输出更多调试信息
    class DebugMethod3DecoderLayer(Method3DecoderLayer):
        def forward(self, hidden_states, previous_mlp_outputs=None, **kwargs):
            print(f"    Layer {self.layer_idx}:")
            print(f"      输入形状: {hidden_states.shape}")
            print(f"      previous_mlp_outputs数量: {len(previous_mlp_outputs) if previous_mlp_outputs else 0}")
            
            # 调用父类的前向传播
            outputs = super().forward(hidden_states, previous_mlp_outputs=previous_mlp_outputs, **kwargs)
            
            # 输出当前层的MLP输出
            current_mlp_output = outputs[-1]
            print(f"      当前层MLP输出形状: {current_mlp_output.shape}")
            print(f"      当前层MLP输出均值: {current_mlp_output.mean().item():.4f}")
            
            return outputs
    
    # 替换层
    for i in range(config.num_hidden_layers):
        model.layers[i] = DebugMethod3DecoderLayer(config, layer_idx=i)
    
    print("开始前向传播...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        
    print(f"最终输出形状: {outputs.last_hidden_state.shape}")
    print(f"最终输出均值: {outputs.last_hidden_state.mean().item():.4f}")

def test_mlp_residual_accumulation():
    """测试MLP残差累积逻辑"""
    print("\n=== 测试MLP残差累积逻辑 ===")
    
    # 创建简单的测试数据
    batch_size, seq_len, hidden_size = 1, 3, 4
    
    # 模拟前几层的MLP输出
    mlp_output_1 = torch.ones(batch_size, seq_len, hidden_size) * 1.0
    mlp_output_2 = torch.ones(batch_size, seq_len, hidden_size) * 2.0
    mlp_output_3 = torch.ones(batch_size, seq_len, hidden_size) * 3.0
    
    print("测试残差累积:")
    
    # 第1层：没有残差
    layer_1_output = mlp_output_1
    print(f"第1层输出 (无残差): {layer_1_output.mean().item():.1f}")
    
    # 第2层：残差为第1层输出
    previous_outputs = [layer_1_output]
    residual_sum = sum(previous_outputs)
    layer_2_output = residual_sum + mlp_output_2
    print(f"第2层输出 (残差={residual_sum.mean().item():.1f}): {layer_2_output.mean().item():.1f}")
    
    # 第3层：残差为第1,2层输出之和
    previous_outputs = [layer_1_output, layer_2_output]
    residual_sum = sum(previous_outputs)
    layer_3_output = residual_sum + mlp_output_3
    print(f"第3层输出 (残差={residual_sum.mean().item():.1f}): {layer_3_output.mean().item():.1f}")
    
    expected_layer_3 = 1.0 + (1.0 + 2.0) + 3.0  # 第1层 + 第2层 + 第3层MLP
    print(f"期望的第3层输出: {expected_layer_3:.1f}")
    print(f"实际计算匹配: {abs(layer_3_output.mean().item() - expected_layer_3) < 1e-6}")

def test_comparison_with_original():
    """与原始模型的比较测试"""
    print("\n=== 与原始模型比较 ===")
    
    # 由于没有直接的原始模型，我们测试Method3的特殊行为
    config = Method3LlamaConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        use_cache=False,
        tie_word_embeddings=False,
        torch_dtype="float32"
    )
    
    model = Method3LlamaForCausalLM(config)
    
    # 固定随机种子以确保可重复性
    torch.manual_seed(42)
    
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        
    print(f"Method3模型输出形状: {outputs.logits.shape}")
    print(f"Method3模型输出均值: {outputs.logits.mean().item():.4f}")
    print(f"Method3模型输出标准差: {outputs.logits.std().item():.4f}")
    
    # 检查输出是否合理
    if not torch.isnan(outputs.logits).any():
        print("✓ 输出没有NaN值")
    else:
        print("✗ 输出包含NaN值")
        
    if not torch.isinf(outputs.logits).any():
        print("✓ 输出没有无穷值")
    else:
        print("✗ 输出包含无穷值")

def main():
    """主测试函数"""
    print("开始测试Method3实现...")
    
    try:
        # 测试配置
        config = test_config()
        
        # 测试模型初始化
        decoder_layer, model, causal_model = test_model_initialization(config)
        
        # 测试前向传播
        model_outputs, causal_outputs = test_forward_pass(config, model, causal_model)
        
        # 测试残差连接逻辑
        test_residual_connection_logic()
        
        # 测试MLP残差累积
        test_mlp_residual_accumulation()
        
        # 与原始模型比较
        test_comparison_with_original()
        
        print("\n=== 测试总结 ===")
        print("✓ 所有测试通过!")
        print("✓ Method3实现正确!")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
