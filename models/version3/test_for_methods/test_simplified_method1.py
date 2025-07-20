#!/usr/bin/env python3
"""
测试简化版Method1_v3的实现
验证attn_weights存储方案是否正常工作
"""
import os
import sys
import torch
import torch.nn as nn

# 添加模型路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.version3.Method1_v3 import Method1LlamaAttention_v3, Method1DecoderLayer_v3
from models.version3.configuration_llama_v3 import Method1Config_v3

def test_attention_with_attn_weights():
    """测试基于attn_weights的注意力计算"""
    print("=== 测试简化版Attention（基于attn_weights存储）===")
    
    # 创建配置
    config = Method1Config_v3(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=256,
        num_hidden_layers=2,
        vocab_size=1000,
    )
    
    # 创建注意力层
    attn = Method1LlamaAttention_v3(config, layer_idx=0)
    
    # 创建测试输入
    batch_size = 2
    seq_len = 10
    hidden_size = 128
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # 创建位置嵌入
    cos = torch.randn(batch_size, seq_len, hidden_size // config.num_attention_heads)
    sin = torch.randn(batch_size, seq_len, hidden_size // config.num_attention_heads)
    position_embeddings = (cos, sin)
    
    print(f"输入形状: {hidden_states.shape}")
    
    try:
        # 第一次前向传播（获取attn_weights和V权重）
        result = attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            output_attentions=True
        )
        
        print(f"注意力输出形状: {result[0].shape}")
        print(f"存储的attn_weights形状: {result[3].shape if result[3] is not None else None}")
        print(f"V权重形状: {result[4].shape if result[4] is not None else None}")
        
        # 使用存储的权重重新计算（模拟重算过程）
        if result[3] is not None and result[4] is not None:
            # 使用不同的输入（模拟当前层的新输入）
            new_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
            
            recomputed_output = attn.forward_with_precomputed_weights(
                hidden_states=new_hidden_states,
                attn_weights=result[3],
                v_proj_weight=result[4]
            )
            
            print(f"重新计算的输出形状: {recomputed_output.shape}")
            print("✅ attn_weights存储方案测试成功!")
        else:
            print("❌ 未能正确获取attn_weights或V权重")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_decoder_layer():
    """测试简化版DecoderLayer"""
    print("\n=== 测试简化版DecoderLayer ===")
    
    # 创建配置
    config = Method1Config_v3(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=256,
        num_hidden_layers=2,
        vocab_size=1000,
    )
    
    # 创建解码器层
    layer = Method1DecoderLayer_v3(config, layer_idx=0)
    
    # 创建测试输入
    batch_size = 2
    seq_len = 10
    hidden_size = 128
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # 创建位置嵌入
    cos = torch.randn(batch_size, seq_len, hidden_size // config.num_attention_heads)
    sin = torch.randn(batch_size, seq_len, hidden_size // config.num_attention_heads)
    position_embeddings = (cos, sin)
    
    print(f"输入形状: {hidden_states.shape}")
    
    try:
        # 前向传播
        outputs = layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            output_attentions=True,
            previous_mlp_outputs=[],  # 第一层没有前面的MLP输出
            current_layer_input=hidden_states
        )
        
        print(f"层输出形状: {outputs[0].shape}")
        print(f"存储的权重信息: {list(outputs[-1].keys())}")
        print("✅ DecoderLayer测试成功!")
        
    except Exception as e:
        print(f"❌ DecoderLayer测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("开始测试简化版Method1_v3实现...")
    
    # 设置随机种子保证可重现性
    torch.manual_seed(42)
    
    test_attention_with_attn_weights()
    test_decoder_layer()
    
    print("\n所有测试完成!")
