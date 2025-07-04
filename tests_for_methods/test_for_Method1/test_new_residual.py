"""
测试新残差连接方式的脚本
"""
import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.configuration_llama import Method1LlamaConfig
from models.Method1 import Method1DecoderLayer, Method1LlamaModel, Method1LlamaForCausalLM

def test_new_residual_model():
    """测试新残差连接模型"""
    print("测试新残差连接模型...")
    
    # 创建一个小的配置用于测试
    config = Method1LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
    )
    
    # 创建模型
    print("创建新残差连接模型...")
    new_model = Method1LlamaForCausalLM(config)
    
    print("创建原始模型...")
    # 注释掉原始模型比较，专注于Method1测试
    # original_model = Method1LlamaForCausalLM(config)
    
    # 准备输入
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"输入形状: {input_ids.shape}")
    
    # 测试前向传播
    print("测试新残差连接模型前向传播...")
    with torch.no_grad():
        new_outputs = new_model(input_ids=input_ids)
        print(f"新模型输出logits形状: {new_outputs.logits.shape}")
        
        print("测试Method1模型前向传播...")
        new_outputs = new_model(input_ids=input_ids)
        print(f"Method1模型输出logits形状: {new_outputs.logits.shape}")
        
        # 简单验证输出形状和数值范围
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert new_outputs.logits.shape == expected_shape, f"输出形状不匹配: {new_outputs.logits.shape} vs {expected_shape}"
        
        # 检查输出是否包含有效数值（不是NaN或Inf）
        assert torch.isfinite(new_outputs.logits).all(), "输出包含无效数值(NaN或Inf)"
        
        print("✓ Method1模型输出验证通过")
    
    print("模型测试完成！")

def analyze_residual_connections():
    """分析残差连接的工作方式"""
    print("\n分析残差连接...")
    
    config = Method1LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,  # 使用3层便于分析
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
    )
    
    model = Method1LlamaForCausalLM(config)
    
    # 创建输入
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    
    print("模型层数:", len(model.model.layers))
    print("每层的layer_idx:", [layer.layer_idx for layer in model.model.layers])
    
    # 手动分析一下前向传播过程
    print("\n残差连接分析:")
    print("- 第0层: 没有残差连接，hidden_states = attn_output")
    print("- 第1层: hidden_states = 第0层的注意力输出 + 当前层的attn_output")
    print("- 第2层: hidden_states = (第0层 + 第1层)的注意力输出之和 + 当前层的attn_output")
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        print(f"\n最终输出形状: {outputs.logits.shape}")

if __name__ == "__main__":
    test_new_residual_model()
    analyze_residual_connections()
