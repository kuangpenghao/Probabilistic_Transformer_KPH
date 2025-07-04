"""
测试新残差连接方式的脚本
"""
import torch
from models.configuration_llama import MyLlamaConfig
from models.modeling_llama import MyLlamaForCausalLM
from models.Method1 import NewResidualDecoderLayer,NewResidualLlamaModel,NewResidualCausalLM

def test_new_residual_model():
    """测试新残差连接模型"""
    print("测试新残差连接模型...")
    
    # 创建一个小的配置用于测试
    config = MyLlamaConfig(
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
    new_model = NewResidualCausalLM(config)
    
    print("创建原始模型...")
    original_model = MyLlamaForCausalLM(config)
    
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
        
        print("测试原始模型前向传播...")
        original_outputs = original_model(input_ids=input_ids)
        print(f"原始模型输出logits形状: {original_outputs.logits.shape}")
        
        # 比较输出差异
        logits_diff = torch.abs(new_outputs.logits - original_outputs.logits).mean()
        print(f"输出差异 (平均绝对差): {logits_diff:.6f}")
        
        # 验证输出不完全相同（因为残差连接方式不同）
        if logits_diff > 1e-6:
            print("✓ 新残差连接产生了不同的输出，符合预期")
        else:
            print("⚠ 输出相同，可能存在问题")
    
    print("模型测试完成！")

def analyze_residual_connections():
    """分析残差连接的工作方式"""
    print("\n分析残差连接...")
    
    config = MyLlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,  # 使用3层便于分析
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
    )
    
    model = NewResidualCausalLM(config)
    
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
