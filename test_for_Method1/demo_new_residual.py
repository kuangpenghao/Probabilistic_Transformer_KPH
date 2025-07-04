"""
新残差连接方式的完整演示
"""
import torch
from models.configuration_llama import MyLlamaConfig
from models.modeling_llama import MyLlamaForCausalLM
from models.Method1 import NewResidualDecoderLayer,NewResidualLlamaModel,NewResidualCausalLM

def demonstrate_new_residual():
    """演示新残差连接方式"""
    print("=" * 60)
    print("新残差连接方式演示")
    print("=" * 60)
    
    # 创建配置
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
    
    print(f"模型配置:")
    print(f"  - 词汇表大小: {config.vocab_size}")
    print(f"  - 隐藏层大小: {config.hidden_size}")
    print(f"  - 层数: {config.num_hidden_layers}")
    print(f"  - 注意力头数: {config.num_attention_heads}")
    
    # 创建模型
    print(f"\n创建模型...")
    new_model = NewResidualCausalLM(config)
    original_model = MyLlamaForCausalLM(config)
    
    # 计算参数数量
    new_params = sum(p.numel() for p in new_model.parameters())
    original_params = sum(p.numel() for p in original_model.parameters())
    
    print(f"新残差模型参数数量: {new_params:,}")
    print(f"原始模型参数数量: {original_params:,}")
    print(f"参数数量差异: {abs(new_params - original_params):,}")
    
    # 准备输入数据
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\n输入数据:")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 序列长度: {seq_len}")
    print(f"  - 输入形状: {input_ids.shape}")
    
    # 进行推理
    print(f"\n进行推理...")
    with torch.no_grad():
        # 新残差连接模型
        new_outputs = new_model(input_ids=input_ids)
        new_logits = new_outputs.logits
        
        # 原始模型
        original_outputs = original_model(input_ids=input_ids)
        original_logits = original_outputs.logits
        
        print(f"新残差模型输出形状: {new_logits.shape}")
        print(f"原始模型输出形状: {original_logits.shape}")
        
        # 计算输出差异
        logits_diff = torch.abs(new_logits - original_logits)
        mean_diff = logits_diff.mean().item()
        max_diff = logits_diff.max().item()
        
        print(f"\n输出差异分析:")
        print(f"  - 平均绝对差异: {mean_diff:.6f}")
        print(f"  - 最大绝对差异: {max_diff:.6f}")
        
        # 计算预测概率的差异
        new_probs = torch.softmax(new_logits, dim=-1)
        original_probs = torch.softmax(original_logits, dim=-1)
        prob_diff = torch.abs(new_probs - original_probs).mean().item()
        
        print(f"  - 预测概率平均差异: {prob_diff:.6f}")
        
        # 分析预测结果
        new_predictions = torch.argmax(new_logits, dim=-1)
        original_predictions = torch.argmax(original_logits, dim=-1)
        
        different_predictions = (new_predictions != original_predictions).sum().item()
        total_predictions = new_predictions.numel()
        
        print(f"\n预测结果分析:")
        print(f"  - 总预测数量: {total_predictions}")
        print(f"  - 不同预测数量: {different_predictions}")
        print(f"  - 预测差异率: {different_predictions/total_predictions*100:.2f}%")
        
        if different_predictions > 0:
            print(f"  ✓ 新残差连接产生了不同的预测结果")
        else:
            print(f"  ⚠ 所有预测结果相同")

def explain_residual_difference():
    """解释残差连接方式的差异"""
    print(f"\n" + "=" * 60)
    print("残差连接方式差异说明")
    print("=" * 60)
    
    print("""
原始残差连接方式:
  每层的注意力部分都使用当前层输入作为残差
  Layer_i: hidden_states = input_i + attention_output_i
  
新残差连接方式:
  - 第1层: hidden_states = attention_output_1 (无残差)
  - 第2层: hidden_states = attention_output_1 + attention_output_2
  - 第3层: hidden_states = attention_output_1 + attention_output_2 + attention_output_3
  - 第N层: hidden_states = sum(attention_output_1 to attention_output_{N-1}) + attention_output_N

关键差异:
1. 第一层没有残差连接，完全依赖注意力输出
2. 后续层的残差来自之前所有层的注意力输出累积
3. 这种方式让信息在层间以累积的方式传递
4. MLP部分仍保持原始的残差连接方式
    """)

if __name__ == "__main__":
    demonstrate_new_residual()
    explain_residual_difference()
    print(f"\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
