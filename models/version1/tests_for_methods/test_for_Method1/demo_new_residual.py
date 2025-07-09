"""
Method1残差连接方式的完整演示
"""
import torch
import sys
import os

# 添加项目根目录和模型目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from configuration_llama import Method1LlamaConfig
from Method1 import Method1DecoderLayer, Method1LlamaModel, Method1LlamaForCausalLM

def demonstrate_new_residual():
    """演示Method1残差连接方式"""
    print("=" * 60)
    print("Method1残差连接方式演示")
    print("=" * 60)
    
    # 创建配置
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
    
    print(f"模型配置:")
    print(f"  - 词汇表大小: {config.vocab_size}")
    print(f"  - 隐藏层大小: {config.hidden_size}")
    print(f"  - 层数: {config.num_hidden_layers}")
    print(f"  - 注意力头数: {config.num_attention_heads}")
    
    # 创建模型
    print(f"\n创建模型...")
    method1_model = Method1LlamaForCausalLM(config)
    
    # 计算参数数量
    method1_params = sum(p.numel() for p in method1_model.parameters())
    
    print(f"Method1模型参数数量: {method1_params:,}")
    
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
        # Method1模型推理
        method1_outputs = method1_model(input_ids=input_ids)
        method1_logits = method1_outputs.logits
        
        print(f"Method1模型输出形状: {method1_logits.shape}")
        
        # 检查输出是否包含有效数值
        assert torch.isfinite(method1_logits).all(), "Method1模型输出包含无效数值"
        
        # 计算输出统计信息
        mean_logits = method1_logits.mean().item()
        std_logits = method1_logits.std().item()
        max_logits = method1_logits.max().item()
        min_logits = method1_logits.min().item()
        
        print(f"\nMethod1模型输出统计:")
        print(f"  - 平均值: {mean_logits:.6f}")
        print(f"  - 标准差: {std_logits:.6f}")
        print(f"  - 最大值: {max_logits:.6f}")
        print(f"  - 最小值: {min_logits:.6f}")
        
        # 计算预测概率
        method1_probs = torch.softmax(method1_logits, dim=-1)
        
        # 分析预测结果
        method1_predictions = torch.argmax(method1_logits, dim=-1)
        
        print(f"\n预测结果分析:")
        print(f"  - 预测shape: {method1_predictions.shape}")
        print(f"  - 前5个预测: {method1_predictions[0, :5].tolist()}")
        
        # 检查概率分布
        max_probs = method1_probs.max(dim=-1)[0]
        print(f"  - 最大概率平均值: {max_probs.mean().item():.6f}")
        print(f"  ✓ Method1模型运行正常")

def explain_residual_difference():
    """解释Method1残差连接方式的特点"""
    print(f"\n" + "=" * 60)
    print("Method1残差连接方式说明")
    print("=" * 60)
    
    print("""
Method1残差连接方式:
  每层的注意力部分累积之前所有层的注意力输出
  - 第0层: hidden_states = attention_output_0 (无残差)
  - 第1层: hidden_states = attention_output_0 + attention_output_1
  - 第2层: hidden_states = (attention_output_0 + attention_output_1) + attention_output_2
  - 第N层: hidden_states = sum(attention_output_0 to attention_output_{N-1}) + attention_output_N

关键特点:
1. 第一层没有残差连接，完全依赖注意力输出
2. 后续层的残差来自之前所有层的注意力输出累积
3. 这种方式让注意力信息在层间以累积的方式传递
4. MLP部分仍保持原始的残差连接方式
5. 使得浅层的注意力特征能够直接影响深层的输出
    """)

if __name__ == "__main__":
    demonstrate_new_residual()
    explain_residual_difference()
    print(f"\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
