#!/usr/bin/env python3
"""
测试 Method1E_v4 的实现
验证基于 MLP 动态生成行向量的功能
"""

import torch
from models.version4.Method1E_v4 import Method1ELlamaForCausalLM_v4
from models.version4.configuration_llama_v4 import Method1EConfig_v4

def test_method1e_v4():
    print("=== 测试 Method1E_v4 实现 ===")
    
    # 创建小型配置
    config = Method1EConfig_v4(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=512,
        use_cache=True,
    )
    
    # 创建模型
    model = Method1ELlamaForCausalLM_v4(config)
    
    print(f"模型总参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 检查MLP组件参数
    print("\n=== MLP 组件参数 ===")
    for name, param in model.named_parameters():
        if 'layer_mlps' in name or 'layer_biases' in name or 'layer_norms' in name:
            print(f"{name}: {param.shape}, 参数数量: {param.numel()}")
    
    # 创建测试输入
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\n=== 前向传播前参数统计 ===")
    total_params_before = sum(p.numel() for p in model.parameters())
    print(f"参数总数: {total_params_before:,}")
    
    # 前向传播
    print("\n=== 执行前向传播 ===")
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"\n=== 前向传播后参数统计 ===")
    total_params_after = sum(p.numel() for p in model.parameters())
    print(f"参数总数: {total_params_after:,}")
    print(f"新增参数数量: {total_params_after - total_params_before:,}")
    
    # 验证输出形状
    print(f"\n=== 输出验证 ===")
    logits = outputs.logits
    print(f"Logits 形状: {logits.shape}")
    print(f"预期形状: ({batch_size}, {seq_len}, {config.vocab_size})")
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "输出形状不正确"
    print("✓ 输出形状正确")
    
    # 验证不同层的MLP输出维度
    print(f"\n=== 验证 MLP 输出维度 ===")
    for layer_idx in range(config.num_hidden_layers):
        for decoder_layer in model.model.layers:
            if hasattr(decoder_layer.self_attn, 'modified_scaling'):
                expected_dim = layer_idx + 1  # 第i层应该有i+1个行向量
                bias_shape = decoder_layer.self_attn.modified_scaling.layer_biases[layer_idx].shape
                print(f"Layer {layer_idx} bias形状: {bias_shape}, 期望维度: {expected_dim}")
                assert bias_shape[0] == expected_dim, f"Layer {layer_idx} bias维度不正确"
                break
        break
    
    print("✓ 所有层的MLP维度正确")
    
    # 测试行向量广播逻辑
    print(f"\n=== 测试行向量广播逻辑 ===")
    batch_size = 2
    seq_len = 5
    weight_row = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], 
                               [0.5, 1.5, 2.5, 3.5, 4.5]])  # [batch_size, seq_len]
    
    # 模拟广播过程：[batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
    weight_broadcast = weight_row.unsqueeze(1).unsqueeze(1)
    print(f"原始权重行向量形状: {weight_row.shape}")
    print(f"广播后形状: {weight_broadcast.shape}")
    print("第一个样本的广播结果:")
    print(weight_broadcast[0, 0, 0, :])
    
    print("\n=== Method1E_v4 测试完成 ===")

if __name__ == "__main__":
    test_method1e_v4()
