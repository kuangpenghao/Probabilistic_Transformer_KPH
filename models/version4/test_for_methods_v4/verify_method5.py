#!/usr/bin/env python3
"""
验证修改后的Method5可学习参数
"""

import torch
import sys
sys.path.append('/home/kuangph/hf-starter')

def verify_method5_parameters():
    """验证Method5的可学习参数"""
    print("=== Method5 参数验证 ===\n")
    
    from models.version4.Method5_v4 import Method5Config_v4, Method5LlamaForCausalLM_v4
    
    # 创建模型
    config = Method5Config_v4(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=512
    )
    
    model = Method5LlamaForCausalLM_v4(config)
    
    # 检查参数
    print("🔍 检查可学习参数:")
    scaling_params = []
    for name, param in model.named_parameters():
        if 'score_params' in name:
            scaling_params.append((name, param))
            print(f"  找到参数: {name}, shape: {param.shape}")
    
    print(f"\n📊 总共找到 {len(scaling_params)} 个score_params")
    
    # 测试梯度
    print("\n🔄 测试梯度计算:")
    model.train()
    
    # 前向传播
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    
    # 反向传播
    loss.backward()
    
    grad_count = 0
    for name, param in scaling_params:
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad_norm = {grad_norm:.6f}")
            grad_count += 1
        else:
            print(f"  {name}: ❌ 没有梯度")
    
    if grad_count == len(scaling_params):
        print("✅ 所有参数都有梯度")
    else:
        print("❌ 部分参数没有梯度")
    
    # 测试约束条件
    print("\n📐 验证约束条件:")
    for layer_idx in range(config.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        scaling_module = layer.self_attn.modified_scaling
        
        scores = scaling_module.score_params.data
        softmax_weights = torch.softmax(scores, dim=0)
        a_params = softmax_weights * len(scores)
        
        constraint_satisfied = abs(a_params.sum().item() - len(scores)) < 1e-6
        print(f"  Layer {layer_idx}: sum(a_params) = {a_params.sum().item():.6f}, 约束满足: {constraint_satisfied}")
    
    print("\n✅ Method5验证完成！")

if __name__ == "__main__":
    verify_method5_parameters()
