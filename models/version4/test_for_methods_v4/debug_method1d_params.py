#!/usr/bin/env python3
"""
测试Method1D模型在forward前后的参数变化
"""

import torch
from models.version4.configuration_llama_v4 import Method1DConfig_v4
from models.version4.Method1D_v4 import Method1DLlamaForCausalLM_v4

def test_parameter_changes():
    print("="*70)
    print("Method1D参数变化测试")
    print("="*70)
    
    # 创建配置
    config = Method1DConfig_v4(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        use_cache=True,
    )
    
    print(f"配置: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")
    
    # 创建模型
    print("创建Method1D模型...")
    model = Method1DLlamaForCausalLM_v4(config)
    
    # 计算初始参数数量
    initial_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(f"初始参数数量: {initial_params:,}")
    
    # 详细列出各部分参数
    print("\n模型参数详情:")
    total_mlp_params = 0
    for name, param in model.named_parameters():
        if 'modified_scaling' in name:
            print(f"  {name}: {param.numel()} 参数")
            total_mlp_params += param.numel()
        elif len(name.split('.')) <= 3:  # 只显示顶层参数
            print(f"  {name}: {param.numel()} 参数")
    
    print(f"MLP相关参数总计: {total_mlp_params}")
    
    # 准备输入
    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"\n准备输入: batch_size={batch_size}, seq_len={seq_len}")
    
    # 第一次forward前再次检查参数
    pre_forward_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(f"Forward前参数数量: {pre_forward_params:,}")
    
    # 执行forward
    print("执行第一次forward...")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Forward后检查参数
    post_forward_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(f"Forward后参数数量: {post_forward_params:,}")
    
    # 检查是否有变化
    if post_forward_params != pre_forward_params:
        change = post_forward_params - pre_forward_params
        print(f"✅ 参数数量发生变化: +{change:,}")
    else:
        print("❌ 参数数量没有变化!")
        
    # 再次执行forward确认
    print("\n执行第二次forward...")
    with torch.no_grad():
        outputs2 = model(input_ids)
    
    second_forward_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(f"第二次Forward后参数数量: {second_forward_params:,}")
    
    if second_forward_params != post_forward_params:
        change2 = second_forward_params - post_forward_params
        print(f"第二次forward参数变化: +{change2:,}")
    else:
        print("第二次forward无参数变化（正常）")
    
    print(f"\n总结:")
    print(f"  初始: {initial_params:,}")
    print(f"  第一次forward后: {post_forward_params:,}")
    print(f"  第二次forward后: {second_forward_params:,}")
    
    # 检查模型是否包含expected的MLP参数
    print(f"\n检查MLP组件:")
    for layer_idx in range(config.num_hidden_layers):
        layer = model.model.layers[layer_idx].self_attn.modified_scaling
        mlp_params = sum(p.numel() for p in layer.layer_mlps[layer_idx].parameters())
        norm_params = sum(p.numel() for p in layer.layer_norms[layer_idx].parameters())
        bias_params = layer.layer_biases[layer_idx].numel()
        layer_total = mlp_params + norm_params + bias_params
        print(f"  层{layer_idx}: MLP={mlp_params}, Norm={norm_params}, Bias={bias_params}, 总计={layer_total}")
    
    return initial_params, post_forward_params

if __name__ == "__main__":
    test_parameter_changes()
