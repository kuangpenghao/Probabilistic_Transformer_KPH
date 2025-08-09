#!/usr/bin/env python3
"""
测试MethodCbase和MethodDbase模型
"""

import torch
from models.version4.configuration_llama_v4 import MethodCbaseConfig_v4, MethodDbaseConfig_v4
from models.version4.MethodCbase_v4 import MethodCbaseLlamaForCausalLM_v4
from models.version4.MethodDbase_v4 import MethodDbaseLlamaForCausalLM_v4

def test_methodcbase():
    print("="*70)
    print("MethodCbase测试 - 基于原始LlamaModel + 可学习权重列向量")
    print("="*70)
    
    # 创建配置
    config = MethodCbaseConfig_v4(
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
    print("创建MethodCbase模型...")
    model = MethodCbaseLlamaForCausalLM_v4(config)
    
    # 计算参数数量
    total_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(f"模型总参数数量: {total_params:,}")
    
    # 准备输入
    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"输入形状: {input_ids.shape}")
    
    # Forward前参数数量
    pre_forward_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(f"Forward前参数数量: {pre_forward_params:,}")
    
    # 执行forward
    print("测试前向传播...")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"✅ MethodCbase前向传播成功")
    print(f"   输出logits形状: {outputs.logits.shape}")
    
    # Forward后参数数量
    post_forward_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(f"Forward后参数数量: {post_forward_params:,}")
    
    if post_forward_params != pre_forward_params:
        change = post_forward_params - pre_forward_params
        print(f"✅ 动态参数创建: +{change:,} 参数")
    else:
        print("⚠️  无参数变化（所有参数在初始化时创建）")
    
    # 检查权重向量
    print(f"\n检查权重向量:")
    for layer_idx in range(config.num_hidden_layers):
        scaling_module = model.model.layers[layer_idx].self_attn.modified_scaling
        if scaling_module.layer_initialized[layer_idx]:
            weight_vector = scaling_module.layer_weight_vectors[layer_idx][0]
            print(f"  层{layer_idx}: 权重向量形状={weight_vector.shape}, 已初始化")
        else:
            print(f"  层{layer_idx}: 未初始化")


def test_methoddbase():
    print("\n" + "="*70)
    print("MethodDbase测试 - 基于原始LlamaModel + MLP动态生成权重列向量")
    print("="*70)
    
    # 创建配置
    config = MethodDbaseConfig_v4(
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
    print("创建MethodDbase模型...")
    model = MethodDbaseLlamaForCausalLM_v4(config)
    
    # 计算参数数量
    total_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(f"模型总参数数量: {total_params:,}")
    
    # 详细分析MLP参数
    print(f"\nMLP参数详情:")
    total_mlp_params = 0
    for layer_idx in range(config.num_hidden_layers):
        scaling_module = model.model.layers[layer_idx].self_attn.modified_scaling
        
        # MLP参数：hidden_size -> 1 -> 1
        mlp = scaling_module.layer_mlps[layer_idx]
        mlp_params = sum(p.numel() for p in mlp.parameters())
        
        # RMSNorm参数
        norm_params = scaling_module.layer_norms[layer_idx].weight.numel()
        
        # Bias参数
        bias_params = scaling_module.layer_biases[layer_idx].numel()
        
        layer_total = mlp_params + norm_params + bias_params
        total_mlp_params += layer_total
        
        print(f"  层{layer_idx}: MLP={mlp_params}, Norm={norm_params}, Bias={bias_params}, 总计={layer_total}")
    
    print(f"  MLP参数总计: {total_mlp_params:,}")
    
    # 准备输入
    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"\n输入形状: {input_ids.shape}")
    
    # Forward前参数数量
    pre_forward_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(f"Forward前参数数量: {pre_forward_params:,}")
    
    # 执行forward
    print("测试前向传播...")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"✅ MethodDbase前向传播成功")
    print(f"   输出logits形状: {outputs.logits.shape}")
    
    # Forward后参数数量
    post_forward_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(f"Forward后参数数量: {post_forward_params:,}")
    
    if post_forward_params != pre_forward_params:
        change = post_forward_params - pre_forward_params
        print(f"✅ 动态参数创建: +{change:,} 参数")
    else:
        print("✅ 无参数变化（MLP参数在初始化时创建）")


def compare_models():
    print("\n" + "="*70)
    print("模型对比总结")
    print("="*70)
    
    config = MethodCbaseConfig_v4(
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
    
    # 创建两个模型
    model_cbase = MethodCbaseLlamaForCausalLM_v4(config)
    model_dbase = MethodDbaseLlamaForCausalLM_v4(config)
    
    params_cbase_init = sum({p.data_ptr(): p.numel() for p in model_cbase.parameters()}.values())
    params_dbase_init = sum({p.data_ptr(): p.numel() for p in model_dbase.parameters()}.values())
    
    print(f"MethodCbase初始参数: {params_cbase_init:,}")
    print(f"MethodDbase初始参数: {params_dbase_init:,}")
    print(f"参数差异: {params_dbase_init - params_cbase_init:+,}")
    
    # 执行forward测试参数变化
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    
    with torch.no_grad():
        model_cbase.eval()
        model_dbase.eval()
        
        _ = model_cbase(input_ids)
        _ = model_dbase(input_ids)
    
    params_cbase_after = sum({p.data_ptr(): p.numel() for p in model_cbase.parameters()}.values())
    params_dbase_after = sum({p.data_ptr(): p.numel() for p in model_dbase.parameters()}.values())
    
    print(f"\nForward后:")
    print(f"MethodCbase: {params_cbase_after:,} (变化: {params_cbase_after - params_cbase_init:+,})")
    print(f"MethodDbase: {params_dbase_after:,} (变化: {params_dbase_after - params_dbase_init:+,})")
    
    print(f"\n核心差异:")
    print(f"- MethodCbase: 权重列向量在第一次forward时动态创建")
    print(f"- MethodDbase: MLP参数在模型初始化时创建，权重列向量动态计算")
    print(f"- 两者都基于原始LlamaModel，只对当前层QK^T应用权重")


if __name__ == "__main__":
    test_methodcbase()
    test_methoddbase()
    compare_models()
    print("\n🎉 所有测试完成！MethodCbase和MethodDbase模型实现成功")
