#!/usr/bin/env python3
"""
比较Method1C和Method1D的参数数量差异
"""

import torch
from models.version4.configuration_llama_v4 import Method1CConfig_v4, Method1DConfig_v4
from models.version4.Method1C_v4 import Method1CLlamaForCausalLM_v4
from models.version4.Method1D_v4 import Method1DLlamaForCausalLM_v4

def compare_method1c_vs_method1d():
    print("="*70)
    print("Method1C vs Method1D 参数数量比较")
    print("="*70)
    
    # 使用相同的配置参数
    config_params = {
        'vocab_size': 32000,
        'hidden_size': 512,
        'intermediate_size': 1024,
        'num_hidden_layers': 8,
        'num_attention_heads': 8,
        'num_key_value_heads': 4,
        'max_position_embeddings': 1024,
        'rms_norm_eps': 1e-05,
        'use_cache': True,
    }
    
    # 创建Method1C模型
    print("创建Method1C模型...")
    config_1c = Method1CConfig_v4(**config_params)
    model_1c = Method1CLlamaForCausalLM_v4(config_1c)
    params_1c = sum({p.data_ptr(): p.numel() for p in model_1c.parameters()}.values())
    
    # 创建Method1D模型  
    print("创建Method1D模型...")
    config_1d = Method1DConfig_v4(**config_params)
    model_1d = Method1DLlamaForCausalLM_v4(config_1d)
    params_1d = sum({p.data_ptr(): p.numel() for p in model_1d.parameters()}.values())
    
    print(f"\n参数数量比较:")
    print(f"  Method1C: {params_1c:,} 参数")
    print(f"  Method1D: {params_1d:,} 参数")
    print(f"  差异: {params_1d - params_1c:,} 参数")
    print(f"  增长比例: {params_1d / params_1c:.2f}x")
    
    # 详细分析Method1D的MLP参数
    print(f"\nMethod1D MLP参数详情:")
    total_mlp_params = 0
    for layer_idx in range(config_1d.num_hidden_layers):
        layer = model_1d.model.layers[layer_idx].self_attn.modified_scaling
        
        # 计算该层的MLP参数
        layer_mlp_params = 0
        for mlp_idx in range(layer_idx + 1):  # 每层有 layer_idx+1 个MLP
            mlp = layer.layer_mlps[mlp_idx]
            mlp_params = sum(p.numel() for p in mlp.parameters())
            layer_mlp_params += mlp_params
        
        # RMSNorm参数
        norm_params = 0
        if hasattr(layer, 'layer_norms'):
            for norm_idx in range(layer_idx + 1):
                norm_params += layer.layer_norms[norm_idx].weight.numel()
        
        # Bias参数
        bias_params = sum(b.numel() for b in layer.layer_biases)
        
        layer_total = layer_mlp_params + norm_params + bias_params
        total_mlp_params += layer_total
        
        print(f"  层{layer_idx}: MLP={layer_mlp_params}, Norm={norm_params}, Bias={bias_params}, 总计={layer_total}")
    
    print(f"  MLP参数总计: {total_mlp_params:,}")
    
    # 执行forward测试
    print(f"\n执行forward测试...")
    batch_size, seq_len = 1, 16
    input_ids = torch.randint(0, config_params['vocab_size'], (batch_size, seq_len))
    
    # Method1C forward
    print("Method1C forward...")
    model_1c.eval()
    with torch.no_grad():
        outputs_1c = model_1c(input_ids)
    params_1c_after = sum({p.data_ptr(): p.numel() for p in model_1c.parameters()}.values())
    
    # Method1D forward  
    print("Method1D forward...")
    model_1d.eval()
    with torch.no_grad():
        outputs_1d = model_1d(input_ids)
    params_1d_after = sum({p.data_ptr(): p.numel() for p in model_1d.parameters()}.values())
    
    print(f"\nForward后参数数量:")
    print(f"  Method1C: {params_1c_after:,} (变化: {params_1c_after - params_1c:+,})")
    print(f"  Method1D: {params_1d_after:,} (变化: {params_1d_after - params_1d:+,})")
    
    if params_1c_after != params_1c:
        print("⚠️  Method1C参数数量发生变化!")
    if params_1d_after != params_1d:
        print("⚠️  Method1D参数数量发生变化!")
    
    print(f"\n结论:")
    print(f"  Method1D相比Method1C增加了 {params_1d - params_1c:,} 个参数")
    print(f"  这些参数来自动态MLP组件，用于生成权重矩阵A_i")
    print(f"  参数在模型初始化时就已创建，forward不会改变参数数量")

if __name__ == "__main__":
    compare_method1c_vs_method1d()
