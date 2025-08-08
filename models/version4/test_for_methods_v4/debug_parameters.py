#!/usr/bin/env python3

import torch
from models.version4.Method1A_v4 import Method1ALlamaForCausalLM_v4
from models.version4.configuration_llama_v4 import Method1AConfig_v4

def count_parameters_detailed(model):
    """详细统计模型参数"""
    total_params = 0
    trainable_params = 0
    
    print("=" * 80)
    print("详细参数统计:")
    print("=" * 80)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        
        # 检查是否是我们新增的权重矩阵参数
        if "layer_weight_matrices" in name:
            print(f"🔍 新增权重矩阵: {name}")
            print(f"   形状: {param.shape}")
            print(f"   参数数量: {param_count:,}")
            print(f"   requires_grad: {param.requires_grad}")
            print(f"   数据类型: {param.dtype}")
            print(f"   设备: {param.device}")
            print()
    
    print("=" * 80)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print("=" * 80)
    
    return total_params, trainable_params

def main():
    # 创建模型配置
    config = Method1AConfig_v4(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    
    print("创建Method1A_v4模型...")
    model = Method1ALlamaForCausalLM_v4(config)
    
    # 统计参数
    total, trainable = count_parameters_detailed(model)
    
    # 检查ModifiedScalingComputation的参数注册
    print("\n检查ModifiedScalingComputation模块:")
    print("=" * 50)
    
    for layer_idx, layer in enumerate(model.model.layers):
        scaling_module = layer.self_attn.modified_scaling
        print(f"\n第{layer_idx}层的ModifiedScalingComputation:")
        print(f"  layer_initialized: {scaling_module.layer_initialized}")
        print(f"  layer_weight_matrices类型: {type(scaling_module.layer_weight_matrices)}")
        print(f"  layer_weight_matrices长度: {len(scaling_module.layer_weight_matrices)}")
        
        # 检查每层的ParameterList
        for i, param_list in enumerate(scaling_module.layer_weight_matrices):
            print(f"    第{i}层ParameterList: 长度={len(param_list)}, 类型={type(param_list)}")
    
    # 强制触发一次前向传播，看看是否会初始化权重矩阵
    print("\n" + "=" * 80)
    print("执行前向传播以触发权重矩阵初始化...")
    print("=" * 80)
    
    input_ids = torch.randint(0, 1000, (2, 32))
    labels = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    print(f"前向传播完成，损失: {outputs.loss:.4f}")
    
    # 再次检查参数
    print("\n前向传播后的参数统计:")
    total_after, trainable_after = count_parameters_detailed(model)
    
    print(f"\n参数变化:")
    print(f"  前向传播前: {total:,} 总参数, {trainable:,} 可训练参数")
    print(f"  前向传播后: {total_after:,} 总参数, {trainable_after:,} 可训练参数")
    print(f"  新增参数: {total_after - total:,}")

if __name__ == "__main__":
    main()
