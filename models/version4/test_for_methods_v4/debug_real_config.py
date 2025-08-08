#!/usr/bin/env python3

import torch
import json
from models.version4.Method1A_v4 import Method1ALlamaForCausalLM_v4
from models.version4.configuration_llama_v4 import Method1AConfig_v4

def load_config_from_json(config_path):
    """从JSON文件加载配置"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return Method1AConfig_v4(**config_dict)

def calculate_expected_new_parameters(config):
    """计算期望的新增参数数量"""
    num_layers = config.num_hidden_layers
    head_dim = config.hidden_size // config.num_attention_heads
    
    total_new_params = 0
    print("计算每层新增的权重矩阵参数:")
    
    for layer_idx in range(num_layers):
        # 第i层有i+1个权重矩阵
        num_matrices = layer_idx + 1
        # 每个权重矩阵的大小是 head_dim x head_dim
        params_per_matrix = head_dim * head_dim
        layer_new_params = num_matrices * params_per_matrix
        total_new_params += layer_new_params
        
        print(f"  第{layer_idx}层: {num_matrices}个矩阵 × {head_dim}×{head_dim} = {layer_new_params:,}参数")
    
    print(f"\n总新增参数: {total_new_params:,}")
    return total_new_params

def count_actual_parameters(model):
    """统计模型实际参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 统计新增的权重矩阵参数
    new_params = 0
    new_param_count = 0
    for name, param in model.named_parameters():
        if "layer_weight_matrices" in name:
            new_params += param.numel()
            new_param_count += 1
            
    return total_params, trainable_params, new_params, new_param_count

def main():
    # 加载实际训练配置
    config_path = "/home/kuangph/hf-starter/configs/Version4_Method1A.json"
    config = load_config_from_json(config_path)
    
    print("=" * 80)
    print("Method1A实际训练配置分析")
    print("=" * 80)
    print(f"vocab_size: {config.vocab_size:,}")
    print(f"hidden_size: {config.hidden_size}")
    print(f"num_hidden_layers: {config.num_hidden_layers}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    head_dim = config.hidden_size // config.num_attention_heads
    print(f"head_dim: {head_dim}")
    
    # 计算期望的新增参数
    print("\n" + "=" * 60)
    expected_new_params = calculate_expected_new_parameters(config)
    
    # 创建模型
    print("\n" + "=" * 60)
    print("创建Method1A模型...")
    model = Method1ALlamaForCausalLM_v4(config)
    
    # 统计初始参数
    total_before, trainable_before, new_before, new_count_before = count_actual_parameters(model)
    print(f"初始参数统计:")
    print(f"  总参数: {total_before:,}")
    print(f"  可训练参数: {trainable_before:,}")
    print(f"  新增权重矩阵参数: {new_before:,} (共{new_count_before}个矩阵)")
    
    # 执行前向传播触发权重初始化
    print("\n执行前向传播以触发权重矩阵初始化...")
    input_ids = torch.randint(0, config.vocab_size, (2, 64))
    labels = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    print(f"前向传播完成，损失: {outputs.loss:.4f}")
    
    # 统计前向传播后的参数
    total_after, trainable_after, new_after, new_count_after = count_actual_parameters(model)
    
    print("\n" + "=" * 60)
    print("参数统计结果:")
    print("=" * 60)
    print(f"前向传播前:")
    print(f"  总参数: {total_before:,}")
    print(f"  可训练参数: {trainable_before:,}")
    print(f"  新增权重矩阵参数: {new_before:,}")
    
    print(f"\n前向传播后:")
    print(f"  总参数: {total_after:,}")
    print(f"  可训练参数: {trainable_after:,}")
    print(f"  新增权重矩阵参数: {new_after:,} (共{new_count_after}个矩阵)")
    
    print(f"\n变化:")
    print(f"  新增总参数: {total_after - total_before:,}")
    print(f"  期望新增参数: {expected_new_params:,}")
    print(f"  实际新增参数: {new_after:,}")
    print(f"  匹配程度: {'✓ 完全匹配' if new_after == expected_new_params else '✗ 不匹配'}")
    
    # 计算新增参数占总参数的百分比
    if total_after > 0:
        new_param_percentage = (new_after / total_after) * 100
        print(f"  新增参数占总参数比例: {new_param_percentage:.4f}%")

if __name__ == "__main__":
    main()
