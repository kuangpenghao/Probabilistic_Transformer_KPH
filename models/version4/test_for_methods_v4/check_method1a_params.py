#!/usr/bin/env python3

def check_method1a_parameters(model):
    """检查Method1A模型的新增参数"""
    print("🔍 检查Method1A新增参数:")
    
    # 统计所有参数
    total_params = sum(p.numel() for p in model.parameters())
    
    # 统计新增的权重矩阵参数
    new_params = 0
    new_matrices = 0
    
    for name, param in model.named_parameters():
        if "layer_weight_matrices" in name:
            new_params += param.numel()
            new_matrices += 1
            print(f"  {name}: {param.shape} -> {param.numel():,} 参数")
    
    print(f"\n📊 参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  新增权重矩阵: {new_matrices}个")
    print(f"  新增参数: {new_params:,}")
    print(f"  新增参数占比: {(new_params/total_params)*100:.4f}%")
    
    return new_params > 0

# 在训练脚本中使用:
# has_new_params = check_method1a_parameters(model)
# print(f"Method1A新增参数检查: {'✓ 通过' if has_new_params else '✗ 失败'}")
