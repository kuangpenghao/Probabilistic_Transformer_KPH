#!/usr/bin/env python
# coding=utf-8
"""
深度调试：检查模块注册和参数层次结构
"""

import torch
from models.version3.configuration_llama_v3 import Method3_2Config_v3
from models.version3.Method3_2_v3 import Method3_2LlamaForCausalLM_v3


def check_module_registration(model, prefix=""):
    """递归检查模块注册"""
    print(f"\n=== 模块注册检查 {prefix} ===")
    
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print(f"模块: {full_name} -> {type(module).__name__}")
        
        # 检查该模块的参数
        for param_name, param in module.named_parameters(recurse=False):
            print(f"  参数: {full_name}.{param_name} -> {param.shape}")
        
        # 如果是我们关心的层，特别检查
        if 'layers' in name:
            print(f"  检查layers模块...")
            for i, layer in enumerate(module):
                print(f"    Layer {i}: {type(layer).__name__}")
                
                # 检查每层的直接参数
                for param_name, param in layer.named_parameters(recurse=False):
                    print(f"      直接参数: {param_name} -> {param.shape}")
                
                # 检查子模块
                for submodule_name, submodule in layer.named_children():
                    print(f"      子模块: {submodule_name} -> {type(submodule).__name__}")
                    
                    # 检查子模块的参数
                    for param_name, param in submodule.named_parameters(recurse=False):
                        print(f"        子模块参数: {submodule_name}.{param_name} -> {param.shape}")
                        if 'layer_weights' in param_name:
                            print(f"        ⭐ 找到layer_weights: {param}")


def test_parameter_updates():
    """测试参数更新"""
    print("\n=== 参数更新测试 ===")
    
    # 创建模型
    config = Method3_2Config_v3(
        vocab_size=50,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        torch_dtype="float32"
    )
    
    model = Method3_2LlamaForCausalLM_v3(config)
    
    # 检查参数注册
    check_module_registration(model)
    
    # 手动收集layer_weights参数
    layer_weights_params = []
    for name, param in model.named_parameters():
        if 'layer_weights' in name:
            layer_weights_params.append((name, param))
    
    print(f"\n通过named_parameters找到的layer_weights: {len(layer_weights_params)}")
    for name, param in layer_weights_params:
        print(f"  {name}: {param.shape}, {param}")
    
    # 直接从层中收集
    direct_layer_weights = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'modified_residual_mlp') and hasattr(layer.modified_residual_mlp, 'layer_weights'):
            param = layer.modified_residual_mlp.layer_weights
            direct_layer_weights.append((f"layer_{i}_weights", param))
    
    print(f"\n直接从层中找到的layer_weights: {len(direct_layer_weights)}")
    for name, param in direct_layer_weights:
        print(f"  {name}: {param.shape}, {param}")
    
    # 测试梯度计算
    print(f"\n=== 梯度测试 ===")
    
    # 创建输入
    input_ids = torch.randint(1, config.vocab_size-1, (1, 10))
    
    # 前向传播
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    
    print(f"损失: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print(f"\n梯度检查:")
    for name, param in direct_layer_weights:
        if param.grad is not None:
            print(f"  {name}: 梯度范数 = {param.grad.norm().item():.6f}")
            print(f"    梯度值: {param.grad}")
        else:
            print(f"  {name}: 没有梯度")
    
    # 优化器测试
    print(f"\n=== 优化器测试 ===")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 保存初始值
    initial_values = {}
    for name, param in direct_layer_weights:
        initial_values[name] = param.data.clone()
    
    # 执行优化步骤
    optimizer.step()
    
    # 检查更新
    print(f"优化后的权重变化:")
    for name, param in direct_layer_weights:
        initial = initial_values[name]
        current = param.data
        change = torch.abs(current - initial).sum().item()
        print(f"  {name}:")
        print(f"    初始: {initial}")
        print(f"    当前: {current}")
        print(f"    变化: {change:.6f}")


if __name__ == "__main__":
    test_parameter_updates()
