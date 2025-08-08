#!/usr/bin/env python3
"""
Method1A_v4模型测试脚本
测试模型是否能成功训练，并验证可学习权重矩阵参数是否正确更新
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
import sys

# 添加路径以导入模型
sys.path.append('/home/kuangph/hf-starter')

from models.version4.Method1A_v4 import Method1ALlamaForCausalLM_v4, Method1AConfig_v4
from transformers import AutoTokenizer


def create_test_config():
    """创建测试用的模型配置"""
    config = Method1AConfig_v4(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        model_type="method1a-v4",
        torch_dtype="float32"
    )
    return config


def create_dummy_dataset(vocab_size=1000, seq_len=64, num_samples=100):
    """创建虚拟数据集用于测试"""
    # 生成随机输入序列
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    # 标签是输入右移一位
    labels = torch.cat([input_ids[:, 1:], torch.zeros(num_samples, 1, dtype=torch.long)], dim=1)
    
    return TensorDataset(input_ids, labels)


def get_weight_matrices_state(model):
    """获取所有权重矩阵的当前状态"""
    weight_states = {}
    
    for layer_idx, layer in enumerate(model.model.layers):
        scaling_module = layer.self_attn.modified_scaling
        if scaling_module.layer_initialized[layer_idx]:
            weight_states[layer_idx] = []
            num_weights = len(scaling_module.layer_weight_matrices[layer_idx])
            for i in range(num_weights):
                weight_matrix = scaling_module.layer_weight_matrices[layer_idx][i]
                weight_states[layer_idx].append(weight_matrix.data.clone())
    
    return weight_states


def compare_weight_states(state1, state2, tolerance=1e-6):
    """比较两个权重状态，返回是否有变化"""
    changes = {}
    
    for layer_idx in state1:
        if layer_idx in state2:
            layer_changes = []
            for i, (w1, w2) in enumerate(zip(state1[layer_idx], state2[layer_idx])):
                diff = torch.abs(w1 - w2).max().item()
                layer_changes.append(diff)
            changes[layer_idx] = layer_changes
    
    return changes


def test_model_initialization():
    """测试模型初始化"""
    print("=" * 50)
    print("测试1: 模型初始化")
    print("=" * 50)
    
    try:
        config = create_test_config()
        model = Method1ALlamaForCausalLM_v4(config)
        print("✓ 模型初始化成功")
        
        # 检查模型结构
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ 总参数量: {total_params:,}")
        print(f"✓ 可训练参数量: {trainable_params:,}")
        
        return model, config
        
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        return None, None


def test_forward_pass(model, config):
    """测试前向传播"""
    print("\n" + "=" * 50)
    print("测试2: 前向传播")
    print("=" * 50)
    
    try:
        # 创建测试输入
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        print(f"✓ 输入形状: {input_ids.shape}")
        
        # 前向传播
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            
        print(f"✓ 输出logits形状: {outputs.logits.shape}")
        print(f"✓ 损失值: {outputs.loss.item():.4f}")
        
        # 检查权重矩阵是否被正确初始化
        initialized_layers = []
        for layer_idx, layer in enumerate(model.model.layers):
            scaling_module = layer.self_attn.modified_scaling
            if scaling_module.layer_initialized[layer_idx]:
                num_weights = len(scaling_module.layer_weight_matrices[layer_idx])
                initialized_layers.append((layer_idx, num_weights))
                print(f"✓ 第{layer_idx}层初始化了{num_weights}个权重矩阵")
        
        return True, initialized_layers
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def test_weight_learning(model, config):
    """测试权重矩阵是否能被学习"""
    print("\n" + "=" * 50)
    print("测试3: 权重矩阵学习能力")
    print("=" * 50)
    
    try:
        # 创建训练数据
        dataset = create_dummy_dataset(config.vocab_size, seq_len=32, num_samples=20)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # 设置优化器
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # 记录初始权重状态
        print("记录初始权重状态...")
        initial_weights = get_weight_matrices_state(model)
        
        # 训练几个步骤
        model.train()
        num_steps = 5
        losses = []
        
        print(f"开始训练 {num_steps} 步...")
        for step, (input_ids, labels) in enumerate(dataloader):
            if step >= num_steps:
                break
                
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            print(f"步骤 {step+1}/{num_steps}, 损失: {loss.item():.4f}")
        
        # 记录训练后权重状态
        print("记录训练后权重状态...")
        final_weights = get_weight_matrices_state(model)
        
        # 比较权重变化
        print("\n权重矩阵变化分析:")
        changes = compare_weight_states(initial_weights, final_weights)
        
        total_changed_weights = 0
        for layer_idx, layer_changes in changes.items():
            print(f"第{layer_idx}层:")
            for i, change in enumerate(layer_changes):
                status = "✓ 已更新" if change > 1e-6 else "✗ 未更新"
                print(f"  权重矩阵{i}: 最大变化 = {change:.2e} {status}")
                if change > 1e-6:
                    total_changed_weights += 1
        
        print(f"\n总结: {total_changed_weights} 个权重矩阵被成功更新")
        
        # 检查损失是否下降
        if len(losses) > 1:
            loss_improvement = losses[0] - losses[-1]
            print(f"损失改善: {loss_improvement:.4f} ({'✓ 下降' if loss_improvement > 0 else '✗ 未下降'})")
        
        return total_changed_weights > 0, changes
        
    except Exception as e:
        print(f"✗ 权重学习测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def test_weight_matrix_properties(model):
    """测试权重矩阵的属性"""
    print("\n" + "=" * 50)
    print("测试4: 权重矩阵属性验证")
    print("=" * 50)
    
    try:
        # 检查每一层的权重矩阵数量和形状
        for layer_idx, layer in enumerate(model.model.layers):
            scaling_module = layer.self_attn.modified_scaling
            if scaling_module.layer_initialized[layer_idx]:
                expected_num_weights = layer_idx + 1
                actual_num_weights = len(scaling_module.layer_weight_matrices[layer_idx])
                
                print(f"第{layer_idx}层:")
                print(f"  期望权重矩阵数量: {expected_num_weights}")
                print(f"  实际权重矩阵数量: {actual_num_weights}")
                
                if expected_num_weights == actual_num_weights:
                    print("  ✓ 权重矩阵数量正确")
                else:
                    print("  ✗ 权重矩阵数量错误")
                
                # 检查每个权重矩阵的形状和属性
                for i in range(actual_num_weights):
                    weight_matrix = scaling_module.layer_weight_matrices[layer_idx][i]
                    print(f"  权重矩阵{i}: 形状 {weight_matrix.shape}, requires_grad={weight_matrix.requires_grad}")
                    
                    # 检查是否为零初始化
                    is_zero_init = torch.allclose(weight_matrix, torch.zeros_like(weight_matrix))
                    print(f"    零初始化: {'✓' if is_zero_init else '✗'}")
        
        return True
        
    except Exception as e:
        print(f"✗ 权重矩阵属性测试失败: {e}")
        return False


def test_gradient_flow(model, config):
    """测试梯度流是否正常"""
    print("\n" + "=" * 50)
    print("测试5: 梯度流测试")
    print("=" * 50)
    
    try:
        # 创建测试输入 - 使用与之前相同的序列长度
        input_ids = torch.randint(0, config.vocab_size, (1, 32))
        labels = torch.randint(0, config.vocab_size, (1, 32))
        
        # 前向传播
        model.train()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # 反向传播
        loss.backward()
        
        # 检查权重矩阵的梯度
        gradient_stats = {}
        for layer_idx, layer in enumerate(model.model.layers):
            scaling_module = layer.self_attn.modified_scaling
            if scaling_module.layer_initialized[layer_idx]:
                layer_grads = []
                num_weights = len(scaling_module.layer_weight_matrices[layer_idx])
                for i in range(num_weights):
                    weight_matrix = scaling_module.layer_weight_matrices[layer_idx][i]
                    if weight_matrix.grad is not None:
                        grad_norm = weight_matrix.grad.norm().item()
                        layer_grads.append(grad_norm)
                        print(f"第{layer_idx}层权重矩阵{i}: 梯度范数 = {grad_norm:.2e}")
                    else:
                        layer_grads.append(0.0)
                        print(f"第{layer_idx}层权重矩阵{i}: 无梯度")
                
                gradient_stats[layer_idx] = layer_grads
        
        # 检查是否有梯度
        has_gradients = any(any(grad > 0 for grad in layer_grads) 
                           for layer_grads in gradient_stats.values())
        
        if has_gradients:
            print("✓ 权重矩阵梯度流正常")
        else:
            print("✗ 权重矩阵没有梯度")
        
        return has_gradients, gradient_stats
        
    except Exception as e:
        print(f"✗ 梯度流测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def main():
    """主测试函数"""
    print("Method1A_v4 模型测试开始")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_results = {}
    
    # 测试1: 模型初始化
    model, config = test_model_initialization()
    if model is None:
        print("模型初始化失败，终止测试")
        return
    test_results['initialization'] = True
    
    # 测试2: 前向传播
    forward_success, initialized_layers = test_forward_pass(model, config)
    test_results['forward_pass'] = forward_success
    
    if not forward_success:
        print("前向传播失败，终止测试")
        return
    
    # 测试3: 权重矩阵属性
    properties_success = test_weight_matrix_properties(model)
    test_results['weight_properties'] = properties_success

    # 测试4: 权重矩阵学习
    learning_success, weight_changes = test_weight_learning(model, config)
    test_results['weight_learning'] = learning_success    
    
    # 测试5: 梯度流
    gradient_success, gradient_stats = test_gradient_flow(model, config)
    test_results['gradient_flow'] = gradient_success
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    for test_name, success in test_results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(test_results.values())
    
    if all_passed:
        print("\n🎉 所有测试通过！Method1A_v4模型可以正常训练，权重矩阵参数能够被成功学习。")
    else:
        print("\n❌ 部分测试失败，请检查模型实现。")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
