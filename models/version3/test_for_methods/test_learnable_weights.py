#!/usr/bin/env python
# coding=utf-8
"""
测试脚本：验证Method3_2和Method4_2中的可学习权重参数是否被正确学习
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from models.version3.configuration_llama_v3 import Method3_2Config_v3, Method4_2Config_v3
from models.version3.Method3_2_v3 import Method3_2LlamaForCausalLM_v3
from models.version3.Method4_2_v3 import Method4_2LlamaForCausalLM_v3


def create_test_model(model_class, config_class, model_name="Method3_2"):
    """创建测试模型"""
    print(f"\n=== 创建{model_name}测试模型 ===")
    
    # 创建小型配置以便快速测试
    config = config_class(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,  # 只用4层便于观察
        num_attention_heads=4,
        max_position_embeddings=512,
        torch_dtype="float32"
    )
    
    model = model_class(config)
    print(f"模型创建成功，总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config


def print_initial_weights(model, model_name):
    """打印模型初始权重"""
    print(f"\n=== {model_name} 初始权重分布 ===")
    
    if hasattr(model, 'get_all_layer_weights'):
        layer_weights = model.get_all_layer_weights()
        for layer_idx, weights in enumerate(layer_weights):
            if len(weights) > 0:
                weights_np = weights.detach().cpu().numpy()
                print(f"Layer {layer_idx}: {weights_np} (sum={weights_np.sum():.6f})")
    else:
        print(f"模型 {model_name} 没有 get_all_layer_weights 方法!")


def print_learnable_parameters(model, model_name):
    """打印所有可学习参数"""
    print(f"\n=== {model_name} 可学习权重参数详情 ===")
    
    learnable_weight_count = 0
    for name, param in model.named_parameters():
        if 'layer_weights' in name:
            learnable_weight_count += 1
            print(f"{name}: {param.data} (requires_grad={param.requires_grad})")
    
    print(f"总计可学习权重参数: {learnable_weight_count}")
    return learnable_weight_count


def create_simple_training_data(vocab_size, config, num_samples=100):
    """创建简单的训练数据"""
    print(f"\n=== 创建训练数据 ===")
    
    # 创建简单的随机token序列
    sequence_length = min(64, config.max_position_embeddings//8)  # 使用较短序列
    
    input_ids = []
    for i in range(num_samples):
        # 创建随机token序列，避免padding token (0)
        sequence = torch.randint(1, vocab_size-1, (sequence_length,))
        input_ids.append(sequence)
    
    # 堆叠成batch
    input_ids = torch.stack(input_ids)
    
    print(f"创建了 {num_samples} 个训练样本，序列长度: {sequence_length}")
    return {"input_ids": input_ids}


def simple_training_step(model, data, optimizer, num_steps=50):
    """执行简单的训练步骤"""
    print(f"\n=== 开始训练 ({num_steps} 步) ===")
    
    model.train()
    initial_loss = None
    final_loss = None
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(input_ids=data['input_ids'], labels=data['input_ids'])
        loss = outputs.loss
        
        if step == 0:
            initial_loss = loss.item()
        if step == num_steps - 1:
            final_loss = loss.item()
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
    
    print(f"训练完成! 初始损失: {initial_loss:.4f}, 最终损失: {final_loss:.4f}")
    return initial_loss, final_loss


def test_gradient_flow(model, data, model_name):
    """测试梯度流是否正常"""
    print(f"\n=== 测试 {model_name} 梯度流 ===")
    
    model.train()
    model.zero_grad()
    
    # 前向传播
    outputs = model(input_ids=data['input_ids'], labels=data['input_ids'])
    loss = outputs.loss
    
    print(f"前向传播损失: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    # 检查可学习权重的梯度
    gradient_found = False
    for name, param in model.named_parameters():
        if 'layer_weights' in name:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"{name} 梯度范数: {grad_norm:.6f}")
                if grad_norm > 1e-8:
                    gradient_found = True
            else:
                print(f"{name} 没有梯度!")
    
    if gradient_found:
        print("✅ 发现可学习权重的梯度，梯度流正常")
    else:
        print("❌ 没有发现可学习权重的梯度，可能存在问题!")
    
    return gradient_found


def compare_weights_before_after(model, model_name, initial_weights, final_weights):
    """比较训练前后的权重变化"""
    print(f"\n=== {model_name} 权重变化分析 ===")
    
    changed_layers = 0
    total_change = 0.0
    
    for layer_idx in range(len(initial_weights)):
        if len(initial_weights[layer_idx]) > 0 and len(final_weights[layer_idx]) > 0:
            init_weights = initial_weights[layer_idx]
            final_weights_layer = final_weights[layer_idx]
            
            # 计算权重变化
            weight_change = torch.abs(final_weights_layer - init_weights).sum().item()
            total_change += weight_change
            
            print(f"Layer {layer_idx}:")
            print(f"  初始: {init_weights.numpy()}")
            print(f"  最终: {final_weights_layer.numpy()}")
            print(f"  变化: {weight_change:.6f}")
            
            if weight_change > 1e-6:
                changed_layers += 1
    
    print(f"\n总结:")
    print(f"  变化的层数: {changed_layers}/{len(initial_weights)}")
    print(f"  总权重变化: {total_change:.6f}")
    
    if total_change > 1e-4:
        print("✅ 权重发生了显著变化，学习正常")
        return True
    else:
        print("❌ 权重几乎没有变化，可能存在学习问题!")
        return False


def save_test_results(results, output_dir="test_results"):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果到JSON文件
    results_file = os.path.join(output_dir, "learnable_weights_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n测试结果已保存到: {results_file}")


def main():
    print("开始可学习权重参数测试...")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_results = {}
    
    # 测试Method3_2
    print("\n" + "="*60)
    print("测试 Method3_2 (MLP可学习权重)")
    print("="*60)
    
    model3_2, config3_2 = create_test_model(Method3_2LlamaForCausalLM_v3, Method3_2Config_v3, "Method3_2")
    
    # 检查可学习参数
    learnable_count_3_2 = print_learnable_parameters(model3_2, "Method3_2")
    
    # 打印初始权重
    print_initial_weights(model3_2, "Method3_2")
    initial_weights_3_2 = model3_2.get_all_layer_weights()
    
    # 创建训练数据
    train_data = create_simple_training_data(config3_2.vocab_size, config3_2)
    
    # 测试梯度流
    gradient_ok_3_2 = test_gradient_flow(model3_2, train_data, "Method3_2")
    
    # 训练
    optimizer3_2 = optim.Adam(model3_2.parameters(), lr=1e-3)
    initial_loss_3_2, final_loss_3_2 = simple_training_step(model3_2, train_data, optimizer3_2)
    
    # 获取训练后权重
    final_weights_3_2 = model3_2.get_all_layer_weights()
    print_initial_weights(model3_2, "Method3_2 (训练后)")
    
    # 比较权重变化
    weights_changed_3_2 = compare_weights_before_after(model3_2, "Method3_2", initial_weights_3_2, final_weights_3_2)
    
    test_results["Method3_2"] = {
        "learnable_parameters_count": learnable_count_3_2,
        "gradient_flow_ok": gradient_ok_3_2,
        "initial_loss": initial_loss_3_2,
        "final_loss": final_loss_3_2,
        "loss_decreased": final_loss_3_2 < initial_loss_3_2,
        "weights_changed": weights_changed_3_2,
        "initial_weights": [w.tolist() if len(w) > 0 else [] for w in initial_weights_3_2],
        "final_weights": [w.tolist() if len(w) > 0 else [] for w in final_weights_3_2]
    }
    
    # 测试Method4_2
    print("\n" + "="*60)
    print("测试 Method4_2 (Attention可学习权重)")
    print("="*60)
    
    model4_2, config4_2 = create_test_model(Method4_2LlamaForCausalLM_v3, Method4_2Config_v3, "Method4_2")
    
    # 检查可学习参数
    learnable_count_4_2 = print_learnable_parameters(model4_2, "Method4_2")
    
    # 打印初始权重
    print_initial_weights(model4_2, "Method4_2")
    initial_weights_4_2 = model4_2.get_all_layer_weights()
    
    # 创建训练数据 (使用相同的数据)
    
    # 测试梯度流
    gradient_ok_4_2 = test_gradient_flow(model4_2, train_data, "Method4_2")
    
    # 训练
    optimizer4_2 = optim.Adam(model4_2.parameters(), lr=1e-3)
    initial_loss_4_2, final_loss_4_2 = simple_training_step(model4_2, train_data, optimizer4_2)
    
    # 获取训练后权重
    final_weights_4_2 = model4_2.get_all_layer_weights()
    print_initial_weights(model4_2, "Method4_2 (训练后)")
    
    # 比较权重变化
    weights_changed_4_2 = compare_weights_before_after(model4_2, "Method4_2", initial_weights_4_2, final_weights_4_2)
    
    test_results["Method4_2"] = {
        "learnable_parameters_count": learnable_count_4_2,
        "gradient_flow_ok": gradient_ok_4_2,
        "initial_loss": initial_loss_4_2,
        "final_loss": final_loss_4_2,
        "loss_decreased": final_loss_4_2 < initial_loss_4_2,
        "weights_changed": weights_changed_4_2,
        "initial_weights": [w.tolist() if len(w) > 0 else [] for w in initial_weights_4_2],
        "final_weights": [w.tolist() if len(w) > 0 else [] for w in final_weights_4_2]
    }
    
    # 保存测试结果
    save_test_results(test_results)
    
    # 打印最终总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for model_name, results in test_results.items():
        print(f"\n{model_name}:")
        print(f"  ✅ 可学习参数数量: {results['learnable_parameters_count']}")
        print(f"  {'✅' if results['gradient_flow_ok'] else '❌'} 梯度流: {'正常' if results['gradient_flow_ok'] else '异常'}")
        print(f"  {'✅' if results['loss_decreased'] else '❌'} 损失下降: {results['initial_loss']:.4f} → {results['final_loss']:.4f}")
        print(f"  {'✅' if results['weights_changed'] else '❌'} 权重变化: {'显著' if results['weights_changed'] else '微小'}")
    
    # 检查整体结果
    all_tests_passed = all(
        results['gradient_flow_ok'] and results['loss_decreased'] and results['weights_changed']
        for results in test_results.values()
    )
    
    if all_tests_passed:
        print("\n🎉 所有测试通过! 可学习权重工作正常")
    else:
        print("\n⚠️  部分测试失败，需要检查实现")
    
    return test_results


if __name__ == "__main__":
    results = main()
