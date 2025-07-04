#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试residual_sum的shape和除法操作的影响
"""

import torch
import sys
import os

def debug_residual_shapes():
    """调试残差计算中的shape变化"""
    print("=== 调试residual_sum的shape ===")
    
    # 模拟真实的attention输出维度
    batch_size = 2
    seq_len = 10
    hidden_size = 512
    
    print(f"假设参数: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
    
    # 创建模拟的注意力输出（这些是previous_attn_outputs列表中的元素）
    attn_output_1 = torch.randn(batch_size, seq_len, hidden_size)
    attn_output_2 = torch.randn(batch_size, seq_len, hidden_size)
    attn_output_3 = torch.randn(batch_size, seq_len, hidden_size)
    
    print(f"\n单个注意力输出的shape: {attn_output_1.shape}")
    
    # 模拟previous_attn_outputs列表
    previous_attn_outputs = [attn_output_1, attn_output_2, attn_output_3]
    
    print(f"previous_attn_outputs列表长度: {len(previous_attn_outputs)}")
    
    # 计算residual_sum (这是sum()操作)
    residual_sum = sum(previous_attn_outputs)
    print(f"residual_sum的shape: {residual_sum.shape}")
    print(f"residual_sum的类型: {type(residual_sum)}")
    
    # 计算residual_avg (这是除法操作)
    residual_avg = residual_sum / len(previous_attn_outputs)
    print(f"residual_avg的shape: {residual_avg.shape}")
    print(f"residual_avg的类型: {type(residual_avg)}")
    
    # 验证shape是否保持一致
    print(f"\nshape是否保持一致:")
    print(f"原始attention输出: {attn_output_1.shape}")
    print(f"residual_sum: {residual_sum.shape}")
    print(f"residual_avg: {residual_avg.shape}")
    
    shapes_consistent = (attn_output_1.shape == residual_sum.shape == residual_avg.shape)
    print(f"所有shape是否一致: {shapes_consistent}")
    
    # 测试除法操作的维度
    print(f"\n=== 测试除法操作 ===")
    len_value = len(previous_attn_outputs)
    print(f"len(previous_attn_outputs) = {len_value} (标量)")
    print(f"len_value的类型: {type(len_value)}")
    
    # 验证广播机制
    print(f"\nPyTorch广播机制测试:")
    test_tensor = torch.randn(batch_size, seq_len, hidden_size)
    test_result = test_tensor / len_value
    print(f"test_tensor.shape: {test_tensor.shape}")
    print(f"test_result.shape: {test_result.shape}")
    print(f"除法操作是否改变shape: {test_tensor.shape != test_result.shape}")
    
    # 验证数值计算正确性
    print(f"\n=== 验证数值计算 ===")
    # 使用简单的值进行验证
    simple_tensor1 = torch.ones(2, 3, 4) * 3.0  # 全部元素为3
    simple_tensor2 = torch.ones(2, 3, 4) * 6.0  # 全部元素为6
    simple_tensor3 = torch.ones(2, 3, 4) * 9.0  # 全部元素为9
    
    simple_list = [simple_tensor1, simple_tensor2, simple_tensor3]
    simple_sum = sum(simple_list)  # 每个元素应该是3+6+9=18
    simple_avg = simple_sum / len(simple_list)  # 每个元素应该是18/3=6
    
    print(f"简单测试 - 期望平均值: 6.0")
    print(f"实际计算的平均值: {simple_avg[0, 0, 0].item()}")
    print(f"计算正确性: {abs(simple_avg[0, 0, 0].item() - 6.0) < 1e-6}")

def test_with_current_attn_output():
    """测试与当前注意力输出相加的情况"""
    print(f"\n=== 测试与当前注意力输出相加 ===")
    
    batch_size, seq_len, hidden_size = 2, 5, 8
    
    # 模拟之前层的注意力输出
    prev_outputs = [
        torch.ones(batch_size, seq_len, hidden_size) * 1.0,
        torch.ones(batch_size, seq_len, hidden_size) * 2.0,
        torch.ones(batch_size, seq_len, hidden_size) * 3.0,
    ]
    
    # 模拟当前层的注意力输出
    current_attn_output = torch.ones(batch_size, seq_len, hidden_size) * 4.0
    
    # Method2的计算方式
    residual_sum = sum(prev_outputs)  # 1+2+3=6
    residual_avg = residual_sum / len(prev_outputs)  # 6/3=2
    result = residual_avg + current_attn_output  # 2+4=6
    
    print(f"previous_attn_outputs数值: [1.0, 2.0, 3.0]")
    print(f"current_attn_output数值: 4.0")
    print(f"residual_sum数值: {residual_sum[0, 0, 0].item()}")
    print(f"residual_avg数值: {residual_avg[0, 0, 0].item()}")
    print(f"最终结果数值: {result[0, 0, 0].item()}")
    print(f"期望最终结果: 6.0")
    print(f"计算正确: {abs(result[0, 0, 0].item() - 6.0) < 1e-6}")
    
    # 检查所有维度的shape
    print(f"\nshape检查:")
    print(f"prev_outputs[0].shape: {prev_outputs[0].shape}")
    print(f"residual_sum.shape: {residual_sum.shape}")
    print(f"residual_avg.shape: {residual_avg.shape}")
    print(f"current_attn_output.shape: {current_attn_output.shape}")
    print(f"result.shape: {result.shape}")
    
    all_shapes_same = all(t.shape == result.shape for t in [prev_outputs[0], residual_sum, residual_avg, current_attn_output])
    print(f"所有tensor的shape都相同: {all_shapes_same}")

if __name__ == "__main__":
    debug_residual_shapes()
    test_with_current_attn_output()
    print(f"\n{'='*50}")
    print("✅ 结论:")
    print("1. residual_sum的shape = (batch_size, seq_len, hidden_size)")
    print("2. 除法操作不会改变tensor的shape")
    print("3. len(previous_attn_outputs)是标量，通过广播机制进行元素级除法")
    print("4. 所有计算都是元素级的，不会影响其他维度")
    print("5. 残差平均化计算在数学上是正确的")
    print("="*50)
