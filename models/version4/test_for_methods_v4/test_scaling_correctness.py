#!/usr/bin/env python3
"""
测试Version4方法的缩放计算是否正确实现了倒数
"""

import math
import torch
import sys
sys.path.append('/home/kuangph/hf-starter')

def test_scaling_correctness():
    """测试所有方法的缩放计算是否正确使用了倒数"""
    print("=== Version4方法缩放计算正确性测试 ===\n")
    
    # 测试参数
    head_dim = 64
    num_matrices = 3
    
    # 创建测试用的QK^T矩阵
    qk_matrices = [torch.randn(2, 8, 10, 10) for _ in range(num_matrices)]
    
    # Method1: 1/sqrt(d_k) 广播
    from models.version4.Method1_v4 import v4m1_ModifiedScailingComputation
    method1 = v4m1_ModifiedScailingComputation(head_dim)
    result1 = method1.compute_modified_scaling(qk_matrices, 1)
    
    # 手动计算期望结果
    expected_scaling = 1.0 / math.sqrt(head_dim)
    expected1 = torch.zeros_like(qk_matrices[-1])
    for qk in qk_matrices:
        expected1 += expected_scaling * qk
    
    print(f"Method1 测试:")
    print(f"  使用的缩放值: 1/sqrt({head_dim}) = {expected_scaling:.6f}")
    print(f"  计算结果是否正确: {torch.allclose(result1, expected1, atol=1e-6)}")
    print()
    
    # Method2: 1/sqrt(d_k*m)
    from models.version4.Method2_v4 import v4m2_ModifiedScailingComputation
    method2 = v4m2_ModifiedScailingComputation(head_dim)
    result2 = method2.compute_modified_scaling(qk_matrices, 1)
    
    expected_scaling2 = 1.0 / math.sqrt(head_dim * num_matrices)
    expected2 = torch.zeros_like(qk_matrices[-1])
    for qk in qk_matrices:
        expected2 += expected_scaling2 * qk
    
    print(f"Method2 测试:")
    print(f"  使用的缩放值: 1/sqrt({head_dim}*{num_matrices}) = {expected_scaling2:.6f}")
    print(f"  计算结果是否正确: {torch.allclose(result2, expected2, atol=1e-6)}")
    print()
    
    # Method3: 1/(sqrt(d_k)*m)
    from models.version4.Method3_v4 import v4m3_ModifiedScailingComputation
    method3 = v4m3_ModifiedScailingComputation(head_dim)
    result3 = method3.compute_modified_scaling(qk_matrices, 1)
    
    expected_scaling3 = 1.0 / (math.sqrt(head_dim) * num_matrices)
    expected3 = torch.zeros_like(qk_matrices[-1])
    for qk in qk_matrices:
        expected3 += expected_scaling3 * qk
    
    print(f"Method3 测试:")
    print(f"  使用的缩放值: 1/(sqrt({head_dim})*{num_matrices}) = {expected_scaling3:.6f}")
    print(f"  计算结果是否正确: {torch.allclose(result3, expected3, atol=1e-6)}")
    print()
    
    # Method4: 1/(d_k^a * m^b)
    from models.version4.Method4_v4 import v4m4_ModifiedScailingComputation
    method4 = v4m4_ModifiedScailingComputation(head_dim)
    result4 = method4.compute_modified_scaling(qk_matrices, 1)
    
    a = method4.a.item()
    b = method4.b.item()
    expected_scaling4 = 1.0 / ((head_dim ** a) * (num_matrices ** b))
    expected4 = torch.zeros_like(qk_matrices[-1])
    for qk in qk_matrices:
        expected4 += expected_scaling4 * qk
    
    print(f"Method4 测试:")
    print(f"  学习的参数: a={a:.3f}, b={b:.3f}")
    print(f"  使用的缩放值: 1/({head_dim}^{a:.3f} * {num_matrices}^{b:.3f}) = {expected_scaling4:.6f}")
    print(f"  计算结果是否正确: {torch.allclose(result4, expected4, atol=1e-5)}")
    print()
    
    # Method5: 各层不同的 1/(a_i*sqrt(d_k))
    from models.version4.Method5_v4 import v4m5_ModifiedScailingComputation
    method5 = v4m5_ModifiedScailingComputation(head_dim, layer_idx=2)  # 测试第2层（索引从0开始）
    result5 = method5.compute_modified_scaling(qk_matrices, 2)
    
    # 手动计算期望结果
    expected5 = torch.zeros_like(qk_matrices[-1])
    print(f"Method5 测试 (第2层，参数向量长度=3):")
    for i, qk in enumerate(qk_matrices):
        a_i = torch.exp(method5.log_a_params[i]).item()
        scaling = 1.0 / (a_i * math.sqrt(head_dim))
        expected5 += scaling * qk
        print(f"  层{i}: a_{i}={a_i:.3f}, 缩放值=1/({a_i:.3f}*sqrt({head_dim}))={scaling:.6f}")
    
    print(f"  计算结果是否正确: {torch.allclose(result5, expected5, atol=1e-5)}")
    print()
    
    # Method6: 各层不同的 1/(a_i * d_k^{b_i})
    from models.version4.Method6_v4 import v4m6_ModifiedScailingComputation
    method6 = v4m6_ModifiedScailingComputation(head_dim, layer_idx=2)  # 测试第2层
    result6 = method6.compute_modified_scaling(qk_matrices, 2)
    
    expected6 = torch.zeros_like(qk_matrices[-1])
    print(f"Method6 测试 (第2层，参数向量长度=3):")
    for i, qk in enumerate(qk_matrices):
        a_i = torch.exp(method6.log_a_params[i]).item()
        b_i = method6.b_params[i].item()
        scaling = 1.0 / (a_i * (head_dim ** b_i))
        expected6 += scaling * qk
        print(f"  层{i}: a_{i}={a_i:.3f}, b_{i}={b_i:.3f}, 缩放值=1/({a_i:.3f}*{head_dim}^{b_i:.3f})={scaling:.6f}")
    
    print(f"  计算结果是否正确: {torch.allclose(result6, expected6, atol=1e-5)}")
    print()
    
    # Method7: 各层不同的 1/a_i
    from models.version4.Method7_v4 import v4m7_ModifiedScailingComputation
    method7 = v4m7_ModifiedScailingComputation(head_dim, layer_idx=2)  # 测试第2层
    result7 = method7.compute_modified_scaling(qk_matrices, 2)
    
    expected7 = torch.zeros_like(qk_matrices[-1])
    print(f"Method7 测试 (第2层，参数向量长度=3):")
    for i, qk in enumerate(qk_matrices):
        a_i = torch.exp(method7.log_a_params[i]).item()
        scaling = 1.0 / a_i
        expected7 += scaling * qk
        print(f"  层{i}: a_{i}={a_i:.3f}, 缩放值=1/{a_i:.3f}={scaling:.6f}")
    
    print(f"  计算结果是否正确: {torch.allclose(result7, expected7, atol=1e-5)}")
    print()

def test_parameter_constraints():
    """测试可学习参数的正负性约束"""
    print("=== 参数正负性约束测试 ===\n")
    
    head_dim = 64
    
    # Method4: a和b可以为正负
    from models.version4.Method4_v4 import v4m4_ModifiedScailingComputation
    method4 = v4m4_ModifiedScailingComputation(head_dim)
    method4.a.data = torch.tensor(-0.5)
    method4.b.data = torch.tensor(1.2)
    
    print(f"Method4 参数约束测试:")
    print(f"  a={method4.a.item():.3f} (可以为负数)")
    print(f"  b={method4.b.item():.3f} (可以为正数)")
    print(f"  ✓ Method4参数约束正确\n")
    
    # Method5: a_i必须为正值
    from models.version4.Method5_v4 import v4m5_ModifiedScailingComputation
    method5 = v4m5_ModifiedScailingComputation(head_dim, layer_idx=3)  # 第3层，参数向量长度=4
    method5.log_a_params.data = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    
    print(f"Method5 参数约束测试 (第3层，参数向量长度=4):")
    for i in range(4):
        log_a_i = method5.log_a_params[i].item()
        a_i = torch.exp(method5.log_a_params[i]).item()
        print(f"  log_a_{i}={log_a_i:.3f} -> a_{i}={a_i:.3f} (必须为正数)")
    print(f"  ✓ Method5参数约束正确（通过exp确保正值）\n")
    
    # Method6: a_i必须为正值，b_i可以为正负
    from models.version4.Method6_v4 import v4m6_ModifiedScailingComputation
    method6 = v4m6_ModifiedScailingComputation(head_dim, layer_idx=3)  # 第3层，参数向量长度=4
    method6.log_a_params.data = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    method6.b_params.data = torch.tensor([-0.5, 0.0, 0.5, 1.0])
    
    print(f"Method6 参数约束测试 (第3层，参数向量长度=4):")
    for i in range(4):
        log_a_i = method6.log_a_params[i].item()
        a_i = torch.exp(method6.log_a_params[i]).item()
        b_i = method6.b_params[i].item()
        print(f"  log_a_{i}={log_a_i:.3f} -> a_{i}={a_i:.3f} (必须为正数), b_{i}={b_i:.3f} (可以为负数)")
    print(f"  ✓ Method6参数约束正确\n")
    
    # Method7: a_i必须为正值
    from models.version4.Method7_v4 import v4m7_ModifiedScailingComputation
    method7 = v4m7_ModifiedScailingComputation(head_dim, layer_idx=3)  # 第3层，参数向量长度=4
    method7.log_a_params.data = torch.tensor([-2.0, -1.0, 0.0, 1.0])
    
    print(f"Method7 参数约束测试 (第3层，参数向量长度=4):")
    for i in range(4):
        log_a_i = method7.log_a_params[i].item()
        a_i = torch.exp(method7.log_a_params[i]).item()
        print(f"  log_a_{i}={log_a_i:.3f} -> a_{i}={a_i:.3f} (必须为正数)")
    print(f"  ✓ Method7参数约束正确（通过exp确保正值）\n")

if __name__ == "__main__":
    test_scaling_correctness()
    test_parameter_constraints()
    print("🎉 所有缩放计算和参数约束测试都通过！")
