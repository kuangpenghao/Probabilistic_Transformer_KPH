#!/usr/bin/env python3
"""
测试Method6和Method7的第一层初始化
"""

import math
import torch
import sys
sys.path.append('/home/kuangph/hf-starter')

def test_method6_method7_first_layer():
    """测试Method6和Method7的第一层处理"""
    print("=== Method6和Method7第一层测试 ===\n")
    
    head_dim = 64
    first_layer_qk = [torch.randn(2, 8, 10, 10)]
    
    # Test Method6
    from models.version4.Method6_v4 import v4m6_ModifiedScailingComputation
    method6 = v4m6_ModifiedScailingComputation(head_dim, layer_idx=0)  # 第0层
    
    # 检查初始化参数
    print("Method6 第0层初始参数:")
    a_0 = torch.exp(method6.log_a_params[0]).item()
    b_0 = method6.b_params[0].item()
    print(f"  a_0 = {a_0:.3f}, b_0 = {b_0:.3f}")
    
    result6 = method6.compute_modified_scaling(first_layer_qk, 0)
    
    # Method6第一层: 1/(a_0*d_k^b_0)
    # 期望：a_0=1.0, b_0=0.5，得到 1/(1.0*d_k^0.5) = 1/sqrt(d_k)
    expected_scaling6 = 1.0 / (a_0 * (head_dim ** b_0))
    expected6 = first_layer_qk[0] * expected_scaling6
    
    print(f"  第一层缩放值: 1/({a_0:.3f}*{head_dim}^{b_0:.3f}) = {expected_scaling6:.6f}")
    print(f"  原始缩放值: 1/sqrt({head_dim}) = {1.0/math.sqrt(head_dim):.6f}")
    print(f"  一般化处理结果与期望一致: {torch.allclose(result6, expected6, atol=1e-6)}")
    
    # 检查是否需要调整初始化
    original_scaling = 1.0 / math.sqrt(head_dim)
    if abs(expected_scaling6 - original_scaling) > 1e-6:
        print("  ⚠️  需要调整Method6初始化: a_0=1.0, b_0=0.5")
    else:
        print("  ✅ Method6初始化正确")
    print()
    
    # Test Method7
    from models.version4.Method7_v4 import v4m7_ModifiedScailingComputation
    method7 = v4m7_ModifiedScailingComputation(head_dim, layer_idx=0)  # 第0层
    
    # 检查初始化参数
    print("Method7 第0层初始参数:")
    c_0 = torch.exp(method7.log_a_params[0]).item()
    print(f"  c_0 = {c_0:.3f}")
    
    result7 = method7.compute_modified_scaling(first_layer_qk, 0)
    
    # Method7第一层: c_0/sqrt(d_k)，但查看代码，应该是a_0，即向量的第一项
    expected_scaling7 = c_0
    expected7 = first_layer_qk[0] * expected_scaling7
    
    # Method7第一层: 1/a_0
    # 期望：a_0应该初始化为sqrt(d_k)，得到 1/sqrt(d_k)
    expected_scaling7 = 1.0 / c_0
    expected7 = first_layer_qk[0] * expected_scaling7
    
    print(f"  第一层缩放值: 1/{c_0:.3f} = {expected_scaling7:.6f}")
    print(f"  原始缩放值: 1/sqrt({head_dim}) = {1.0/math.sqrt(head_dim):.6f}")
    print(f"  一般化处理结果与期望一致: {torch.allclose(result7, expected7, atol=1e-6)}")
    
    # 检查是否需要调整初始化
    if abs(expected_scaling7 - original_scaling) > 1e-6:
        print(f"  ⚠️  需要调整Method7初始化: a_0=sqrt({head_dim})={math.sqrt(head_dim):.3f}")
    else:
        print("  ✅ Method7初始化正确")
    print()

if __name__ == "__main__":
    test_method6_method7_first_layer()
