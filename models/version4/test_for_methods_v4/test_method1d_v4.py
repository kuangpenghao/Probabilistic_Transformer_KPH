#!/usr/bin/env python3
"""
测试Method1D的动态权重生成机制
"""

import torch
import sys
sys.path.append('.')

from models.version4.Method1D_v4 import Method1DLlamaForCausalLM_v4
from models.version4.configuration_llama_v4 import Method1DConfig_v4

def test_method1d_dynamic_weights():
    """测试Method1D的动态权重生成是否正常工作"""
    print("测试Method1D动态权重生成机制")
    print("=" * 80)
    
    # 创建配置
    config = Method1DConfig_v4()
    print(f"配置: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
    
    # 创建模型
    print("\n创建Method1D模型...")
    model = Method1DLlamaForCausalLM_v4(config)
    model.eval()
    
    # 创建测试输入
    batch_size = 2
    seq_len = 16  # 使用较小的序列长度进行测试
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"输入形状: {input_ids.shape}")
    
    # 测试前向传播
    print("\n测试前向传播...")
    try:
        with torch.no_grad():
            output = model(input_ids)
            
        loss = output.loss if hasattr(output, 'loss') and output.loss is not None else "N/A"
        logits_shape = output.logits.shape if hasattr(output, 'logits') else "N/A"
        
        print(f"✅ Method1D前向传播成功")
        print(f"   损失: {loss}")
        print(f"   输出logits形状: {logits_shape}")
        
    except Exception as e:
        print(f"❌ Method1D前向传播失败: {e}")
        return False
    
    # 详细分析MLP参数
    print("\n" + "=" * 80)
    print("MLP参数分析")
    print("=" * 80)
    
    def count_mlp_parameters(model):
        """统计MLP相关的参数"""
        total_params = 0
        mlp_details = []
        
        for name, param in model.named_parameters():
            if "modified_scaling" in name and ("layer_mlps" in name or "layer_biases" in name or "layer_norms" in name):
                param_count = param.numel()
                total_params += param_count
                mlp_details.append(f"  {name}: {list(param.shape)} -> {param_count} 参数")
        
        return total_params, mlp_details
    
    total_mlp_params, mlp_details = count_mlp_parameters(model)
    print(f"MLP相关参数总数: {total_mlp_params}")
    
    # 按层分组显示
    for i in range(config.num_hidden_layers):
        print(f"\n第{i}层MLP组件:")
        layer_details = [detail for detail in mlp_details if f"modified_scaling.layer_" in detail and f".{i}." in detail]
        for detail in layer_details[:3]:  # 显示前3个
            print(detail)
        if len(layer_details) > 3:
            print(f"    ... 还有 {len(layer_details) - 3} 个参数")
    
    # 计算理论参数量
    print(f"\n理论MLP参数计算:")
    total_theoretical = 0
    for layer_idx in range(config.num_hidden_layers):
        # RMSNorm: hidden_size
        norm_params = config.hidden_size
        # Linear1: hidden_size * 4*hidden_size  
        linear1_params = config.hidden_size * 4 * config.hidden_size
        # Linear2: 4*hidden_size * (layer_idx + 1)
        linear2_params = 4 * config.hidden_size * (layer_idx + 1)
        # Bias: (layer_idx + 1)
        bias_params = layer_idx + 1
        
        layer_total = norm_params + linear1_params + linear2_params + bias_params
        total_theoretical += layer_total
        
        print(f"  第{layer_idx}层: {layer_total} 参数 (输出维度: {layer_idx + 1})")
    
    print(f"理论总计: {total_theoretical} 参数")
    print(f"实际统计: {total_mlp_params} 参数")
    print(f"匹配度: {'✅' if total_theoretical == total_mlp_params else '❌'}")
    
    print("\n✅ Method1D测试完成！")
    return True

def test_dynamic_weight_matrix_generation():
    """测试动态权重矩阵生成的详细过程"""
    print("\n" + "=" * 80)
    print("测试动态权重矩阵生成")
    print("=" * 80)
    
    from models.version4.Method1D_v4 import v4m1D_ModifiedScailingComputation
    
    hidden_size = 64
    head_dim = 8
    num_layers = 2
    batch_size = 1
    seq_len = 8
    
    # 创建缩放计算模块
    scaling_module = v4m1D_ModifiedScailingComputation(hidden_size, head_dim, num_layers)
    
    # 模拟输入
    input_embedding = torch.randn(batch_size, seq_len, hidden_size)
    qk_matrices = [torch.randn(batch_size, 4, seq_len, seq_len) for _ in range(2)]  # 假设2个QK矩阵
    
    print(f"输入嵌入形状: {input_embedding.shape}")
    print(f"QK矩阵数量: {len(qk_matrices)}")
    print(f"每个QK矩阵形状: {qk_matrices[0].shape}")
    
    # 测试第1层（layer_idx=1，有2个QK矩阵）
    layer_idx = 1
    print(f"\n测试第{layer_idx}层（预期输出维度: {layer_idx + 1}）")
    
    try:
        result = scaling_module.compute_modified_scaling(qk_matrices, layer_idx, input_embedding)
        print(f"✅ 动态权重生成成功")
        print(f"   输出形状: {result.shape}")
        print(f"   预期形状: {qk_matrices[0].shape}")
        print(f"   形状匹配: {'✅' if result.shape == qk_matrices[0].shape else '❌'}")
        
        # 检查中间的权重矩阵A_i生成
        print(f"\n权重矩阵A_i生成过程:")
        normed_input = scaling_module.layer_norms[layer_idx](input_embedding)
        mlp_output = scaling_module.layer_mlps[layer_idx](normed_input)
        weight_matrix_A = mlp_output + scaling_module.layer_biases[layer_idx]
        
        print(f"   normed_input形状: {normed_input.shape}")
        print(f"   mlp_output形状: {mlp_output.shape}")
        print(f"   weight_matrix_A形状: {weight_matrix_A.shape}")
        print(f"   预期A_i形状: [{batch_size}, {seq_len}, {layer_idx + 1}]")
        
    except Exception as e:
        print(f"❌ 动态权重生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ 动态权重矩阵生成测试完成！")
    return True

if __name__ == "__main__":
    success1 = test_method1d_dynamic_weights()
    success2 = test_dynamic_weight_matrix_generation()
    
    if success1 and success2:
        print(f"\n🎉 所有测试通过！Method1D实现成功")
    else:
        print(f"\n❌ 部分测试失败，需要修复问题")
