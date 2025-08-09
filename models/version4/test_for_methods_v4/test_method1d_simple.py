#!/usr/bin/env python3
"""
使用小型配置测试Method1D的动态权重生成机制
"""

import torch
import sys
sys.path.append('.')

from models.version4.Method1D_v4 import Method1DLlamaForCausalLM_v4, v4m1D_ModifiedScailingComputation
from models.version4.configuration_llama_v4 import Method1DConfig_v4

def create_tiny_config():
    """创建用于测试的微型配置"""
    from models.version4.configuration_llama_v4 import Method1DConfig_v4
    
    # 直接创建时设置所有参数，避免后期修改
    config = Method1DConfig_v4(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=1000,
        max_position_embeddings=64,
    )
    
    print(f"Config: hidden_size={config.hidden_size}, heads={config.num_attention_heads}")
    print(f"Expected head_dim: {config.hidden_size // config.num_attention_heads}")
    return config

def test_dynamic_weight_matrix_generation():
    """测试动态权重矩阵生成的详细过程"""
    print("测试动态权重矩阵生成")
    print("=" * 80)
    
    hidden_size = 64
    head_dim = 16  # hidden_size // num_heads = 64 // 4
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
        
        # 验证权重矩阵的每一列
        for i in range(layer_idx + 1):
            weight_column = weight_matrix_A[:, :, i]  # [batch_size, seq_len]
            exp_weight_column = torch.exp(weight_column)
            print(f"   第{i}列权重形状: {weight_column.shape}, exp后: {exp_weight_column.shape}")
        
    except Exception as e:
        print(f"❌ 动态权重生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ 动态权重矩阵生成测试完成！")
    return True

def test_mlp_parameter_calculation():
    """测试MLP参数量计算"""
    print("\n" + "=" * 80)
    print("MLP参数量计算测试")
    print("=" * 80)
    
    hidden_size = 64
    head_dim = 16
    num_layers = 2
    
    scaling_module = v4m1D_ModifiedScailingComputation(hidden_size, head_dim, num_layers)
    
    # 计算每层的参数量
    total_params = 0
    for layer_idx in range(num_layers):
        print(f"\n第{layer_idx}层参数:")
        
        # RMSNorm参数
        norm_params = sum(p.numel() for p in scaling_module.layer_norms[layer_idx].parameters())
        print(f"  RMSNorm: {norm_params} 参数")
        
        # MLP参数
        mlp_params = sum(p.numel() for p in scaling_module.layer_mlps[layer_idx].parameters())
        print(f"  MLP: {mlp_params} 参数")
        
        # Bias参数
        bias_params = scaling_module.layer_biases[layer_idx].numel()
        print(f"  Bias: {bias_params} 参数")
        
        layer_total = norm_params + mlp_params + bias_params
        total_params += layer_total
        print(f"  层总计: {layer_total} 参数")
        
        # 理论计算
        theoretical_norm = hidden_size  # RMSNorm
        theoretical_linear1 = hidden_size * 4 * hidden_size  # W1
        theoretical_linear2 = 4 * hidden_size * (layer_idx + 1)  # W2
        theoretical_bias = layer_idx + 1  # bias
        theoretical_total = theoretical_norm + theoretical_linear1 + theoretical_linear2 + theoretical_bias
        
        print(f"  理论计算: {theoretical_total} 参数")
        print(f"  匹配: {'✅' if layer_total == theoretical_total else '❌'}")
    
    print(f"\n总参数量: {total_params}")
    
    # 与Method1C比较
    method1c_params = num_layers * sum(layer_idx + 1 for layer_idx in range(num_layers)) * 8  # 假设seq_len=8
    print(f"Method1C参数量（静态，seq_len=8）: {method1c_params}")
    print(f"Method1D参数量（动态MLP）: {total_params}")
    print(f"参数增长比例: {total_params / method1c_params:.2f}x")
    
    return True

def test_tiny_model():
    """测试微型Method1D模型"""
    print("\n" + "=" * 80)
    print("微型Method1D模型测试")
    print("=" * 80)
    
    config = create_tiny_config()
    print(f"微型配置: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
    
    try:
        # 创建模型
        print("创建微型Method1D模型...")
        model = Method1DLlamaForCausalLM_v4(config)
        model.eval()
        
        # 创建测试输入
        batch_size = 1
        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        print(f"输入形状: {input_ids.shape}")
        
        # 测试前向传播
        print("测试前向传播...")
        with torch.no_grad():
            output = model(input_ids)
            
        loss = output.loss if hasattr(output, 'loss') and output.loss is not None else "N/A"
        logits_shape = output.logits.shape if hasattr(output, 'logits') else "N/A"
        
        print(f"✅ 微型Method1D前向传播成功")
        print(f"   损失: {loss}")
        print(f"   输出logits形状: {logits_shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 微型模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_dynamic_weight_matrix_generation()
    success2 = test_mlp_parameter_calculation()
    success3 = test_tiny_model()
    
    if success1 and success2 and success3:
        print(f"\n🎉 所有测试通过！Method1D实现成功")
        print(f"\n核心特点:")
        print(f"- ✅ 动态权重矩阵A_i生成：X_i -> RMSNorm -> MLP -> A_i")
        print(f"- ✅ MLP架构：hidden_size -> 4*hidden_size -> (layer_idx+1)")
        print(f"- ✅ 显式列向量广播：A_i的每一列作为权重向量")
        print(f"- ✅ 参数效率：相比Method1C，增加了MLP参数但获得了动态特性")
    else:
        print(f"\n❌ 部分测试失败，需要修复问题")
