#!/usr/bin/env python
# coding=utf-8
"""
调试脚本：检查Method3_2和Method4_2的参数注册情况
"""

import torch
from models.version3.configuration_llama_v3 import Method3_2Config_v3, Method4_2Config_v3
from models.version3.Method3_2_v3 import Method3_2LlamaForCausalLM_v3
from models.version3.Method4_2_v3 import Method4_2LlamaForCausalLM_v3


def debug_model_parameters(model, model_name):
    """调试模型参数"""
    print(f"\n=== {model_name} 参数调试 ===")
    
    # 检查所有命名参数
    print("\n所有参数:")
    total_params = 0
    layer_weight_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if 'layer_weights' in name:
            layer_weight_params += 1
            print(f"  📌 {name}: {param.shape}, requires_grad={param.requires_grad}")
            print(f"     值: {param.data}")
        elif 'weight' in name and len(name.split('.')) < 6:  # 只显示主要权重
            print(f"  🔧 {name}: {param.shape}")
    
    print(f"\n总参数数: {total_params}")
    print(f"可学习权重参数数: {layer_weight_params}")
    
    # 检查layers模块
    print(f"\n模型结构:")
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        print(f"  层数: {len(layers)}")
        
        for i, layer in enumerate(layers):
            print(f"\n  Layer {i}:")
            
            # 检查MLP残差连接器
            if hasattr(layer, 'modified_residual_mlp'):
                mlp_residual = layer.modified_residual_mlp
                print(f"    MLP残差连接器: {type(mlp_residual).__name__}")
                print(f"    layer_idx: {mlp_residual.layer_idx}")
                
                if hasattr(mlp_residual, 'layer_weights'):
                    print(f"    layer_weights: {mlp_residual.layer_weights}")
                    print(f"    layer_weights.shape: {mlp_residual.layer_weights.shape}")
                    print(f"    layer_weights.requires_grad: {mlp_residual.layer_weights.requires_grad}")
                else:
                    print(f"    ❌ 没有layer_weights属性!")
            
            # 检查Attention残差连接器
            if hasattr(layer, 'modified_residual_attn'):
                attn_residual = layer.modified_residual_attn
                print(f"    Attention残差连接器: {type(attn_residual).__name__}")
                print(f"    layer_idx: {attn_residual.layer_idx}")
                
                if hasattr(attn_residual, 'layer_weights'):
                    print(f"    layer_weights: {attn_residual.layer_weights}")
                    print(f"    layer_weights.shape: {attn_residual.layer_weights.shape}")
                    print(f"    layer_weights.requires_grad: {attn_residual.layer_weights.requires_grad}")
                else:
                    print(f"    ❌ 没有layer_weights属性!")


def test_forward_pass(model, model_name):
    """测试前向传播"""
    print(f"\n=== {model_name} 前向传播测试 ===")
    
    # 创建测试输入
    batch_size = 2
    seq_len = 8
    vocab_size = model.config.vocab_size
    
    input_ids = torch.randint(1, vocab_size-1, (batch_size, seq_len))
    
    print(f"输入形状: {input_ids.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        print(f"输出形状: {outputs.logits.shape}")
        print(f"输出范围: [{outputs.logits.min():.3f}, {outputs.logits.max():.3f}]")
    
    # 检查权重
    if hasattr(model, 'get_all_layer_weights'):
        weights = model.get_all_layer_weights()
        print(f"权重层数: {len(weights)}")
        for i, w in enumerate(weights):
            if len(w) > 0:
                print(f"  Layer {i}: {w}")


def main():
    print("开始模型参数调试...")
    
    # 创建小型配置
    config3_2 = Method3_2Config_v3(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=2,
        max_position_embeddings=128,
        torch_dtype="float32"
    )
    
    config4_2 = Method4_2Config_v3(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=2,
        max_position_embeddings=128,
        torch_dtype="float32"
    )
    
    # 创建模型
    print("创建Method3_2模型...")
    model3_2 = Method3_2LlamaForCausalLM_v3(config3_2)
    
    print("创建Method4_2模型...")
    model4_2 = Method4_2LlamaForCausalLM_v3(config4_2)
    
    # 调试参数
    debug_model_parameters(model3_2, "Method3_2")
    debug_model_parameters(model4_2, "Method4_2")
    
    # 测试前向传播
    test_forward_pass(model3_2, "Method3_2")
    test_forward_pass(model4_2, "Method4_2")


if __name__ == "__main__":
    main()
