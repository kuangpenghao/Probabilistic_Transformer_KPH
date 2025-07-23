#!/usr/bin/env python
# coding=utf-8
"""
测试run_clm.py的权重保存功能
"""

import os
import tempfile
import json
from models.version3.configuration_llama_v3 import Method3_2Config_v3, Method4_2Config_v3
from models.version3.Method3_2_v3 import Method3_2LlamaForCausalLM_v3
from models.version3.Method4_2_v3 import Method4_2LlamaForCausalLM_v3


def test_weight_saving_logic():
    """测试权重保存逻辑（模拟run_clm.py中的逻辑）"""
    print("=== 测试权重保存逻辑 ===")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"临时目录: {temp_dir}")
        
        # 测试Method3_2
        config3_2 = Method3_2Config_v3(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=2,
            max_position_embeddings=64,
            torch_dtype="float32"
        )
        
        model3_2 = Method3_2LlamaForCausalLM_v3(config3_2)
        print(f"\n创建 Method3_2 模型: {model3_2.__class__.__name__}")
        
        # 模拟run_clm.py中的权重保存逻辑
        model_to_check = model3_2
        model_class_name = model_to_check.__class__.__name__
        training_args_output_dir = temp_dir
        
        if hasattr(model_to_check, 'get_all_layer_weights'):
            print(f"✅ 检测到具有可学习权重的模型 ({model_class_name})")
            
            # 获取所有层的权重分布
            layer_weights = model_to_check.get_all_layer_weights()
            
            # 保存权重到文件
            weights_data = {}
            
            for layer_idx, weights in enumerate(layer_weights):
                if len(weights) > 0:
                    weights_data[f"layer_{layer_idx}"] = weights.cpu().numpy().tolist()
            
            # 确定保存路径和文件名
            if "Method3_2" in model_class_name:
                weights_file = os.path.join(training_args_output_dir, "method3_2_learned_weights.json")
                print("保存Method3_2模型的MLP权重分布...")
            elif "Method4_2" in model_class_name:
                weights_file = os.path.join(training_args_output_dir, "method4_2_learned_weights.json")
                print("保存Method4_2模型的Attention权重分布...")
            else:
                weights_file = os.path.join(training_args_output_dir, "learned_weights.json")
                print(f"保存{model_class_name}模型的可学习权重分布...")
            
            # 保存权重数据
            with open(weights_file, 'w') as f:
                json.dump(weights_data, f, indent=2)
            
            print(f"✅ 可学习权重参数已保存到: {weights_file}")
            
            # 验证文件内容
            with open(weights_file, 'r') as f:
                saved_data = json.load(f)
            
            print(f"保存的权重数据:")
            for layer_name, weights in saved_data.items():
                print(f"  {layer_name}: {weights}")
            
            # 额外保存权重统计信息
            stats_file = weights_file.replace('.json', '_stats.txt')
            with open(stats_file, 'w') as f:
                f.write(f"Model: {model_class_name}\n")
                f.write(f"Total layers with weights: {len(weights_data)}\n")
                f.write("="*50 + "\n")
                
                for layer_idx, weights in enumerate(layer_weights):
                    if len(weights) > 0:
                        weights_np = weights.cpu().numpy()
                        f.write(f"Layer {layer_idx}:\n")
                        f.write(f"  Weights: {weights_np}\n")
                        f.write(f"  Shape: {weights_np.shape}\n")
                        f.write(f"  Mean: {weights_np.mean():.6f}\n")
                        f.write(f"  Std: {weights_np.std():.6f}\n")
                        f.write(f"  Min: {weights_np.min():.6f}\n")
                        f.write(f"  Max: {weights_np.max():.6f}\n")
                        f.write("-" * 30 + "\n")
            
            print(f"✅ 权重统计信息已保存到: {stats_file}")
            
            # 显示统计文件内容
            with open(stats_file, 'r') as f:
                print(f"\n统计文件内容:")
                print(f.read())
        
        # 测试Method4_2
        print(f"\n{'='*50}")
        config4_2 = Method4_2Config_v3(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=2,
            max_position_embeddings=64,
            torch_dtype="float32"
        )
        
        model4_2 = Method4_2LlamaForCausalLM_v3(config4_2)
        print(f"创建 Method4_2 模型: {model4_2.__class__.__name__}")
        
        # 重复相同的逻辑
        model_to_check = model4_2
        model_class_name = model_to_check.__class__.__name__
        
        if hasattr(model_to_check, 'get_all_layer_weights'):
            print(f"✅ 检测到具有可学习权重的模型 ({model_class_name})")
            
            layer_weights = model_to_check.get_all_layer_weights()
            weights_data = {}
            
            for layer_idx, weights in enumerate(layer_weights):
                if len(weights) > 0:
                    weights_data[f"layer_{layer_idx}"] = weights.cpu().numpy().tolist()
            
            if "Method4_2" in model_class_name:
                weights_file = os.path.join(training_args_output_dir, "method4_2_learned_weights.json")
                print("保存Method4_2模型的Attention权重分布...")
            
            with open(weights_file, 'w') as f:
                json.dump(weights_data, f, indent=2)
            
            print(f"✅ 可学习权重参数已保存到: {weights_file}")
            
            with open(weights_file, 'r') as f:
                saved_data = json.load(f)
            
            print(f"保存的权重数据:")
            for layer_name, weights in saved_data.items():
                print(f"  {layer_name}: {weights}")


if __name__ == "__main__":
    test_weight_saving_logic()
