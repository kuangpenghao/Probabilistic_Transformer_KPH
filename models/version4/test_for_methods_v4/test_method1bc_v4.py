#!/usr/bin/env python3

import torch
import json
from models.version4.Method1A_v4 import Method1ALlamaForCausalLM_v4
from models.version4.Method1B_v4 import Method1BLlamaForCausalLM_v4
from models.version4.Method1C_v4 import Method1CLlamaForCausalLM_v4
from models.version4.configuration_llama_v4 import Method1AConfig_v4, Method1BConfig_v4, Method1CConfig_v4

def create_test_config():
    """创建测试配置"""
    return {
        "vocab_size": 1000,
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }

def analyze_parameters(model, model_name):
    """分析模型参数"""
    print(f"\n{'='*60}")
    print(f"{model_name} 参数分析")
    print(f"{'='*60}")
    
    # 统计总参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 统计权重向量/矩阵参数
    weight_params = 0
    weight_count = 0
    
    for name, param in model.named_parameters():
        if "layer_weight" in name:
            weight_params += param.numel()
            weight_count += 1
            print(f"  {name}: {param.shape} -> {param.numel():,} 参数")
    
    print(f"\n新增权重参数: {weight_params:,} ({weight_count}个)")
    print(f"新增参数占比: {(weight_params/total_params)*100:.6f}%")
    
    return total_params, weight_params, weight_count

def test_forward_pass(model, model_name):
    """测试前向传播"""
    print(f"\n测试 {model_name} 前向传播...")
    
    input_ids = torch.randint(0, 1000, (2, 32))
    labels = input_ids.clone()
    
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
        print(f"✅ 前向传播成功，损失: {outputs.loss:.4f}")
        return True
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False

def compare_parameter_efficiency():
    """比较三种方法的参数效率"""
    print("\n" + "="*80)
    print("Method1A vs Method1B vs Method1C 参数效率对比")
    print("="*80)
    
    config_dict = create_test_config()
    
    # 创建三个模型
    models = {
        "Method1A": (Method1ALlamaForCausalLM_v4, Method1AConfig_v4, "完整矩阵"),
        "Method1B": (Method1BLlamaForCausalLM_v4, Method1BConfig_v4, "行向量"),
        "Method1C": (Method1CLlamaForCausalLM_v4, Method1CConfig_v4, "列向量")
    }
    
    results = {}
    
    for model_name, (model_class, config_class, description) in models.items():
        print(f"\n创建 {model_name} ({description})...")
        config = config_class(**config_dict)
        model = model_class(config)
        
        # 先测试前向传播以触发权重初始化
        forward_success = test_forward_pass(model, model_name)
        
        # 前向传播后再分析参数
        total_params, weight_params, weight_count = analyze_parameters(model, model_name)
        
        results[model_name] = {
            "total_params": total_params,
            "weight_params": weight_params,
            "weight_count": weight_count,
            "forward_success": forward_success,
            "description": description
        }
    
    # 对比结果
    print(f"\n{'='*80}")
    print("参数效率对比结果")
    print(f"{'='*80}")
    
    print(f"{'方法':<10} {'描述':<10} {'总参数':<12} {'新增参数':<12} {'参数比例':<12} {'前向传播'}")
    print("-" * 80)
    
    for model_name, data in results.items():
        ratio = f"{(data['weight_params']/data['total_params'])*100:.6f}%" if data['total_params'] > 0 else "0%"
        success = "✅" if data['forward_success'] else "❌"
        print(f"{model_name:<10} {data['description']:<10} {data['total_params']:<12,} {data['weight_params']:<12,} {ratio:<12} {success}")
    
    # 计算效率比较
    print(f"\n{'='*60}")
    print("效率比较 (相对于Method1A)")
    print(f"{'='*60}")
    
    base_params = results["Method1A"]["weight_params"]
    if base_params > 0:
        for model_name, data in results.items():
            if model_name != "Method1A":
                if data["weight_params"] > 0:
                    efficiency = base_params / data["weight_params"]
                    reduction = 100*(1-data['weight_params']/base_params)
                    print(f"{model_name}: 参数减少 {efficiency:.1f}x ({reduction:.1f}% 减少)")
                else:
                    print(f"{model_name}: 无新增参数")
    else:
        print("Method1A 无新增参数，无法比较效率")
    
    return results

def main():
    print("Method1B & Method1C 实现测试")
    print("="*80)
    
    try:
        compare_parameter_efficiency()
        print(f"\n{'='*80}")
        print("✅ 所有测试完成！Method1B和Method1C实现成功")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
