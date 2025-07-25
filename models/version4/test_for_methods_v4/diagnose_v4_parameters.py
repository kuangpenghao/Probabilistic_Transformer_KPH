#!/usr/bin/env python3
"""
诊断Version4方法的可学习参数问题
"""

import torch
import sys
sys.path.append('/home/kuangph/hf-starter')

def diagnose_learnable_parameters():
    """诊断可学习参数是否正确注册和参与训练"""
    print("=== Version4 可学习参数诊断 ===\n")
    
    methods_info = [
        ("Method4", "models.version4.Method4_v4", "Method4Config_v4", "Method4LlamaForCausalLM_v4"),
        ("Method5", "models.version4.Method5_v4", "Method5Config_v4", "Method5LlamaForCausalLM_v4"),
        ("Method6", "models.version4.Method6_v4", "Method6Config_v4", "Method6LlamaForCausalLM_v4"),
        ("Method7", "models.version4.Method7_v4", "Method7Config_v4", "Method7LlamaForCausalLM_v4"),
    ]
    
    all_issues = []
    
    for method_name, module_path, config_class_name, model_class_name in methods_info:
        try:
            print(f"🔍 诊断 {method_name}:")
            
            # 动态导入
            module = __import__(module_path, fromlist=[config_class_name, model_class_name])
            config_class = getattr(module, config_class_name)
            model_class = getattr(module, model_class_name)
            
            # 创建小型测试配置
            config = config_class(
                vocab_size=1000,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=3,
                num_attention_heads=4,
                max_position_embeddings=512
            )
            
            # 创建模型实例
            model = model_class(config)
            
            # 1. 检查模型的所有参数
            all_params = list(model.parameters())
            total_params = sum(p.numel() for p in all_params)
            print(f"  📊 模型总参数数: {total_params:,}")
            
            # 2. 查找可学习缩放参数
            scaling_params = []
            scaling_param_names = []
            
            for name, param in model.named_parameters():
                if any(keyword in name for keyword in ['modified_scaling', 'log_a_params', 'a_params', 'b_params', '.a', '.b']):
                    scaling_params.append(param)
                    scaling_param_names.append(name)
            
            print(f"  🎯 找到缩放参数: {len(scaling_params)} 个")
            for name in scaling_param_names:
                print(f"    - {name}")
            
            if len(scaling_params) == 0:
                print("  ❌ 严重问题：没有找到任何缩放参数！")
                all_issues.append(f"{method_name}: 没有找到缩放参数")
                continue
            
            # 3. 检查参数是否需要梯度
            requires_grad_count = sum(1 for p in scaling_params if p.requires_grad)
            print(f"  🔄 需要梯度的缩放参数: {requires_grad_count}/{len(scaling_params)}")
            
            if requires_grad_count != len(scaling_params):
                print("  ⚠️  警告：部分缩放参数不需要梯度！")
                all_issues.append(f"{method_name}: 部分参数不需要梯度")
            
            # 4. 检查参数初始值
            print("  📋 参数初始值:")
            for name, param in zip(scaling_param_names, scaling_params):
                if param.numel() <= 10:  # 只显示小参数
                    print(f"    {name}: {param.data}")
                else:
                    print(f"    {name}: shape={param.shape}, mean={param.data.mean():.4f}")
            
            # 5. 模拟前向传播，检查参数是否参与计算
            print("  🔄 测试前向传播...")
            
            # 创建虚拟输入
            batch_size, seq_len = 2, 10
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            
            # 设置梯度追踪
            for param in scaling_params:
                param.requires_grad_(True)
            
            # 前向传播
            model.train()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            
            # 检查梯度
            grad_count = 0
            for name, param in zip(scaling_param_names, scaling_params):
                if param.grad is not None:
                    grad_count += 1
                    grad_norm = param.grad.norm().item()
                    print(f"    {name}: grad_norm={grad_norm:.6f}")
                else:
                    print(f"    {name}: ❌ 没有梯度！")
            
            if grad_count == 0:
                print("  ❌ 严重问题：所有缩放参数都没有梯度！")
                all_issues.append(f"{method_name}: 参数没有参与梯度计算")
            elif grad_count < len(scaling_params):
                print("  ⚠️  警告：部分缩放参数没有梯度！")
                all_issues.append(f"{method_name}: 部分参数没有梯度")
            else:
                print("  ✅ 所有缩放参数都有梯度")
            
            # 6. 检查参数值是否会变化
            old_values = [p.data.clone() for p in scaling_params]
            
            # 模拟优化器步骤
            optimizer = torch.optim.AdamW(scaling_params, lr=0.01)
            optimizer.step()
            
            changed_count = 0
            for i, (param, old_val) in enumerate(zip(scaling_params, old_values)):
                if not torch.equal(param.data, old_val):
                    changed_count += 1
            
            print(f"  📈 优化器步骤后变化的参数: {changed_count}/{len(scaling_params)}")
            
            if changed_count == 0:
                print("  ❌ 严重问题：优化器步骤后参数没有变化！")
                all_issues.append(f"{method_name}: 参数不会被优化器更新")
            
            print("  ✅ 参数诊断完成\n")
                
        except Exception as e:
            print(f"  ❌ {method_name} 诊断失败: {str(e)}")
            all_issues.append(f"{method_name}: 诊断失败 - {str(e)}")
            print()
    
    # 总结报告
    print("=" * 60)
    print("🔍 诊断总结:")
    
    if not all_issues:
        print("🎉 所有方法的可学习参数都工作正常！")
    else:
        print("❌ 发现以下问题:")
        for issue in all_issues:
            print(f"  - {issue}")
        
        print("\n💡 可能的解决方案:")
        print("1. 检查参数是否正确注册为nn.Parameter")
        print("2. 检查forward函数中是否使用了这些参数")
        print("3. 检查参数是否被正确传递到计算图中")
        print("4. 检查优化器是否包含了这些参数")
    
    return len(all_issues) == 0

if __name__ == "__main__":
    success = diagnose_learnable_parameters()
    if not success:
        print("\n🚨 需要修复参数问题后才能正常训练！")
    else:
        print("\n🚀 可学习参数诊断通过，可以开始训练！")
