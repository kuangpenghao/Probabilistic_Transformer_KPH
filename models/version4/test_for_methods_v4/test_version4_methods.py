#!/usr/bin/env python3
"""
测试所有Version4方法的基本功能
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoConfig

# 添加项目路径
sys.path.append('/home/kuangph/hf-starter')

def test_method(method_name, config_path):
    """测试单个方法的基本功能"""
    print(f"\n=== Testing {method_name} ===")
    
    try:
        # 先导入version4以注册模型类型
        import models.version4
        
        # 导入模块
        if method_name == "Method1":
            from models.version4.Method1_v4 import Method1LlamaForCausalLM_v4 as ModelClass
            from models.version4.configuration_llama_v4 import Method1Config_v4 as ConfigClass
        elif method_name == "Method2":
            from models.version4.Method2_v4 import Method2LlamaForCausalLM_v4 as ModelClass
            from models.version4.configuration_llama_v4 import Method2Config_v4 as ConfigClass
        elif method_name == "Method3":
            from models.version4.Method3_v4 import Method3LlamaForCausalLM_v4 as ModelClass
            from models.version4.configuration_llama_v4 import Method3Config_v4 as ConfigClass
        elif method_name == "Method4":
            from models.version4.Method4_v4 import Method4LlamaForCausalLM_v4 as ModelClass
            from models.version4.configuration_llama_v4 import Method4Config_v4 as ConfigClass
        elif method_name == "Method5":
            from models.version4.Method5_v4 import Method5LlamaForCausalLM_v4 as ModelClass
            from models.version4.configuration_llama_v4 import Method5Config_v4 as ConfigClass
        elif method_name == "Method6":
            from models.version4.Method6_v4 import Method6LlamaForCausalLM_v4 as ModelClass
            from models.version4.configuration_llama_v4 import Method6Config_v4 as ConfigClass
        elif method_name == "Method7":
            from models.version4.Method7_v4 import Method7LlamaForCausalLM_v4 as ModelClass
            from models.version4.configuration_llama_v4 import Method7Config_v4 as ConfigClass
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        print(f"✓ Successfully imported {method_name}")
        
        # 直接使用配置类而不是AutoConfig
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ConfigClass(**config_dict)
        print(f"✓ Successfully created config from {config_path}")
        
        # 创建模型实例
        model = ModelClass(config)
        print(f"✓ Successfully created {method_name} model instance")
        
        # 检查模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # 简单前向传播测试
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # 简单的输入序列
        with torch.no_grad():
            outputs = model(input_ids)
            print(f"✓ Forward pass successful, output shape: {outputs.logits.shape}")
        
        print(f"✓ {method_name} test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ {method_name} test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("Version4 Methods Comprehensive Test")
    print("=" * 50)
    
    # 测试配置
    test_configs = [
        ("Method1", "/home/kuangph/hf-starter/configs/Version4_Method1.json"),
        ("Method2", "/home/kuangph/hf-starter/configs/Version4_Method2.json"),
        ("Method3", "/home/kuangph/hf-starter/configs/Version4_Method3.json"),
        ("Method4", "/home/kuangph/hf-starter/configs/Version4_Method4.json"),
        ("Method5", "/home/kuangph/hf-starter/configs/Version4_Method5.json"),
        ("Method6", "/home/kuangph/hf-starter/configs/Version4_Method6.json"),
        ("Method7", "/home/kuangph/hf-starter/configs/Version4_Method7.json"),
    ]
    
    # 运行测试
    results = []
    for method_name, config_path in test_configs:
        success = test_method(method_name, config_path)
        results.append((method_name, success))
    
    # 总结结果
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for method_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{method_name:>8}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Version4 methods are ready for use.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
