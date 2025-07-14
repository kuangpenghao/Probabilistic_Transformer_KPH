#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.version3.configuration_llama_v3 import Method1Config_v3
    print("✓ 配置模块导入成功")
    
    from models.version3.Method1_v3 import Method1LlamaForCausalLM_v3, Method1LlamaModel_v3
    print("✓ Method1_v3 模块导入成功")
    
    # 测试配置创建
    config = Method1Config_v3(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        vocab_size=1000
    )
    print("✓ 配置创建成功")
    
    print("所有测试通过！")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
