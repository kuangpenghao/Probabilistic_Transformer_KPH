#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/kuangph/hf-starter')

# 简单测试保存功能
try:
    from models.version2.Method3_v2 import Method3LlamaForCausalLM_v2, Method3Config_v2
    import torch
    
    print("创建配置...")
    config = Method3Config_v2(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=100
    )
    
    print("创建模型...")
    model = Method3LlamaForCausalLM_v2(config)
    
    print("保存权重参数...")
    save_path = model.save_learned_parameters('/home/kuangph/hf-starter/models/version2')
    
    print(f"成功！文件保存到: {save_path}")
    
    if os.path.exists(save_path):
        print("文件确实存在")
        with open(save_path, 'r') as f:
            content = f.read()
            print(f"文件长度: {len(content)} 字符")
            print("前200个字符:")
            print(content[:200])
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
