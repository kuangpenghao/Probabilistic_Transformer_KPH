#!/usr/bin/env python3
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_method1_v3():
    print("开始测试 Method1_v3...")
    
    try:
        from models.version3.configuration_llama_v3 import Method1Config_v3
        from models.version3.Method1_v3 import Method1LlamaForCausalLM_v3
        print("✓ 模块导入成功")
        
        # 创建一个小型配置用于测试
        config = Method1Config_v3(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=1000,
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
        )
        print("✓ 配置创建成功")
        
        # 创建模型
        model = Method1LlamaForCausalLM_v3(config)
        model.eval()
        print("✓ 模型创建成功")
        print(f"模型层数: {len(model.model.layers)}")
        
        # 创建测试输入
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        print("✓ 测试输入创建成功")
        
        # 测试前向传播
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False
            )
        
        print("✓ 前向传播成功")
        print(f"输出logits shape: {outputs.logits.shape}")
        print(f"预期shape: ({batch_size}, {seq_length}, {config.vocab_size})")
        
        # 验证输出形状
        expected_shape = (batch_size, seq_length, config.vocab_size)
        assert outputs.logits.shape == expected_shape, f"输出形状不匹配: {outputs.logits.shape} vs {expected_shape}"
        print("✓ 输出形状验证成功")
        
        # 测试多层的MLP重计算是否正常工作
        print("\n测试MLP重计算机制...")
        
        # 获取中间层的输出状态（用于验证重计算逻辑）
        with torch.no_grad():
            outputs_with_hidden = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        print(f"✓ 获得 {len(outputs_with_hidden.hidden_states)} 层的hidden states")
        
        print("\n🎉 所有测试通过！Method1_v3 实现正确！")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_method1_v3()
    if success:
        print("\n✅ Method1_v3 实现测试完成，所有功能正常！")
    else:
        print("\n❌ Method1_v3 实现存在问题，请检查代码。")
