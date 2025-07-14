#!/usr/bin/env python3
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_optimized_method1_v3():
    print("测试优化后的 Method1_v3...")
    
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
        
        # 创建测试输入
        batch_size = 2
        seq_length = 8
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
        
        # 测试自定义attention方法
        attention = model.model.layers[0].self_attn
        hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
        v_weight = attention.v_proj.weight
        
        # 创建假的Q、K矩阵用于测试
        query_states = torch.randn(batch_size, config.num_attention_heads, seq_length, config.hidden_size // config.num_attention_heads)
        key_states = torch.randn(batch_size, config.num_key_value_heads, seq_length, config.hidden_size // config.num_key_value_heads)
        
        # 测试forward_with_precomputed_qkv方法
        attn_output = attention.forward_with_precomputed_qkv(
            hidden_states=hidden_states,
            precomputed_query_states=query_states,
            precomputed_key_states=key_states,
            v_proj_weight=v_weight,
            attention_mask=None,
            position_embeddings=None,
        )
        print(f"✓ forward_with_precomputed_qkv 成功, 输出shape: {attn_output.shape}")
        
        print("\n🎉 所有测试通过！优化后的Method1_v3工作正常！")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optimized_method1_v3()
    if success:
        print("\n✅ 优化后的Method1_v3测试完成，功能正常！")
    else:
        print("\n❌ 优化存在问题，请检查代码。")
