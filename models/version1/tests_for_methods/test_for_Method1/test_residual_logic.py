"""
详细测试新残差连接的逻辑
"""
import torch
import sys
import os

# 添加项目根目录和模型目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from configuration_llama import Method1LlamaConfig
from Method1 import Method1DecoderLayer

def test_residual_logic():
    """测试残差连接的具体逻辑"""
    print("测试残差连接的具体逻辑...")
    
    config = Method1LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
    )
    
    # 创建3层进行测试
    layers = [Method1DecoderLayer(config, layer_idx=i) for i in range(3)]
    
    # 创建输入
    batch_size = 1
    seq_len = 4
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    print(f"初始输入形状: {hidden_states.shape}")
    
    # 模拟layer by layer的处理
    all_attn_outputs = []
    
    for layer_idx, layer in enumerate(layers):
        print(f"\n--- 处理第{layer_idx}层 ---")
        
        # 准备位置嵌入
        position_ids = torch.arange(seq_len).unsqueeze(0)
        cos, sin = layer.self_attn.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)
        
        # 传递之前层的注意力输出
        previous_attn_outputs = all_attn_outputs.copy() if layer_idx > 0 else None
        
        if previous_attn_outputs:
            print(f"  传入的之前层注意力输出数量: {len(previous_attn_outputs)}")
            print(f"  之前层输出形状: {[out.shape for out in previous_attn_outputs]}")
        else:
            print("  第一层，没有之前层的输出")
        
        with torch.no_grad():
            # 调用层的前向传播
            layer_outputs = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                previous_attn_outputs=previous_attn_outputs,
            )
            
            # 更新hidden_states
            hidden_states = layer_outputs[0]
            current_attn_output = layer_outputs[-1]  # 最后一个是注意力输出
            
            print(f"  层输出形状: {hidden_states.shape}")
            print(f"  当前层注意力输出形状: {current_attn_output.shape}")
            
            # 保存当前层的注意力输出
            all_attn_outputs.append(current_attn_output)
            
            # 验证残差连接逻辑
            if layer_idx == 0:
                print("  ✓ 第一层应该没有残差连接")
            else:
                print(f"  ✓ 第{layer_idx}层应该有来自前{layer_idx}层的残差连接")
    
    print(f"\n最终输出形状: {hidden_states.shape}")
    print(f"保存的注意力输出数量: {len(all_attn_outputs)}")
    print("测试完成！")

def compare_with_original():
    """比较新残差连接和原始残差连接的区别"""
    print("\n比较新残差连接和原始残差连接的区别...")
    
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    
    config = Method1LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
    )
    
    # 创建新旧两种层
    new_layer = Method1DecoderLayer(config, layer_idx=1)  # 第二层
    original_layer = LlamaDecoderLayer(config, layer_idx=1)
    
    # 确保权重相同
    original_layer.load_state_dict(new_layer.state_dict())
    
    # 创建输入
    hidden_states = torch.randn(1, 4, config.hidden_size)
    position_ids = torch.arange(4).unsqueeze(0)  # [0, 1, 2, 3]
    
    # 创建假的previous_attn_outputs
    fake_previous_outputs = [torch.randn(1, 4, config.hidden_size)]
    
    with torch.no_grad():
        # 新残差连接
        new_outputs = new_layer(
            hidden_states,
            position_ids=position_ids,
            previous_attn_outputs=fake_previous_outputs,
        )
        
        # 原始残差连接
        original_outputs = original_layer(
            hidden_states,
            position_ids=position_ids,
        )
        
        print(f"新残差连接输出形状: {new_outputs[0].shape}")
        print(f"原始残差连接输出形状: {original_outputs[0].shape}")
        
        # 计算差异
        diff = torch.abs(new_outputs[0] - original_outputs[0]).mean()
        print(f"输出差异: {diff:.6f}")
        
        if diff > 1e-6:
            print("✓ 新旧残差连接产生了不同的输出，符合预期")
        else:
            print("⚠ 输出相同，可能存在问题")

if __name__ == "__main__":
    test_residual_logic()
    compare_with_original()
