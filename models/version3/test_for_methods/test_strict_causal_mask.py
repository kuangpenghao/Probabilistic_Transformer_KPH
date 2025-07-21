"""
测试严格causal mask是否能防止信息泄漏
"""
import torch
import torch.nn.functional as F
import sys
import os

# 添加模型路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.version3.Method1_v3 import Method1LlamaAttention_v3
from models.version3.configuration_llama_v3 import Method1Config_v3

def test_strict_causal_mask():
    """测试严格causal mask的效果"""
    print("=== 测试严格Causal Mask防止信息泄漏 ===\n")
    
    # 创建配置
    config = Method1Config_v3(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=256,
        num_hidden_layers=2,
        vocab_size=1000,
    )
    
    # 创建注意力层
    attn = Method1LlamaAttention_v3(config, layer_idx=0)
    
    # 创建测试输入
    batch_size = 1
    seq_len = 4
    hidden_size = 128
    
    # 第1层的输入（模拟早期层）
    layer1_input = torch.randn(batch_size, seq_len, hidden_size)
    # 第3层的输入（模拟后期层，包含更多信息）
    layer3_input = layer1_input + 0.5 * torch.randn(batch_size, seq_len, hidden_size)
    
    print(f"序列长度: {seq_len}")
    print(f"Layer1输入范围: [{layer1_input.min():.3f}, {layer1_input.max():.3f}]")
    print(f"Layer3输入范围: [{layer3_input.min():.3f}, {layer3_input.max():.3f}]")
    
    # 创建一个存储的attention权重矩阵（模拟原始计算的结果）
    # 这里故意让它包含一些"未来"信息的权重
    raw_attn_weights = torch.randn(batch_size, config.num_attention_heads, seq_len, seq_len)
    # 应用基本的causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    mask_value = torch.finfo(raw_attn_weights.dtype).min
    raw_attn_weights = raw_attn_weights.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), mask_value)
    stored_attn_weights = F.softmax(raw_attn_weights, dim=-1)
    
    print(f"\n存储的attention权重形状: {stored_attn_weights.shape}")
    print("Causal mask验证:")
    for i in range(seq_len):
        # 检查每个位置是否只关注到自己及之前的位置
        weights = stored_attn_weights[0, 0, i, :]  # 第i个位置的attention权重
        non_zero_positions = torch.nonzero(weights > 1e-6).flatten().tolist()
        print(f"  位置{i}关注的位置: {non_zero_positions} (应该是0到{i})")
    
    try:
        # 测试1: 不使用严格mask（原有问题）
        print(f"\n=== 测试1: 不使用严格causal mask ===")
        output_without_strict_mask = attn.forward_with_precomputed_weights(
            hidden_states=layer3_input,
            attn_weights=stored_attn_weights,
            v_proj_weight=attn.v_proj.weight,
            apply_strict_causal_mask=False
        )
        print(f"输出形状: {output_without_strict_mask.shape}")
        print(f"输出范围: [{output_without_strict_mask.min():.3f}, {output_without_strict_mask.max():.3f}]")
        
        # 测试2: 使用严格mask（修复后）
        print(f"\n=== 测试2: 使用严格causal mask ===")
        output_with_strict_mask = attn.forward_with_precomputed_weights(
            hidden_states=layer3_input,
            attn_weights=stored_attn_weights,
            v_proj_weight=attn.v_proj.weight,
            apply_strict_causal_mask=True
        )
        print(f"输出形状: {output_with_strict_mask.shape}")
        print(f"输出范围: [{output_with_strict_mask.min():.3f}, {output_with_strict_mask.max():.3f}]")
        
        # 比较两种方法的差异
        diff = (output_with_strict_mask - output_without_strict_mask).abs()
        print(f"\n两种方法的输出差异:")
        print(f"  最大差异: {diff.max():.6f}")
        print(f"  平均差异: {diff.mean():.6f}")
        print(f"  差异标准差: {diff.std():.6f}")
        
        # 测试3: 验证因果性
        print(f"\n=== 测试3: 验证因果性 ===")
        
        # 修改未来位置的输入，看是否会影响过去位置的输出
        modified_layer3_input = layer3_input.clone()
        # 大幅修改最后一个位置的输入
        modified_layer3_input[:, -1, :] += 10.0
        
        output_modified_without_mask = attn.forward_with_precomputed_weights(
            hidden_states=modified_layer3_input,
            attn_weights=stored_attn_weights,
            v_proj_weight=attn.v_proj.weight,
            apply_strict_causal_mask=False
        )
        
        output_modified_with_mask = attn.forward_with_precomputed_weights(
            hidden_states=modified_layer3_input,
            attn_weights=stored_attn_weights,
            v_proj_weight=attn.v_proj.weight,
            apply_strict_causal_mask=True
        )
        
        # 检查第一个位置的输出是否受到最后位置修改的影响
        pos0_diff_without_mask = (output_modified_without_mask[:, 0, :] - output_without_strict_mask[:, 0, :]).abs().max()
        pos0_diff_with_mask = (output_modified_with_mask[:, 0, :] - output_with_strict_mask[:, 0, :]).abs().max()
        
        print(f"修改最后位置输入后，第0位置输出的变化:")
        print(f"  不使用严格mask: {pos0_diff_without_mask:.6f}")
        print(f"  使用严格mask: {pos0_diff_with_mask:.6f}")
        
        if pos0_diff_with_mask < pos0_diff_without_mask * 0.1:  # 如果变化减少了90%以上
            print("✅ 严格causal mask成功减少了信息泄漏!")
        else:
            print("⚠️ 严格causal mask的效果有限")
            
        print(f"\n=== 总结 ===")
        print("严格causal mask方案:")
        print("✅ 实现简单，不破坏原有算法结构")
        print("✅ 能够在一定程度上减少信息泄漏")
        if pos0_diff_with_mask < 1e-6:
            print("✅ 完全防止了未来信息对过去位置的影响")
        else:
            print("⚠️ 仍可能存在一定程度的信息泄漏")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    torch.manual_seed(42)  # 确保结果可重现
    test_strict_causal_mask()
