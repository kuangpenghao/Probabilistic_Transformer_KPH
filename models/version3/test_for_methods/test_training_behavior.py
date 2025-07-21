"""
更全面的测试：验证严格causal mask对训练过程的影响
"""
import torch
import torch.nn.functional as F
import sys
import os

# 添加模型路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.version3.Method1_v3 import Method1DecoderLayer_v3
from models.version3.configuration_llama_v3 import Method1Config_v3

def test_training_behavior():
    """测试严格causal mask对训练行为的影响"""
    print("=== 测试训练行为影响 ===\n")
    
    # 创建配置
    config = Method1Config_v3(
        hidden_size=64,  # 较小的模型便于测试
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        num_hidden_layers=3,
        vocab_size=1000,
    )
    
    # 创建几个decoder layer用于测试
    layers = [Method1DecoderLayer_v3(config, i) for i in range(3)]
    
    # 模拟一个小的训练样本
    batch_size = 2
    seq_len = 8
    hidden_size = 64
    
    # 创建输入和目标
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"输入序列长度: {seq_len}")
    print(f"隐藏层大小: {hidden_size}")
    print(f"批次大小: {batch_size}")
    
    # 模拟前向传播
    def forward_pass_with_method1(use_strict_mask=True):
        """模拟Method1的前向传播"""
        hidden_states = inputs_embeds
        stored_weights = []
        
        # 创建位置嵌入（简化版）
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        cos = torch.cos(position_ids.float().unsqueeze(-1) / 10000.0)
        sin = torch.sin(position_ids.float().unsqueeze(-1) / 10000.0)
        position_embeddings = (cos, sin)
        
        for layer_idx, layer in enumerate(layers):
            # 第一层正常处理
            if layer_idx == 0:
                layer_output = layer(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    previous_mlp_outputs=[],
                    output_attentions=True
                )
            else:
                # 后续层需要重新计算前面的MLP输出
                recomputed_mlp_outputs = []
                for i in range(layer_idx):
                    weights = stored_weights[i]
                    prev_layer = layers[i]
                    
                    if weights['attn_weights'] is not None and weights['v_proj_weight'] is not None:
                        # 使用当前层输入重新计算第i层的输出
                        normalized_input = prev_layer.input_layernorm(hidden_states)
                        
                        # 关键：这里使用strict mask
                        attn_output = prev_layer.self_attn.forward_with_precomputed_weights(
                            hidden_states=normalized_input,
                            attn_weights=weights['attn_weights'],
                            v_proj_weight=weights['v_proj_weight'],
                            apply_strict_causal_mask=use_strict_mask
                        )
                        
                        layer_output_recomputed = hidden_states + attn_output
                        mlp_input = layer_output_recomputed
                        mlp_normalized = weights['post_attention_layernorm'](mlp_input)
                        mlp_output = weights['mlp'](mlp_normalized)
                        
                        recomputed_mlp_outputs.append(mlp_output)
                
                layer_output = layer(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    previous_mlp_outputs=recomputed_mlp_outputs,
                    output_attentions=True
                )
            
            hidden_states = layer_output[0]
            current_weights = layer_output[-1]
            stored_weights.append(current_weights)
        
        return hidden_states
    
    # 测试不同设置的表现
    print("\n=== 测试1: 不使用严格mask ===")
    torch.manual_seed(42)
    output_without_strict = forward_pass_with_method1(use_strict_mask=False)
    print(f"输出统计: mean={output_without_strict.mean():.4f}, std={output_without_strict.std():.4f}")
    print(f"输出范围: [{output_without_strict.min():.4f}, {output_without_strict.max():.4f}]")
    
    print("\n=== 测试2: 使用严格mask ===")
    torch.manual_seed(42)
    output_with_strict = forward_pass_with_method1(use_strict_mask=True)
    print(f"输出统计: mean={output_with_strict.mean():.4f}, std={output_with_strict.std():.4f}")
    print(f"输出范围: [{output_with_strict.min():.4f}, {output_with_strict.max():.4f}]")
    
    # 比较差异
    diff = (output_with_strict - output_without_strict).abs()
    print(f"\n输出差异分析:")
    print(f"  最大差异: {diff.max():.6f}")
    print(f"  平均差异: {diff.mean():.6f}")
    print(f"  相对差异: {(diff.mean() / output_without_strict.abs().mean() * 100):.2f}%")
    
    # 测试梯度稳定性
    def test_gradient_stability():
        print(f"\n=== 梯度稳定性测试 ===")
        
        # 创建一个简单的损失函数
        lm_head = torch.nn.Linear(hidden_size, 1000)
        
        for name, use_strict in [("不使用严格mask", False), ("使用严格mask", True)]:
            torch.manual_seed(42)
            
            # 前向传播
            output = forward_pass_with_method1(use_strict_mask=use_strict)
            logits = lm_head(output)
            
            # 计算损失
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
            
            # 反向传播
            loss.backward()
            
            # 检查梯度
            total_grad_norm = 0
            param_count = 0
            for layer in layers:
                for param in layer.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.norm().item() ** 2
                        param_count += 1
            
            avg_grad_norm = (total_grad_norm / param_count) ** 0.5 if param_count > 0 else 0
            
            print(f"  {name}:")
            print(f"    Loss: {loss.item():.6f}")
            print(f"    平均梯度范数: {avg_grad_norm:.6f}")
            
            # 清除梯度
            for layer in layers:
                layer.zero_grad()
            lm_head.zero_grad()
    
    test_gradient_stability()
    
    print(f"\n=== 结论 ===")
    print("严格causal mask的效果:")
    if diff.mean() > 1e-4:
        print("✅ 确实改变了模型行为，可能有助于防止过拟合")
    else:
        print("⚠️ 对模型行为的影响很小")
    
    print("建议:")
    print("1. 可以先试用这个修复，观察训练是否更稳定")
    print("2. 监控training loss和validation loss的变化")
    print("3. 如果过拟合问题缓解，说明修复有效")

if __name__ == "__main__":
    test_training_behavior()
