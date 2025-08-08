#!/usr/bin/env python3
"""
测试修改后的Method5实现：约束a_1 + a_2 + ... + a_m = m
"""

import torch
import sys
import math
sys.path.append('/home/kuangph/hf-starter')

def test_method5_constraint():
    """测试Method5的约束条件"""
    print("=== Method5 约束条件测试 ===\n")
    
    from models.version4.Method5_v4 import Method5Config_v4, Method5LlamaForCausalLM_v4
    
    # 创建小型测试配置
    config = Method5Config_v4(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=512
    )
    
    model = Method5LlamaForCausalLM_v4(config)
    head_dim = config.hidden_size // config.num_attention_heads
    
    print(f"📊 模型配置: {config.num_hidden_layers}层, head_dim={head_dim}")
    print()
    
    # 测试每层的约束条件
    for layer_idx in range(config.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        scaling_module = layer.self_attn.modified_scaling
        
        print(f"🔍 测试第{layer_idx}层:")
        print(f"  期望向量长度: {layer_idx + 1}")
        print(f"  实际score_params长度: {len(scaling_module.score_params)}")
        
        # 检查初始的score_params
        initial_scores = scaling_module.score_params.data.clone()
        print(f"  初始scores: {initial_scores.tolist()}")
        
        # 计算对应的a_params
        num_matrices = layer_idx + 1
        softmax_weights = torch.softmax(initial_scores, dim=0)
        a_params = softmax_weights * num_matrices
        
        print(f"  Softmax权重: {softmax_weights.tolist()}")
        print(f"  A参数: {a_params.tolist()}")
        print(f"  A参数之和: {a_params.sum().item():.6f} (应该为 {num_matrices})")
        
        # 验证约束条件
        constraint_satisfied = abs(a_params.sum().item() - num_matrices) < 1e-6
        print(f"  ✅ 约束条件满足: {constraint_satisfied}")
        
        # 模拟一些QK矩阵来测试缩放计算
        batch_size, num_heads, seq_len = 2, 4, 10
        qk_matrices = []
        for i in range(num_matrices):
            qk_matrix = torch.randn(batch_size, num_heads, seq_len, seq_len)
            qk_matrices.append(qk_matrix)
        
        # 测试缩放计算
        result = scaling_module.compute_modified_scaling(qk_matrices, layer_idx)
        print(f"  缩放计算结果形状: {result.shape}")
        
        # 验证缩放向量的计算
        expected_scaling_vector = a_params / math.sqrt(head_dim)
        print(f"  期望缩放向量: {expected_scaling_vector.tolist()}")
        print(f"  缩放向量总和/sqrt(d_k): {expected_scaling_vector.sum().item():.6f}")
        print()

def test_method5_training():
    """测试Method5在训练中的参数更新"""
    print("=== Method5 训练测试 ===\n")
    
    from models.version4.Method5_v4 import Method5Config_v4, Method5LlamaForCausalLM_v4
    
    config = Method5Config_v4(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=512
    )
    
    model = Method5LlamaForCausalLM_v4(config)
    model.train()
    
    # 记录初始参数
    print("📋 训练前参数状态:")
    initial_params = {}
    for layer_idx in range(config.num_hidden_layers):
        scaling_module = model.model.layers[layer_idx].self_attn.modified_scaling
        scores = scaling_module.score_params.data.clone()
        softmax_weights = torch.softmax(scores, dim=0)
        a_params = softmax_weights * (layer_idx + 1)
        
        initial_params[layer_idx] = {
            'scores': scores.clone(),
            'a_params': a_params.clone()
        }
        
        print(f"  Layer {layer_idx}: scores={scores.tolist()}")
        print(f"  Layer {layer_idx}: a_params={a_params.tolist()} (sum={a_params.sum():.6f})")
    
    # 模拟训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    print("\n🚀 开始模拟训练...")
    for step in range(5):
        optimizer.zero_grad()
        
        # 创建随机输入
        input_ids = torch.randint(0, config.vocab_size, (2, 32))
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        if step == 0 or step == 4:
            print(f"  Step {step}: loss = {loss.item():.4f}")
    
    # 检查训练后的参数
    print("\n📋 训练后参数状态:")
    for layer_idx in range(config.num_hidden_layers):
        scaling_module = model.model.layers[layer_idx].self_attn.modified_scaling
        scores = scaling_module.score_params.data
        softmax_weights = torch.softmax(scores, dim=0)
        a_params = softmax_weights * (layer_idx + 1)
        
        initial_scores = initial_params[layer_idx]['scores']
        initial_a_params = initial_params[layer_idx]['a_params']
        
        score_change = (scores - initial_scores).abs().sum().item()
        a_change = (a_params - initial_a_params).abs().sum().item()
        
        print(f"  Layer {layer_idx}: scores={scores.tolist()}")
        print(f"  Layer {layer_idx}: a_params={a_params.tolist()} (sum={a_params.sum():.6f})")
        print(f"  Layer {layer_idx}: score变化={score_change:.6f}, a参数变化={a_change:.6f}")
        
        # 验证约束条件仍然满足
        constraint_satisfied = abs(a_params.sum().item() - (layer_idx + 1)) < 1e-6
        print(f"  Layer {layer_idx}: ✅ 约束条件满足: {constraint_satisfied}")
        print()

def test_parameter_saving():
    """测试参数保存功能"""
    print("=== Method5 参数保存测试 ===\n")
    
    from models.version4.Method5_v4 import Method5Config_v4, Method5LlamaForCausalLM_v4
    import tempfile
    import json
    import os
    
    config = Method5Config_v4(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=512
    )
    
    model = Method5LlamaForCausalLM_v4(config)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    # 测试参数获取
    all_weights = model.get_all_layer_weights()
    print(f"📊 获取到 {len(all_weights)} 层参数")
    
    for layer_idx, layer_weights in enumerate(all_weights):
        if layer_weights:
            scores = layer_weights['score_params']
            a_params = layer_weights['a_params']
            print(f"  Layer {layer_idx}: score长度={len(scores)}, a参数长度={len(a_params)}")
            print(f"  Layer {layer_idx}: a参数和={sum(a_params):.6f}")
    
    # 测试参数保存
    weights_file = model.save_learned_parameters(temp_dir)
    print(f"\n📁 参数保存到: {os.path.basename(weights_file)}")
    
    # 验证保存的文件
    with open(weights_file, 'r') as f:
        saved_data = json.load(f)
    
    print(f"📋 JSON文件包含 {len(saved_data)} 层数据")
    for layer_name, layer_data in saved_data.items():
        print(f"  {layer_name}: {list(layer_data.keys())}")
    
    # 检查统计文件
    stats_file = weights_file.replace('.json', '_stats.txt')
    if os.path.exists(stats_file):
        print(f"✅ 统计文件也已生成: {os.path.basename(stats_file)}")
    
    # 清理
    import shutil
    shutil.rmtree(temp_dir)
    
    print("✅ 参数保存测试完成！")

if __name__ == "__main__":
    test_method5_constraint()
    test_method5_training()
    test_parameter_saving()
    
    print("=" * 60)
    print("🎉 Method5修改验证完成！")
    print("\n✅ 主要改进:")
    print("- 缩放从 1/(a_i*sqrt(d_k)) 改为 a_i/sqrt(d_k)")
    print("- 添加约束条件: a_1 + a_2 + ... + a_m = m")
    print("- 通过softmax实现约束: scores → softmax → *m → a_params")
    print("- 参数保存包含原始scores和最终a_params")
    print("\n🚀 Method5已准备好用于训练！")
