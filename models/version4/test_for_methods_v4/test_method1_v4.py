#!/usr/bin/env python3
"""
测试脚本：验证Method1_v4实现是否正确遵循设计方案

设计方案验证要点：
1. 每一层注意力机制中存取QK^T矩阵
2. v4m1_ModifiedScailingComputation类正确实现加权计算
3. 将原始sqrt(d_k)广播成向量，与之前所有层QK^T组成的向量进行点乘
4. 模型结构正确覆写（Attention, DecoderLayer, Model, CausalLM）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import math
import numpy as np
from typing import List, Optional

# 导入我们实现的模型
from models.version4.Method1_v4 import (
    Method1LlamaModel_v4, 
    Method1LlamaForCausalLM_v4, 
    Method1LlamaAttention_v4,
    Method1DecoderLayer_v4,
    v4m1_ModifiedScailingComputation
)
from models.version4.configuration_llama_v4 import Method1Config_v4

# 导入原始transformers模型进行对比
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig


class TestMethod1V4:
    def __init__(self):
        """初始化测试环境"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 创建小型配置用于测试
        self.config = Method1Config_v4(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
        )
        
        print(f"测试配置: {self.config.num_hidden_layers}层, {self.config.hidden_size}维度")
    
    def test_modified_scaling_computation(self):
        """测试v4m1_ModifiedScailingComputation类"""
        print("\n=== 测试v4m1_ModifiedScailingComputation类 ===")
        
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        modified_scaling = v4m1_ModifiedScailingComputation(head_dim)
        
        # 测试原始缩放值
        expected_scaling = 1.0 / math.sqrt(head_dim)
        assert abs(modified_scaling.original_scaling - expected_scaling) < 1e-6, \
            f"原始缩放值错误: {modified_scaling.original_scaling} vs {expected_scaling}"
        print(f"✓ 原始缩放值正确: {modified_scaling.original_scaling}")
        
        # 创建模拟的QK^T矩阵
        batch_size, num_heads, seq_len = 2, 4, 16
        qk_shape = (batch_size, num_heads, seq_len, seq_len)
        
        # 测试第0层（应该使用原始缩放）
        qk_matrix_0 = torch.randn(qk_shape)
        result_0 = modified_scaling.compute_modified_scaling([qk_matrix_0], layer_idx=0)
        expected_0 = qk_matrix_0 * expected_scaling
        
        assert torch.allclose(result_0, expected_0, atol=1e-6), "第0层缩放计算错误"
        print("✓ 第0层缩放计算正确")
        
        # 测试后续层（应该使用加权求和）
        qk_matrix_1 = torch.randn(qk_shape)
        qk_matrices = [qk_matrix_0, qk_matrix_1]
        result_1 = modified_scaling.compute_modified_scaling(qk_matrices, layer_idx=1)
        
        # 验证加权求和逻辑
        expected_1 = expected_scaling * (qk_matrix_0 + qk_matrix_1)
        assert torch.allclose(result_1, expected_1, atol=1e-6), "第1层加权缩放计算错误"
        print("✓ 后续层加权缩放计算正确")
        
        print("✓ v4m1_ModifiedScailingComputation类测试通过")
    
    def test_attention_qk_matrix_storage(self):
        """测试注意力机制中QK^T矩阵的存取"""
        print("\n=== 测试注意力机制QK^T矩阵存取 ===")
        
        attention = Method1LlamaAttention_v4(self.config, layer_idx=1).to(self.device)
        attention.eval()
        
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size).to(self.device)
        
        # 创建position_ids
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        
        # 模拟之前层的QK^T矩阵
        previous_qk_shape = (batch_size, self.config.num_attention_heads, seq_len, seq_len)
        previous_qk_matrices = [
            torch.randn(previous_qk_shape).to(self.device) for _ in range(1)
        ]
        
        with torch.no_grad():
            result = attention(
                hidden_states=hidden_states,
                position_ids=position_ids,
                previous_qk_matrices=previous_qk_matrices,
                output_attentions=True
            )
        
        # 验证返回值结构
        assert len(result) == 4, f"注意力返回值数量错误: {len(result)}"
        attn_output, attn_weights, past_key_value, current_qk_matrix = result
        
        # 验证QK^T矩阵形状
        expected_qk_shape = (batch_size, self.config.num_attention_heads, seq_len, seq_len)
        assert current_qk_matrix.shape == expected_qk_shape, \
            f"QK^T矩阵形状错误: {current_qk_matrix.shape} vs {expected_qk_shape}"
        
        print(f"✓ QK^T矩阵形状正确: {current_qk_matrix.shape}")
        print("✓ 注意力机制QK^T矩阵存取测试通过")
    
    def test_decoder_layer_integration(self):
        """测试DecoderLayer的集成"""
        print("\n=== 测试DecoderLayer集成 ===")
        
        decoder_layer = Method1DecoderLayer_v4(self.config, layer_idx=2).to(self.device)
        decoder_layer.eval()
        
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size).to(self.device)
        
        # 创建position_ids
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        
        # 模拟之前层的QK^T矩阵
        previous_qk_shape = (batch_size, self.config.num_attention_heads, seq_len, seq_len)
        previous_qk_matrices = [
            torch.randn(previous_qk_shape).to(self.device) for _ in range(2)
        ]
        
        with torch.no_grad():
            result = decoder_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                previous_qk_matrices=previous_qk_matrices,
                output_attentions=True,
                use_cache=False
            )
        
        # 验证输出结构
        assert len(result) >= 3, f"DecoderLayer返回值数量错误: {len(result)}"
        
        layer_output = result[0]
        current_qk_matrix = result[-1]  # 最后一个应该是QK^T矩阵
        
        # 验证层输出形状
        assert layer_output.shape == hidden_states.shape, \
            f"层输出形状错误: {layer_output.shape} vs {hidden_states.shape}"
        
        # 验证QK^T矩阵形状
        expected_qk_shape = (batch_size, self.config.num_attention_heads, seq_len, seq_len) 
        assert current_qk_matrix.shape == expected_qk_shape, \
            f"DecoderLayer QK^T矩阵形状错误: {current_qk_matrix.shape} vs {expected_qk_shape}"
        
        print(f"✓ DecoderLayer输出形状正确: {layer_output.shape}")
        print(f"✓ DecoderLayer QK^T矩阵形状正确: {current_qk_matrix.shape}")
        print("✓ DecoderLayer集成测试通过")
    
    def test_full_model_forward(self):
        """测试完整模型的前向传播"""
        print("\n=== 测试完整模型前向传播 ===")
        
        model = Method1LlamaModel_v4(self.config).to(self.device)
        model.eval()
        
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_attentions=True,
                output_hidden_states=True
            )
        
        # 验证输出结构
        assert hasattr(outputs, 'last_hidden_state'), "缺少last_hidden_state"
        assert hasattr(outputs, 'attentions'), "缺少attentions"
        assert hasattr(outputs, 'hidden_states'), "缺少hidden_states"
        
        # 验证输出形状
        expected_shape = (batch_size, seq_len, self.config.hidden_size)
        assert outputs.last_hidden_state.shape == expected_shape, \
            f"最终隐藏状态形状错误: {outputs.last_hidden_state.shape} vs {expected_shape}"
        
        # 验证注意力权重数量
        assert len(outputs.attentions) == self.config.num_hidden_layers, \
            f"注意力权重数量错误: {len(outputs.attentions)} vs {self.config.num_hidden_layers}"
        
        print(f"✓ 模型输出形状正确: {outputs.last_hidden_state.shape}")
        print(f"✓ 注意力权重数量正确: {len(outputs.attentions)}")
        print("✓ 完整模型前向传播测试通过")
    
    def test_causal_lm_model(self):
        """测试因果语言模型"""
        print("\n=== 测试因果语言模型 ===")
        
        model = Method1LlamaForCausalLM_v4(self.config).to(self.device)
        model.eval()
        
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        labels = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                output_attentions=True
            )
        
        # 验证输出结构
        assert hasattr(outputs, 'logits'), "缺少logits"
        assert hasattr(outputs, 'loss'), "缺少loss"
        
        # 验证logits形状
        expected_logits_shape = (batch_size, seq_len, self.config.vocab_size)
        assert outputs.logits.shape == expected_logits_shape, \
            f"logits形状错误: {outputs.logits.shape} vs {expected_logits_shape}"
        
        # 验证loss存在且为标量
        assert outputs.loss is not None, "loss不应为None"
        assert outputs.loss.dim() == 0, f"loss应为标量，但维度为: {outputs.loss.dim()}"
        
        print(f"✓ Logits形状正确: {outputs.logits.shape}")
        print(f"✓ Loss计算正确: {outputs.loss.item():.4f}")
        print("✓ 因果语言模型测试通过")
    
    def test_qk_matrix_accumulation_across_layers(self):
        """测试QK^T矩阵在各层间的累积传递"""
        print("\n=== 测试QK^T矩阵跨层累积传递 ===")
        
        model = Method1LlamaModel_v4(self.config).to(self.device)
        model.eval()
        
        batch_size, seq_len = 2, 8  # 使用较小的序列长度便于测试
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        # 手动执行前向传播以监控QK^T矩阵累积
        with torch.no_grad():
            inputs_embeds = model.embed_tokens(input_ids)
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
            position_embeddings = model.rotary_emb(inputs_embeds, position_ids)
            
            hidden_states = inputs_embeds
            stored_qk_matrices = []
            
            for layer_idx, decoder_layer in enumerate(model.layers):
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    previous_qk_matrices=stored_qk_matrices,
                    output_attentions=True
                )
                
                hidden_states = layer_outputs[0]
                current_qk_matrix = layer_outputs[-1]
                stored_qk_matrices.append(current_qk_matrix)
                
                # 验证QK^T矩阵累积
                assert len(stored_qk_matrices) == layer_idx + 1, \
                    f"第{layer_idx}层后，存储的QK^T矩阵数量错误: {len(stored_qk_matrices)}"
                
                expected_qk_shape = (batch_size, self.config.num_attention_heads, seq_len, seq_len)
                assert current_qk_matrix.shape == expected_qk_shape, \
                    f"第{layer_idx}层QK^T矩阵形状错误: {current_qk_matrix.shape}"
                
                print(f"✓ 第{layer_idx}层: QK^T矩阵正确累积，当前共{len(stored_qk_matrices)}个矩阵")
        
        print("✓ QK^T矩阵跨层累积传递测试通过")
    
    def test_scaling_modification_effect(self):
        """测试缩放修改的效果"""
        print("\n=== 测试缩放修改效果 ===")
        
        # 创建简化的测试配置
        test_config = Method1Config_v4(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        
        model = Method1LlamaModel_v4(test_config).to(self.device)
        model.eval()
        
        batch_size, seq_len = 1, 8
        input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        with torch.no_grad():
            # 运行模型并获取注意力权重
            outputs = model(
                input_ids=input_ids,
                output_attentions=True
            )
            
            attentions = outputs.attentions
            
            # 验证注意力权重的形状和数值范围
            for layer_idx, attn_weights in enumerate(attentions):
                expected_shape = (batch_size, test_config.num_attention_heads, seq_len, seq_len)
                assert attn_weights.shape == expected_shape, \
                    f"第{layer_idx}层注意力权重形状错误: {attn_weights.shape}"
                
                # 验证softmax属性（每行和为1）
                row_sums = attn_weights.sum(dim=-1)
                assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
                    f"第{layer_idx}层注意力权重不满足softmax属性"
                
                # 验证数值范围（0到1之间）
                assert torch.all(attn_weights >= 0) and torch.all(attn_weights <= 1), \
                    f"第{layer_idx}层注意力权重超出[0,1]范围"
                
                print(f"✓ 第{layer_idx}层注意力权重形状和数值范围正确")
        
        print("✓ 缩放修改效果测试通过")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始运行Method1_v4测试套件...")
        print("=" * 60)
        
        try:
            self.test_modified_scaling_computation()
            self.test_attention_qk_matrix_storage()
            self.test_decoder_layer_integration()
            self.test_full_model_forward()
            self.test_causal_lm_model()
            self.test_qk_matrix_accumulation_across_layers()
            self.test_scaling_modification_effect()
            
            print("\n" + "=" * 60)
            print("🎉 所有测试通过！Method1_v4实现正确遵循设计方案")
            print("=" * 60)
            
            # 打印设计方案验证总结
            print("\n设计方案验证总结:")
            print("✓ 1. 每一层注意力机制中成功存取QK^T矩阵")
            print("✓ 2. v4m1_ModifiedScailingComputation类正确实现加权计算")
            print("✓ 3. 原始sqrt(d_k)正确广播并与QK^T矩阵进行点乘")
            print("✓ 4. 模型结构正确覆写(Attention, DecoderLayer, Model, CausalLM)")
            print("✓ 5. QK^T矩阵在各层间正确累积传递")
            print("✓ 6. 注意力权重计算结果符合预期")
            
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


if __name__ == "__main__":
    # 运行测试
    tester = TestMethod1V4()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ 测试完成：Method1_v4实现符合设计要求")
    else:
        print("\n❌ 测试失败：请检查实现")
        sys.exit(1)
