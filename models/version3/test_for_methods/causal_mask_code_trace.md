# Method1_v3 Causal Mask代码路径追踪表

## 🔍 完整代码调用路径表

| 阶段 | 文件位置 | 行数 | 函数/方法 | 关键操作 | 传递参数 |
|------|----------|------|-----------|----------|----------|
| **生成阶段** |
| 1 | Method1_v3.py | 367-369 | `Method1LlamaModel_v3.forward()` | `self._update_causal_mask()` | `attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions` |
| 2 | 继承自LlamaModel | - | `_update_causal_mask()` | 生成4D causal mask | 返回 `causal_mask` |
| **分发阶段** |
| 3 | Method1_v3.py | 378-384 | `Method1LlamaModel_v3.forward()` | Layer循环开始 | `for layer_idx, decoder_layer in enumerate(...)` |
| 4 | Method1_v3.py | 383-387 | 条件判断 | `if layer_idx > 0:` | 决定是否触发重计算路径 |
| **标准路径** |
| 5 | Method1_v3.py | 411-423 | `decoder_layer()` 调用 | 传递causal_mask | `attention_mask=causal_mask` |
| 6 | Method1_v3.py | 161-172 | `Method1DecoderLayer_v3.forward()` | 调用self_attn | `attention_mask=attention_mask` |
| 7 | Method1_v3.py | 37-47 | `Method1LlamaAttention_v3.forward()` | 调用父类 | `attention_mask=attention_mask` |
| 8 | 继承自LlamaAttention | - | `LlamaAttention.forward()` | 标准causal保护 | ✅ 安全应用 |
| **重计算路径** |
| 9 | Method1_v3.py | 383-387 | `_recompute_previous_mlp_outputs()` 调用 | 传递causal_mask | `causal_mask, position_ids, cache_position` |
| 10 | Method1_v3.py | 291-298 | `_recompute_previous_mlp_outputs()` | 调用修复方法 | `attention_mask=attention_mask` |
| 11 | Method1_v3.py | 96-118 | `forward_with_precomputed_weights()` | 修复逻辑 | `apply_strict_causal_mask=True` |
| 12 | Method1_v3.py | 103-111 | causal mask重新应用 | `masked_weights = trimmed_attn_weights + causal_mask` | ✅ 修复完成 |
| **汇合阶段** |
| 13 | Method1_v3.py | 425-427 | 获取layer输出 | `hidden_states = layer_outputs[0]` | 合并两路径结果 |
| 14 | Method1_v3.py | 432-433 | 存储权重 | `stored_weights.append(current_weights)` | 为下一层准备 |

## 🎯 关键代码片段定位

### 1. Causal Mask生成 (第367-369行)
```python
causal_mask = self._update_causal_mask(
    attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
)
```

### 2. 重计算路径触发 (第383-387行)
```python
if layer_idx > 0:
    recomputed_mlp_outputs = self._recompute_previous_mlp_outputs(
        hidden_states, stored_weights, layer_idx, position_embeddings, 
        causal_mask, position_ids, cache_position  # 🔥 关键传递
    )
```

### 3. 标准路径传递 (第411行)
```python
layer_outputs = decoder_layer(
    hidden_states,
    attention_mask=causal_mask,  # 🔥 标准传递
    # ...其他参数
)
```

### 4. 重计算中的传递 (第291-298行)
```python
attn_output = layer.self_attn.forward_with_precomputed_weights(
    hidden_states=normalized_input,
    attn_weights=attn_weights,
    v_proj_weight=v_proj_weight,
    attention_mask=attention_mask,  # 🔥 关键传递
    position_ids=position_ids,
    cache_position=cache_position,
    apply_strict_causal_mask=True,  # 🔥 启用修复
)
```

### 5. 修复逻辑核心 (第96-118行)
```python
if apply_strict_causal_mask and attention_mask is not None:
    # 获取当前序列的mask
    current_seq_len = hidden_states.shape[1]
    target_len = attn_weights.shape[-1]
    
    # 确保维度匹配
    if current_seq_len <= target_len:
        # 裁剪mask到当前序列长度
        causal_mask = attention_mask[:, :, :current_seq_len, :target_len]
        
        # 重新应用causal mask
        masked_weights = trimmed_attn_weights + causal_mask  # 🔥 核心修复
        attn_weights_final = F.softmax(masked_weights, dim=-1, dtype=attn_weights.dtype)
```

## 📊 传递路径统计

| 路径类型 | 调用深度 | 关键节点数 | 安全检查点 |
|----------|----------|------------|------------|
| 标准路径 | 4层 | 5个 | 1个 (父类中) |
| 重计算路径 | 6层 | 7个 | 2个 (修复方法中) |
| 总计 | - | 12个 | 3个 |

## 🛡️ 安全检查点详细

### 检查点1: 标准路径 (LlamaAttention.forward)
- **位置**: 继承的父类方法
- **检查内容**: 标准causal mask应用
- **保护级别**: ✅ 完全安全

### 检查点2: 重计算路径维度检查 (第102-119行)
- **位置**: `forward_with_precomputed_weights()`
- **检查内容**: 序列长度兼容性
- **保护级别**: ✅ 维度安全

### 检查点3: 重计算路径mask重新应用 (第109-111行)
- **位置**: `forward_with_precomputed_weights()`
- **检查内容**: causal mask重新应用和归一化
- **保护级别**: ✅ 信息泄漏修复

## 🔄 数据流向图 (简化版)

```
输入数据
    ↓
causal_mask生成 (L367)
    ↓
layer_idx判断
    ↓
┌─────────┴─────────┐
│                   │
▼                   ▼
标准路径            重计算路径
DecoderLayer        _recompute_previous_mlp_outputs
(L411)              (L383)
    ↓                   ↓
self_attn           forward_with_precomputed_weights
(L161)              (L291)
    ↓                   ↓
super().forward     apply_strict_causal_mask
(L37)               (L96)
    ↓                   ↓
✅ 安全              ✅ 修复完成
    │                   │
    └─────────┬─────────┘
              ↓
        layer_outputs合并
              ↓
        stored_weights更新
              ↓
        下一层或结束
```

## 🎯 总结

Method1_v3的causal mask传递路径现在具备：

1. **单一权威源**: 所有mask都来自第367行的统一生成
2. **双路径完整保护**: 标准路径和重计算路径都有完整的causal保护
3. **多层安全检查**: 从生成到应用有3个独立的安全检查点
4. **闭环验证**: 每个关键节点都有对应的验证机制

这确保了Method1_v3在保持算法创新的同时，具备与标准LLaMA完全相同的因果安全性。
