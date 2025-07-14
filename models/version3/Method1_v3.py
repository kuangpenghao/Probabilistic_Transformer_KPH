import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaDecoderLayer, LlamaAttention
from .configuration_llama_v3 import LlamaConfig, Method1Config_v3


class Method1LlamaAttention_v3(LlamaAttention):
    """
    自定义Attention类，支持使用预计算的Q、K、W^V进行attention计算
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache], torch.Tensor, torch.Tensor, torch.Tensor]:
        # 调用父类的forward方法
        attention_output = super().forward( 
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        # 计算Q、K矩阵用于存储（应用位置编码后的版本）
        bsz, q_len, _ = hidden_states.size()
        
        # 计算原始Q、K投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        
        # Reshape为多头格式
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # 应用位置编码
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # 确保返回值格式正确
        if isinstance(attention_output, tuple):
            attn_output, attn_weights, past_key_value = attention_output[0], attention_output[1] if len(attention_output) > 1 else None, attention_output[2] if len(attention_output) > 2 else None
        else:
            attn_output, attn_weights, past_key_value = attention_output, None, None
        
        # 返回attention输出以及应用位置编码后的Q、K矩阵和W^V权重
        return (attn_output, attn_weights, past_key_value, query_states, key_states, self.v_proj.weight)
    
    def forward_with_precomputed_qkv(
        self,
        hidden_states: torch.Tensor,
        precomputed_query_states: torch.Tensor,
        precomputed_key_states: torch.Tensor,
        v_proj_weight: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        使用预计算的Q、K矩阵和W^V权重进行attention计算
        只重新计算V矩阵（使用新的输入嵌入和存储的W^V权重）
        
        Args:
            hidden_states: 输入的隐藏状态（当前层的输入）
            precomputed_query_states: 预计算的Q矩阵（已应用位置编码和reshape）
            precomputed_key_states: 预计算的K矩阵（已应用位置编码和reshape）
            v_proj_weight: 预计算的V投影权重
            attention_mask: 注意力掩码
            position_embeddings: 位置嵌入
            
        Returns:
            attention输出
        """
        bsz, q_len, _ = hidden_states.size()
        
        # 使用预计算的Q、K矩阵（已经应用了位置编码和reshape）
        query_states = precomputed_query_states
        key_states = precomputed_key_states
        
        # 只重新计算V矩阵（使用新输入 + 存储的W^V权重）
        value_states = F.linear(hidden_states, v_proj_weight, bias=self.v_proj.bias)
        
        # Reshape V为attention所需格式
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # 计算完整的attention
        attn_output = self._compute_scaled_dot_product_attention(
            query_states, key_states, value_states, attention_mask, bsz, q_len
        )
        
        # Reshape并应用输出投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output
    
    def _apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """应用旋转位置编码"""
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _rotate_half(self, x: torch.Tensor):
        """旋转张量的一半隐藏维度"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """重复key-value张量以匹配查询头数"""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def _compute_scaled_dot_product_attention(self, query_states, key_states, value_states, attention_mask, bsz, q_len):
        """计算缩放点积注意力"""
        # 重复K、V以匹配查询头数
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # 计算attention权重 - 使用head_dim进行缩放
        scaling = self.head_dim ** -0.5
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
        
        # 应用causal mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        # Softmax和dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(
            attn_weights, 
            p=0.0 if not self.training else self.attention_dropout, 
            training=self.training
        )
        
        # 计算最终的attention输出
        attn_output = torch.matmul(attn_weights, value_states)
        
        return attn_output

class Method1DecoderLayer_v3(LlamaDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        # 使用自定义的Attention类
        self.self_attn = Method1LlamaAttention_v3(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # 新增参数：存储之前层的MLP输出和权重矩阵
        previous_mlp_outputs: Optional[List[torch.Tensor]] = None,
        stored_weights: Optional[dict] = None,  # 存储Q、K、W^V权重
        current_layer_input: Optional[torch.Tensor] = None,  # 当前层的原始输入
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]], Optional[torch.Tensor], dict]:
        
        # 保存当前层输入，用于注意力的残差连接
        residual = hidden_states
        
        # 输入层归一化
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention - 使用自定义的attention类
        attn_result = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        # 处理注意力模块的返回值（现在包含Q、K、W^V）
        if len(attn_result) >= 6:
            attn_output = attn_result[0]
            self_attn_weights = attn_result[1] if output_attentions else None
            present_key_value = attn_result[2] if use_cache else None
            query_states = attn_result[3]
            key_states = attn_result[4] 
            v_proj_weight = attn_result[5]
        else:
            # 回退到原始行为
            attn_output = attn_result[0] if isinstance(attn_result, tuple) else attn_result
            self_attn_weights = attn_result[1] if isinstance(attn_result, tuple) and len(attn_result) > 1 and output_attentions else None
            present_key_value = attn_result[2] if isinstance(attn_result, tuple) and len(attn_result) > 2 and use_cache else None
            query_states = None
            key_states = None
            v_proj_weight = None
        
        # 注意力部分保持原始的残差连接
        hidden_states = residual + attn_output

        # MLP部分的处理
        mlp_input = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        
        # 存储当前层的权重信息
        current_weights = {
            'query_states': query_states,
            'key_states': key_states,
            'v_proj_weight': v_proj_weight,
            'mlp': self.mlp,
            'post_attention_layernorm': self.post_attention_layernorm
        }
        
        if self.layer_idx == 0:
            # 第一层：直接使用MLP输出，不进行残差连接
            hidden_states = mlp_input + mlp_output
        else:
            # 其他层：使用重新计算的MLP输出作为残差
            if previous_mlp_outputs is not None and len(previous_mlp_outputs) > 0:
                residual_sum = sum(previous_mlp_outputs)
                hidden_states = residual_sum + mlp_output
            else:
                # 如果没有提供之前的输出，回退到原始行为
                hidden_states = mlp_input + mlp_output

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,) 
        
        # 添加当前层的权重信息到返回值中
        outputs += (current_weights,)

        return outputs


class Method1LlamaModel_v3(LlamaModel):
    config_class = Method1Config_v3

    def __init__(self, config: Method1Config_v3):
        super().__init__(config)
        # 替换所有的decoder layer为新的实现
        self.layers = nn.ModuleList(
            [Method1DecoderLayer_v3(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # 重新初始化权重
        self.post_init()

    def _recompute_previous_mlp_outputs(self, current_input: torch.Tensor, stored_weights: List[dict], 
                                      layer_idx: int, position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
                                      attention_mask: Optional[torch.Tensor] = None,
                                      position_ids: Optional[torch.LongTensor] = None,
                                      cache_position: Optional[torch.LongTensor] = None) -> List[torch.Tensor]:
        """
        重新计算前面所有层的MLP输出，使用当前层的输入嵌入
        充分利用stored_weights中已计算的Q、K矩阵和W^V权重
        
        Args:
            current_input: 当前层的输入嵌入
            stored_weights: 存储的所有前面层的权重信息
            layer_idx: 当前层索引
            position_embeddings: 位置嵌入
            attention_mask: 注意力掩码
            position_ids: 位置ID
            cache_position: 缓存位置
            
        Returns:
            重新计算的前面所有层的MLP输出列表
        """
        recomputed_mlp_outputs = []
        
        for i in range(layer_idx):
            weights = stored_weights[i]
            layer = self.layers[i]
            
            # 获取存储的Q、K矩阵和V权重
            query_states = weights['query_states']
            key_states = weights['key_states']
            v_proj_weight = weights['v_proj_weight']
            
            if v_proj_weight is None or query_states is None or key_states is None:
                continue
                
            # 对当前输入进行LayerNorm（第i层的input_layernorm）
            normalized_input = layer.input_layernorm(current_input)
            
            # 使用预计算的Q、K和V权重的attention方法
            attn_output = layer.self_attn.forward_with_precomputed_qkv(
                hidden_states=normalized_input,
                precomputed_query_states=query_states,
                precomputed_key_states=key_states,
                v_proj_weight=v_proj_weight,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
            
            # 进行attention部分的残差连接
            layer_output = current_input + attn_output
            
            # 然后进行MLP处理（使用存储的模块）
            mlp_module = weights['mlp']
            post_attn_layernorm = weights['post_attention_layernorm']
            
            normalized = post_attn_layernorm(layer_output)
            mlp_output = mlp_module(normalized)
            
            recomputed_mlp_outputs.append(mlp_output)
            
        return recomputed_mlp_outputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            warnings.warn(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            from transformers.cache_utils import DynamicCache
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 存储所有层的权重信息和层输入
        stored_weights = []
        layer_inputs = []
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 存储当前层的输入
            layer_inputs.append(hidden_states.clone())
            
            # 重新计算前面层的MLP输出（使用当前层输入）
            if layer_idx > 0:
                recomputed_mlp_outputs = self._recompute_previous_mlp_outputs(
                    hidden_states, stored_weights, layer_idx, position_embeddings, 
                    causal_mask, position_ids, cache_position
                )
            else:
                recomputed_mlp_outputs = []

            if self.gradient_checkpointing and self.training:
                from functools import partial
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    recomputed_mlp_outputs,
                    None,  # stored_weights参数
                    hidden_states,  # current_layer_input参数
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    previous_mlp_outputs=recomputed_mlp_outputs,
                    stored_weights=None,
                    current_layer_input=hidden_states,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            # 存储当前层的权重信息
            current_weights = layer_outputs[-1]  # 最后一个是权重信息
            stored_weights.append(current_weights)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Method1LlamaForCausalLM_v3(LlamaForCausalLM):
    config_class = Method1Config_v3

    def __init__(self, config: Method1Config_v3):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = Method1LlamaModel_v3(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
