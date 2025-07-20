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
    简化版自定义Attention类，存储attn_weights和V权重来重新计算attention
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache], torch.Tensor, torch.Tensor]:
        # 调用父类的forward方法获取attention输出和权重
        attention_result = super().forward( 
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,  # 强制获取attn_weights
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        # 解析返回值
        if isinstance(attention_result, tuple):
            attn_output = attention_result[0]
            attn_weights = attention_result[1]  # 这是已经计算好的注意力权重
            past_key_value = attention_result[2] if len(attention_result) > 2 else None
        else:
            attn_output = attention_result
            attn_weights = None
            past_key_value = None
        
        # 返回attention输出、原始权重、past_key_value、注意力权重矩阵和V权重
        return (attn_output, attn_weights if output_attentions else None, past_key_value, 
                attn_weights, self.v_proj.weight)
    
    def forward_with_precomputed_weights(
        self,
        hidden_states: torch.Tensor,
        attn_weights: torch.Tensor,
        v_proj_weight: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        使用预计算的注意力权重和V权重进行attention计算
        只重新计算V矩阵（使用新的输入嵌入和存储的W^V权重）
        
        Args:
            hidden_states: 输入的隐藏状态（当前层的输入）
            attn_weights: 预计算的注意力权重矩阵 (bsz, num_heads, q_len, k_len)
            v_proj_weight: 预计算的V投影权重
            
        Returns:
            attention输出
        """
        bsz, q_len, _ = hidden_states.size()
        
        # 重新计算V矩阵,Reshape V为attention所需格式,重复V以匹配注意力头数
        value_states = F.linear(hidden_states, v_proj_weight, bias=self.v_proj.bias)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # 直接使用存储的注意力权重与新的V相乘
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape并应用输出投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

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
        
        # 处理注意力模块的返回值（现在包含attn_weights和V权重）
        if len(attn_result) >= 5:
            attn_output = attn_result[0]
            self_attn_weights = attn_result[1] if output_attentions else None
            present_key_value = attn_result[2] if use_cache else None
            stored_attn_weights = attn_result[3]  # 用于存储的注意力权重
            v_proj_weight = attn_result[4]  # V投影权重
        else:
            # 回退到原始行为
            attn_output = attn_result[0] if isinstance(attn_result, tuple) else attn_result
            self_attn_weights = attn_result[1] if isinstance(attn_result, tuple) and len(attn_result) > 1 and output_attentions else None
            present_key_value = attn_result[2] if isinstance(attn_result, tuple) and len(attn_result) > 2 and use_cache else None
            stored_attn_weights = None
            v_proj_weight = None
        
        # 注意力部分保持原始的残差连接
        hidden_states = residual + attn_output

        # MLP部分的处理
        mlp_input = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        
        # 存储当前层的权重信息
        current_weights = {
            'attn_weights': stored_attn_weights,  # 注意力权重矩阵
            'v_proj_weight': v_proj_weight,       # V投影权重
            'mlp': self.mlp,                      # MLP模块
            'post_attention_layernorm': self.post_attention_layernorm  # 后注意力层归一化
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
        使用存储的attn_weights和V权重，大幅简化计算复杂度
        
        Args:
            current_input: 当前层的输入嵌入
            stored_weights: 存储的所有前面层的权重信息
            layer_idx: 当前层索引
            position_embeddings: 位置嵌入（未使用，保持接口一致性）
            attention_mask: 注意力掩码（未使用，attn_weights已经包含了掩码信息）
            position_ids: 位置ID（未使用）
            cache_position: 缓存位置（未使用）
            
        Returns:
            重新计算的前面所有层的MLP输出列表
        """
        recomputed_mlp_outputs = []
        
        for i in range(layer_idx):
            weights = stored_weights[i]
            layer = self.layers[i]
            
            # 获取存储的注意力权重和V权重
            attn_weights = weights['attn_weights']
            v_proj_weight = weights['v_proj_weight']
            
            if v_proj_weight is None or attn_weights is None:
                continue
                
            # 对当前输入进行LayerNorm（第i层的input_layernorm）
            normalized_input = layer.input_layernorm(current_input)
            
            # 使用简化的attention方法：预计算的attn_weights + 重新计算的V
            attn_output = layer.self_attn.forward_with_precomputed_weights(
                hidden_states=normalized_input,
                attn_weights=attn_weights,
                v_proj_weight=v_proj_weight,
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
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
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
