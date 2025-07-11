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
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaDecoderLayer
from .configuration_llama_v2 import LlamaConfig, Method3Config_v2

class Method3DecoderLayer_v2(LlamaDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        
        # 为MLP残差连接添加可学习的权重参数，每一层都需要为从第1层到当前层的所有MLP输出学习权重
        if layer_idx > 0:
            # layer_idx+1 是因为包含当前层，从0索引转换为层数
            self.mlp_residual_weights = nn.Parameter(
                torch.ones(layer_idx + 1, dtype=torch.float32)
            )
        else:
            # 第0层不需要权重参数
            self.mlp_residual_weights = None

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
        # 新增参数：存储之前层的MLP输出
        previous_mlp_outputs: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]], Optional[torch.Tensor]]:
        
        # 保存当前层输入，用于注意力的残差连接
        residual = hidden_states
        
        # 输入层归一化
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention - 保持原始的残差连接方式
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
        
        # 处理注意力模块的返回值
        if isinstance(attn_result, tuple) and len(attn_result) >= 2:
            attn_output = attn_result[0]
            self_attn_weights = attn_result[1] if output_attentions else None
        else:
            attn_output = attn_result
            self_attn_weights = None
        
        # 注意力部分保持原始的残差连接
        hidden_states = residual + attn_output

        # MLP处的残差连接修改
        mlp_input = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        
        # 计算前面所有层MLP输出的累加和作为残差
        if previous_mlp_outputs is not None and len(previous_mlp_outputs) > 0 and self.mlp_residual_weights is not None:
            # 使用可学习的权重参数，创建包含所有相关MLP输出的张量列表（包括当前层的输出）
            all_mlp_outputs = previous_mlp_outputs + [mlp_output]
            
            # 对权重进行softmax归一化，确保总和为1
            normalized_weights = F.softmax(self.mlp_residual_weights, dim=0)
            
            # 计算加权和
            weighted_sum = torch.zeros_like(mlp_output)
            for i, output in enumerate(all_mlp_outputs):
                weighted_sum += normalized_weights[i] * output
            
            hidden_states = weighted_sum + mlp_input
        else:
            # 如果没有提供之前的输出或者是第0层，回退到原始行为
            hidden_states = mlp_input + mlp_output

        # 保存当前层的最终MLP输出，用于后续层
        current_mlp_output = hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        
        # 添加当前层的MLP输出到返回值中
        outputs += (mlp_output,)

        return outputs


class Method3LlamaModel_v2(LlamaModel):
    config_class = Method3Config_v2

    def __init__(self, config: Method3Config_v2):
        super().__init__(config)
        # 替换所有的decoder layer为新的实现
        self.layers = nn.ModuleList(
            [Method3DecoderLayer_v2(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # 重新初始化权重
        self.post_init()

    def save_learned_parameters(self, save_dir: str = None):
        """
        保存每一层的可学习权重参数到Parameters_learned.txt文件
        
        Args:
            save_dir: 保存目录，如果为None则保存到当前模型文件所在目录
        """
        import os
        import torch
        import torch.nn.functional as F
        from datetime import datetime
        
        if save_dir is None:
            # 默认保存到当前文件所在目录
            save_dir = os.path.dirname(os.path.abspath(__file__))
        
        save_path = os.path.join(save_dir, "Parameters_learned.txt")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Method3_v2 可学习权重参数保存文件\n")
            f.write(f"保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("模型配置信息:\n")
            f.write(f"- 总层数: {self.config.num_hidden_layers}\n")
            f.write(f"- 隐藏层大小: {self.config.hidden_size}\n")
            f.write(f"- 注意力头数: {self.config.num_attention_heads}\n")
            f.write(f"- 中间层大小: {self.config.intermediate_size}\n")
            f.write("-" * 80 + "\n\n")
            
            for layer_idx, layer in enumerate(self.layers):
                f.write(f"第 {layer_idx} 层 (索引: {layer_idx}):\n")
                
                if hasattr(layer, 'mlp_residual_weights') and layer.mlp_residual_weights is not None:
                    # 获取原始权重参数
                    raw_weights = layer.mlp_residual_weights.detach().cpu()
                    # 计算归一化权重
                    normalized_weights = F.softmax(raw_weights, dim=0)
                    
                    f.write(f"  权重参数数量: {len(raw_weights)}\n")
                    f.write(f"  原始权重参数: {raw_weights.tolist()}\n")
                    f.write(f"  归一化权重: {normalized_weights.tolist()}\n")
                                            
                else:
                    f.write(f"  无权重参数 (第0层)\n")
                
                f.write("\n" + "-" * 60 + "\n\n")
        
        print(f"可学习权重参数已保存到: {save_path}")
        return save_path

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

        # 存储所有层的MLP输出
        all_mlp_outputs = []
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 传递之前层的MLP输出
            previous_mlp_outputs = all_mlp_outputs.copy() if layer_idx > 0 else None

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
                    previous_mlp_outputs,
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
                    previous_mlp_outputs=previous_mlp_outputs,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            # 保存当前层的MLP输出
            current_mlp_output = layer_outputs[-1]  # 最后一个是MLP输出
            all_mlp_outputs.append(current_mlp_output)

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


class Method3LlamaForCausalLM_v2(LlamaForCausalLM):
    config_class = Method3Config_v2

    def __init__(self, config: Method3Config_v2):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = Method3LlamaModel_v2(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def save_learned_parameters(self, save_dir: str = None):
        """
        保存每一层的可学习权重参数到Parameters_learned.txt文件
        这是一个便捷方法，直接调用内部模型的保存方法
        
        Args:
            save_dir: 保存目录，如果为None则保存到当前模型文件所在目录
            
        Returns:
            str: 保存文件的路径
        """
        return self.model.save_learned_parameters(save_dir)

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
