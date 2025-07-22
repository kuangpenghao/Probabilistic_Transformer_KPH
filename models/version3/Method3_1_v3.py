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

# 导入基础类和配置
from .Method1_v3 import (
    Method1LlamaAttention_v3, 
    Method1LlamaModel_v3, 
    Method1DecoderLayer_v3,
    ModifiedResidualMLP
)
from .configuration_llama_v3 import Method3_1Config_v3


class Method3_1ModifiedResidualMLP(ModifiedResidualMLP):
    def compute_residual(self, previous_mlp_outputs: Optional[List[torch.Tensor]], 
                        mlp_input: torch.Tensor, mlp_output: torch.Tensor) -> torch.Tensor:
        """
        计算Method3_1的MLP残差连接：对前面层输出求平均
        
        Args:
            previous_mlp_outputs: 前面层的MLP输出列表
            mlp_input: 当前层MLP的输入（attention后的输出）
            mlp_output: 当前层MLP的输出
            
        Returns:
            最终的层输出
        """
        if self.layer_idx == 0:
            # 第一层：使用标准残差连接
            return mlp_input + mlp_output
        else:
            # 其他层：使用重新计算的MLP输出的平均值作为残差
            if previous_mlp_outputs is not None and len(previous_mlp_outputs) > 0:
                residual_sum = sum(previous_mlp_outputs)
                return (residual_sum + mlp_output)/(len(previous_mlp_outputs) + 1)+mlp_input
            else:
                # 如果没有提供之前的输出，回退到原始行为
                return mlp_input + mlp_output


class Method3_1DecoderLayer_v3(Method1DecoderLayer_v3):
    """
    Method3_1的解码层：继承Method1的所有功能，仅修改MLP残差连接方式
    """
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # 替换为Method3_1的MLP残差连接处理器
        self.modified_residual_mlp = Method3_1ModifiedResidualMLP(layer_idx)


class Method3_1LlamaModel_v3(Method1LlamaModel_v3):
    """
    Method3_1的模型：继承Method1的所有功能，仅使用不同的解码层
    """
    config_class = Method3_1Config_v3

    def __init__(self, config: Method3_1Config_v3):
        # 先调用LlamaModel的初始化（跳过Method1LlamaModel_v3的初始化以避免重复）
        LlamaModel.__init__(self, config)
        # 替换所有的decoder layer为Method3_1的实现
        self.layers = nn.ModuleList(
            [Method3_1DecoderLayer_v3(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # 重新初始化权重
        self.post_init()


class Method3_1LlamaForCausalLM_v3(LlamaForCausalLM):
    """
    Method3_1的因果语言模型
    """
    config_class = Method3_1Config_v3

    def __init__(self, config: Method3_1Config_v3):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = Method3_1LlamaModel_v3(config)
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
