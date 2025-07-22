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
from .Method2_v3 import (
    Method2LlamaAttention_v3, 
    Method2LlamaModel_v3, 
    Method2DecoderLayer_v3,
    ModifiedResidualAttention
)
from .configuration_llama_v3 import Method4_2Config_v3


class Method4_2ModifiedResidualAttention(ModifiedResidualAttention):
    """
    Method4_2版本的Attention残差连接：对前面层的Attention输出进行可学习加权求和
    """
    def __init__(self, layer_idx: int, num_hidden_layers: int):
        super().__init__(layer_idx)
        self.num_hidden_layers = num_hidden_layers
        
        if layer_idx > 0:
            # 为前面的每一层创建可学习的权重参数
            self.layer_weights = nn.Parameter(torch.ones(layer_idx))
    
    def compute_residual(self, previous_attn_outputs: Optional[List[torch.Tensor]], 
                        residual: torch.Tensor, attn_output: torch.Tensor) -> torch.Tensor:
        """
        计算Method4_2的Attention残差连接：对前面层输出进行可学习加权求和
        
        Args:
            previous_attn_outputs: 前面层的Attention输出列表
            residual: 当前层的原始输入（attention前的输入）
            attn_output: 当前层Attention的输出
            
        Returns:
            最终的attention后输出
        """
        if self.layer_idx == 0:
            # 第一层：使用标准残差连接
            return residual + attn_output
        else:
            # 其他层：使用重新计算的Attention输出的可学习加权求和作为残差
            if previous_attn_outputs is not None and len(previous_attn_outputs) > 0:
                # 对权重进行softmax归一化
                layer_weights_normalized = F.softmax(self.layer_weights[:len(previous_attn_outputs)], dim=0)
                
                # 加权求和
                residual_sum = sum(weight * output for weight, output in zip(layer_weights_normalized, previous_attn_outputs))
                return residual_sum + attn_output
            else:
                # 如果没有提供之前的输出，回退到原始行为
                return residual + attn_output
    
    def get_layer_weights(self) -> torch.Tensor:
        """
        获取当前层的权重分布（经过softmax归一化）
        """
        if self.layer_idx == 0:
            return torch.tensor([])
        return F.softmax(self.layer_weights, dim=0).detach()


class Method4_2DecoderLayer_v3(Method2DecoderLayer_v3):
    """
    Method4_2的解码层：继承Method2的所有功能，仅修改Attention残差连接方式为可学习权重
    """
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # 替换为Method4_2的Attention残差连接处理器
        self.modified_residual_attn = Method4_2ModifiedResidualAttention(layer_idx, config.num_hidden_layers)
    
    def get_layer_weights(self) -> torch.Tensor:
        """
        获取当前层的Attention权重分布
        """
        return self.modified_residual_attn.get_layer_weights()


class Method4_2LlamaModel_v3(Method2LlamaModel_v3):
    """
    Method4_2的模型：继承Method2的所有功能，仅使用不同的解码层
    """
    config_class = Method4_2Config_v3

    def __init__(self, config: Method4_2Config_v3):
        # 先调用LlamaModel的初始化（跳过Method2LlamaModel_v3的初始化以避免重复）
        LlamaModel.__init__(self, config)
        # 替换所有的decoder layer为Method4_2的实现
        self.layers = nn.ModuleList(
            [Method4_2DecoderLayer_v3(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # 重新初始化权重
        self.post_init()
    
    def get_all_layer_weights(self) -> List[torch.Tensor]:
        """
        获取所有层的Attention权重分布
        """
        return [layer.get_layer_weights() for layer in self.layers]


class Method4_2LlamaForCausalLM_v3(LlamaForCausalLM):
    """
    Method4_2的因果语言模型
    """
    config_class = Method4_2Config_v3

    def __init__(self, config: Method4_2Config_v3):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = Method4_2LlamaModel_v3(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_all_layer_weights(self) -> List[torch.Tensor]:
        """
        获取所有层的Attention权重分布
        """
        return self.model.get_all_layer_weights()

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
