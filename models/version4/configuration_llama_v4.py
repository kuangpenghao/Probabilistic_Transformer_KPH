# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" LLaMA model configuration"""
from transformers.models.llama.configuration_llama import LlamaConfig

class Method1Config_v4(LlamaConfig):
    model_type = "method1-v4"

class Method2Config_v4(LlamaConfig):
    model_type = "method2-v4"

class Method3Config_v4(LlamaConfig):
    model_type = "method3-v4"

class Method4Config_v4(LlamaConfig):
    model_type = "method4-v4"

class Method5Config_v4(LlamaConfig):
    model_type = "method5-v4"

class Method6Config_v4(LlamaConfig):
    model_type = "method6-v4"

class Method7Config_v4(LlamaConfig):
    model_type = "method7-v4"

class Method8Config_v4(LlamaConfig):
    model_type = "method8-v4"

class Method1AConfig_v4(LlamaConfig):
    model_type = "method1a-v4"

class Method1BConfig_v4(LlamaConfig):
    model_type = "method1b-v4"

class Method1CConfig_v4(LlamaConfig):
    model_type = "method1c-v4"
