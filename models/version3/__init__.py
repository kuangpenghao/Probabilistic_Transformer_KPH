from .configuration_llama_v3 import Method1Config_v3
from .Method1_v3 import Method1LlamaModel_v3, Method1LlamaForCausalLM_v3

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForTokenClassification


AutoConfig.register("method1-v3", Method1Config_v3)
AutoModel.register(Method1Config_v3, Method1LlamaModel_v3)
AutoModelForCausalLM.register(Method1Config_v3, Method1LlamaForCausalLM_v3)
''''''