from .configuration_llama_v2 import Method1Config_v2, Method2Config_v2
from .Method1_v2 import Method1LlamaModel_v2, Method1LlamaForCausalLM_v2
from .Method2_v2 import Method2LlamaModel_v2, Method2LlamaForCausalLM_v2

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForTokenClassification


AutoConfig.register("method1-v2", Method1Config_v2)
AutoModel.register(Method1Config_v2, Method1LlamaModel_v2)
AutoModelForCausalLM.register(Method1Config_v2, Method1LlamaForCausalLM_v2)
''''''

AutoConfig.register("method2-v2", Method2Config_v2)
AutoModel.register(Method2Config_v2, Method2LlamaModel_v2)
AutoModelForCausalLM.register(Method2Config_v2, Method2LlamaForCausalLM_v2)
''''''