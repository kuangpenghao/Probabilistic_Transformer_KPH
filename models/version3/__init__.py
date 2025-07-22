from .configuration_llama_v3 import Method1Config_v3,Method2Config_v3,Method3_1Config_v3
from .Method1_v3 import Method1LlamaModel_v3, Method1LlamaForCausalLM_v3
from .Method2_v3 import Method2LlamaModel_v3, Method2LlamaForCausalLM_v3
from .Method3_1_v3 import Method3_1LlamaModel_v3, Method3_1LlamaForCausalLM_v3

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForTokenClassification


AutoConfig.register("method1-v3", Method1Config_v3)
AutoModel.register(Method1Config_v3, Method1LlamaModel_v3)
AutoModelForCausalLM.register(Method1Config_v3, Method1LlamaForCausalLM_v3)

AutoConfig.register("method2-v3", Method2Config_v3)
AutoModel.register(Method2Config_v3, Method2LlamaModel_v3)
AutoModelForCausalLM.register(Method2Config_v3, Method2LlamaForCausalLM_v3)

AutoConfig.register("method3-1-v3", Method3_1Config_v3)
AutoModel.register(Method3_1Config_v3, Method3_1LlamaModel_v3)
AutoModelForCausalLM.register(Method3_1Config_v3, Method3_1LlamaForCausalLM_v3)