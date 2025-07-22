from .configuration_llama_v3 import Method1Config_v3,Method2Config_v3,Method3_1Config_v3
from .configuration_llama_v3 import Method3_2Config_v3,Method4_1Config_v3,Method4_2Config_v3

from .Method1_v3 import Method1LlamaModel_v3, Method1LlamaForCausalLM_v3
from .Method2_v3 import Method2LlamaModel_v3, Method2LlamaForCausalLM_v3
from .Method3_1_v3 import Method3_1LlamaModel_v3, Method3_1LlamaForCausalLM_v3
from .Method3_2_v3 import Method3_2LlamaModel_v3, Method3_2LlamaForCausalLM_v3
from .Method4_1_v3 import Method4_1LlamaModel_v3, Method4_1LlamaForCausalLM_v3
from .Method4_2_v3 import Method4_2LlamaModel_v3, Method4_2LlamaForCausalLM_v3

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

AutoConfig.register("method3-2-v3", Method3_2Config_v3)
AutoModel.register(Method3_2Config_v3, Method3_2LlamaModel_v3)
AutoModelForCausalLM.register(Method3_2Config_v3, Method3_2LlamaForCausalLM_v3)

AutoConfig.register("method4-1-v3", Method4_1Config_v3)
AutoModel.register(Method4_1Config_v3, Method4_1LlamaModel_v3)
AutoModelForCausalLM.register(Method4_1Config_v3, Method4_1LlamaForCausalLM_v3)

AutoConfig.register("method4-2-v3", Method4_2Config_v3)
AutoModel.register(Method4_2Config_v3, Method4_2LlamaModel_v3)
AutoModelForCausalLM.register(Method4_2Config_v3, Method4_2LlamaForCausalLM_v3)