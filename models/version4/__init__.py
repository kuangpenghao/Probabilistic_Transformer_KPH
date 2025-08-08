from .configuration_llama_v4 import (
    Method1Config_v4, Method1AConfig_v4, Method2Config_v4, Method3Config_v4, Method4Config_v4,
    Method5Config_v4, Method6Config_v4, Method7Config_v4, Method8Config_v4
)

from .Method1_v4 import Method1LlamaModel_v4, Method1LlamaForCausalLM_v4
from .Method1A_v4 import Method1ALlamaModel_v4, Method1ALlamaForCausalLM_v4
from .Method2_v4 import Method2LlamaModel_v4, Method2LlamaForCausalLM_v4
from .Method3_v4 import Method3LlamaModel_v4, Method3LlamaForCausalLM_v4
from .Method4_v4 import Method4LlamaModel_v4, Method4LlamaForCausalLM_v4
from .Method5_v4 import Method5LlamaModel_v4, Method5LlamaForCausalLM_v4
from .Method6_v4 import Method6LlamaModel_v4, Method6LlamaForCausalLM_v4
from .Method7_v4 import Method7LlamaModel_v4, Method7LlamaForCausalLM_v4
from .Method8_v4 import Method8LlamaModel_v4, Method8LlamaForCausalLM_v4

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM


# Method1_v4
AutoConfig.register("method1-v4", Method1Config_v4)
AutoModel.register(Method1Config_v4, Method1LlamaModel_v4)
AutoModelForCausalLM.register(Method1Config_v4, Method1LlamaForCausalLM_v4)

# Method1A_v4
AutoConfig.register("method1a-v4", Method1AConfig_v4)
AutoModel.register(Method1AConfig_v4, Method1ALlamaModel_v4)
AutoModelForCausalLM.register(Method1AConfig_v4, Method1ALlamaForCausalLM_v4)

# Method2_v4
AutoConfig.register("method2-v4", Method2Config_v4)
AutoModel.register(Method2Config_v4, Method2LlamaModel_v4)
AutoModelForCausalLM.register(Method2Config_v4, Method2LlamaForCausalLM_v4)

# Method3_v4
AutoConfig.register("method3-v4", Method3Config_v4)
AutoModel.register(Method3Config_v4, Method3LlamaModel_v4)
AutoModelForCausalLM.register(Method3Config_v4, Method3LlamaForCausalLM_v4)

# Method4_v4
AutoConfig.register("method4-v4", Method4Config_v4)
AutoModel.register(Method4Config_v4, Method4LlamaModel_v4)
AutoModelForCausalLM.register(Method4Config_v4, Method4LlamaForCausalLM_v4)

# Method5_v4
AutoConfig.register("method5-v4", Method5Config_v4)
AutoModel.register(Method5Config_v4, Method5LlamaModel_v4)
AutoModelForCausalLM.register(Method5Config_v4, Method5LlamaForCausalLM_v4)

# Method6_v4
AutoConfig.register("method6-v4", Method6Config_v4)
AutoModel.register(Method6Config_v4, Method6LlamaModel_v4)
AutoModelForCausalLM.register(Method6Config_v4, Method6LlamaForCausalLM_v4)

# Method7_v4
AutoConfig.register("method7-v4", Method7Config_v4)
AutoModel.register(Method7Config_v4, Method7LlamaModel_v4)
AutoModelForCausalLM.register(Method7Config_v4, Method7LlamaForCausalLM_v4)

# Method8_v4
AutoConfig.register("method8-v4", Method8Config_v4)
AutoModel.register(Method8Config_v4, Method8LlamaModel_v4)
AutoModelForCausalLM.register(Method8Config_v4, Method8LlamaForCausalLM_v4)