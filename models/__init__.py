from .configuration_llama import MyLlamaConfig, Method1LlamaConfig,Method2LlamaConfig, Method3LlamaConfig
from .configuration_llama import Method4LlamaConfig,Method5LlamaConfig, Method6LlamaConfig, Method7LlamaConfig
from .modeling_llama import MyLlamaModel, MyLlamaForCausalLM
from .Method1 import Method1LlamaModel, Method1LlamaForCausalLM
from .Method2 import Method2LlamaModel, Method2LlamaForCausalLM
from .Method3 import Method3LlamaModel, Method3LlamaForCausalLM
from .Method4 import Method4LlamaModel, Method4LlamaForCausalLM
from .Method5 import Method5LlamaModel, Method5LlamaForCausalLM
from .Method6 import Method6LlamaModel, Method6LlamaForCausalLM
from .Method7 import Method7LlamaModel, Method7LlamaForCausalLM

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForTokenClassification

'''
AutoConfig.register("my-llama", MyLlamaConfig)
AutoModel.register(MyLlamaConfig, MyLlamaModel)
AutoModelForCausalLM.register(MyLlamaConfig, MyLlamaForCausalLM)
'''

'''
AutoConfig.register("method1-llama", Method1LlamaConfig)
AutoModel.register(Method1LlamaConfig, Method1LlamaModel)
AutoModelForCausalLM.register(Method1LlamaConfig, Method1LlamaForCausalLM)
'''

'''
AutoConfig.register("method2-llama", Method2LlamaConfig)
AutoModel.register(Method2LlamaConfig, Method2LlamaModel)
AutoModelForCausalLM.register(Method2LlamaConfig, Method2LlamaForCausalLM)
'''

'''
AutoConfig.register("method3-llama", Method3LlamaConfig)
AutoModel.register(Method3LlamaConfig, Method3LlamaModel)
AutoModelForCausalLM.register(Method3LlamaConfig, Method3LlamaForCausalLM)
'''

'''
AutoConfig.register("method4-llama", Method4LlamaConfig)
AutoModel.register(Method4LlamaConfig, Method4LlamaModel)
AutoModelForCausalLM.register(Method4LlamaConfig, Method4LlamaForCausalLM)
'''

'''
AutoConfig.register("method5-llama", Method5LlamaConfig)
AutoModel.register(Method5LlamaConfig, Method5LlamaModel)
AutoModelForCausalLM.register(Method5LlamaConfig, Method5LlamaForCausalLM)
'''

'''
AutoConfig.register("method6-llama", Method6LlamaConfig)
AutoModel.register(Method6LlamaConfig, Method6LlamaModel)
AutoModelForCausalLM.register(Method6LlamaConfig, Method6LlamaForCausalLM)
'''

'''
AutoConfig.register("method7-llama", Method7LlamaConfig)
AutoModel.register(Method7LlamaConfig, Method7LlamaModel)
AutoModelForCausalLM.register(Method7LlamaConfig, Method7LlamaForCausalLM)
'''