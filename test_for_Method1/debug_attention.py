"""
调试新残差连接实现中的问题
"""
import torch
from models.configuration_llama import MyLlamaConfig
from models.Method1 import NewResidualDecoderLayer

def debug_attention_output():
    """调试注意力模块的输出"""
    print("调试注意力模块输出...")
    
    config = MyLlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
    )
    
    # 创建单个解码器层
    layer = NewResidualDecoderLayer(config, layer_idx=0)
    
    # 创建输入
    batch_size = 1
    seq_len = 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    print(f"输入形状: {hidden_states.shape}")
    
    # 测试注意力模块
    with torch.no_grad():
        try:
            # 先测试注意力模块本身
            normalized_hidden_states = layer.input_layernorm(hidden_states)
            print(f"归一化后形状: {normalized_hidden_states.shape}")
            
            # 创建位置嵌入
            position_ids = torch.arange(seq_len).unsqueeze(0)
            cos, sin = layer.self_attn.rotary_emb(normalized_hidden_states, position_ids)
            position_embeddings = (cos, sin)
            
            attn_result = layer.self_attn(
                hidden_states=normalized_hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            
            print(f"注意力模块返回值类型: {type(attn_result)}")
            print(f"注意力模块返回值长度: {len(attn_result) if isinstance(attn_result, (tuple, list)) else 'not a tuple/list'}")
            
            if isinstance(attn_result, (tuple, list)):
                for i, item in enumerate(attn_result):
                    if isinstance(item, torch.Tensor):
                        print(f"  返回值[{i}]形状: {item.shape}")
                    else:
                        print(f"  返回值[{i}]类型: {type(item)}")
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_attention_output()
