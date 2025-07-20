"""
演示Method1中信息泄漏问题的具体例子
"""
import torch
import torch.nn as nn

def demonstrate_information_leakage():
    """演示信息泄漏的具体例子"""
    
    print("=== 信息泄漏问题演示 ===\n")
    
    # 模拟序列: ["The", "cat", "sits"]
    sequence_length = 3
    hidden_size = 4
    batch_size = 1
    
    # 模拟每一层的输入嵌入（逐层累积信息）
    layer1_input = torch.randn(batch_size, sequence_length, hidden_size)  
    layer2_input = layer1_input + 0.1 * torch.randn(batch_size, sequence_length, hidden_size)  # 累积了layer1的信息
    layer3_input = layer2_input + 0.1 * torch.randn(batch_size, sequence_length, hidden_size)  # 累积了layer1+layer2的信息
    
    print("1. 正常情况（无泄漏）:")
    print(f"   Layer1处理token[2]时，使用输入: {layer1_input[0, 2, :2]}")
    print(f"   Layer2处理token[2]时，使用输入: {layer2_input[0, 2, :2]}")
    print(f"   Layer3处理token[2]时，使用输入: {layer3_input[0, 2, :2]}")
    
    # 模拟存储的attention权重（来自layer1的原始计算）
    stored_attn_weights_layer1 = torch.softmax(torch.randn(batch_size, 2, sequence_length, sequence_length), dim=-1)
    
    print(f"\n2. Method1的问题情况:")
    print(f"   存储的Layer1 attention权重来自: layer1_input")
    print(f"   但重新计算Layer1的V矩阵时使用: layer3_input ← 这就是信息泄漏!")
    
    # 演示信息泄漏的具体影响
    def compute_normal_attention(input_emb, attn_weights):
        """正常的attention计算"""
        # 模拟V矩阵计算
        v_matrix = input_emb @ torch.randn(hidden_size, hidden_size)
        # 简化的attention输出
        return attn_weights @ v_matrix.unsqueeze(1)
    
    def compute_method1_attention(layer1_input, layer3_input, stored_attn_weights):
        """Method1的有问题计算"""
        # 使用layer1的attention权重，但用layer3的输入计算V
        v_matrix = layer3_input @ torch.randn(hidden_size, hidden_size)  # ← 信息泄漏！
        return stored_attn_weights @ v_matrix.unsqueeze(1)
    
    # 正常计算
    normal_output = compute_normal_attention(layer1_input, stored_attn_weights_layer1)
    
    # 有问题的Method1计算
    leaked_output = compute_method1_attention(layer1_input, layer3_input, stored_attn_weights_layer1)
    
    print(f"\n3. 计算结果对比:")
    print(f"   正常计算的输出: {normal_output[0, 0, 2, :2]}")
    print(f"   信息泄漏的输出: {leaked_output[0, 0, 2, :2]}")
    print(f"   差异: {(leaked_output - normal_output)[0, 0, 2, :2]}")
    
    print(f"\n4. 为什么这会导致训练异常:")
    print(f"   - 模型在训练时能够'预见未来'，快速记住训练数据")
    print(f"   - Loss急剧下降，因为模型获得了额外信息")
    print(f"   - 但在推理时这种泄漏不存在，导致性能下降")
    
    print(f"\n5. 类比:")
    print(f"   就像考试时，你在回答第1题时偷看了第3题的答案")
    print(f"   虽然能帮你答对第1题，但这不是真正的理解")

def show_causal_violation():
    """展示因果性违反的问题"""
    print(f"\n=== 因果性违反演示 ===")
    
    # 在position i处，正常情况下只能看到位置 <= i 的信息
    print("正常的因果依赖:")
    print("Position 0: 只依赖 token[0]")
    print("Position 1: 只依赖 token[0], token[1]") 
    print("Position 2: 只依赖 token[0], token[1], token[2]")
    
    print("\nMethod1造成的异常依赖:")
    print("Position 0 在layer3重算时: 依赖了经过layer2处理的信息")
    print("                         → 间接依赖了所有position的信息!")
    print("这违反了Transformer的基本假设!")

if __name__ == "__main__":
    demonstrate_information_leakage()
    show_causal_violation()
