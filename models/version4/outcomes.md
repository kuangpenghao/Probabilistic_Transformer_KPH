# Version 4

模拟PT中多组Z node的情况修改Transformer的残差连接：

对于注意力机制计算中的$QK^T$矩阵，改成$\sum_i QK^T_i$，即存储之前所有层计算得的$QK^T$矩阵，计算累加结果。所以新的attn_weight矩阵为Softmax($\frac{\sum_i QK^T_i}{d_s}$)。

令之前所有层计算得的$QK^T$矩阵为$\vec{qk}$，维数为当前transformer层数m。令scaling矩阵为$\vec{s}$，维数为当前transformer层数m。所以attn_weight矩阵可统一改写为：Softmax$(\vec{qk}·\vec{s})$。对于$\vec{s}$取值有以下7种方法的尝试：

### Method1
对于第m层，将$\dfrac{1}{\sqrt{d_k}}$广播为向量$\vec{s}$

### Method2
对于第m层，将$\dfrac{1}{\sqrt{d_k*m}}$广播为向量$\vec{s}$

### Method3
对于第m层，将$\dfrac{1}{m\sqrt{d_k}}$广播为向量$\vec{s}$

### Method4
对于第m层，将$\dfrac{1}{d_k^{a_m}*m^{b_m}}$广播为向量$\vec{s}$。其中$a_m,b_m$为可学习参数，正负性不做限定，初始值分别为0.5,1

### Method5
对于第m层，$\vec{s_i}=\dfrac{1}{a_{mi}\sqrt{d_k}}$，其中$a_{mi}$为可学习参数，限定为正值，初始值均为1

### Method6
对于第m层，$\vec{s_i}=\dfrac{1}{a_{mi}d_k^{b_{mi}}}$，其中$a_{mi},b_{mi}$为可学习参数，$a_{mi}$限定为正值，初始值均为1；$b_{mi}$正负性不做限定，初始值为0.5

### Method7
对于第m层，$\vec{s_i}=\dfrac{1}{a_{mi}}$，其中$a_{mi}$为可学习参数，限定为正值，初始值为$\sqrt{d_k}$

# 其余对照方法

### Baseline

### Version2_Method1
CLT复现。MLP处，第m层残差连接为$\dfrac{\sum_{i=1}^{m-1}\vec{y'_i}+\vec{y'_m}}{m}+\vec{y_m}$。Attention处残差不变

### Version2_Method3
CLT的微调方案。将每一层权重$\dfrac{1}{m}$改为可学习参数。具体来说：除第0层外，每层都拥有一个独立的权重列表，第m层权重列表长度为m+1，涵盖了0-m层所有层的可学习权重。对权重做Softmax归一化后得到每层权重分布。

### Version3_Method1
修改MLP处的残差连接为累加先前层MLP输出&该层MlpInput。与CLT不同之处在于先前层MLP输出需要重算

### Version3_Method3.1
与Method1基本相同，唯一不同之处在于MLP处残差和进行了归一化，且每一层的权重分布为1/m

### Version3_Method3.2
与Method1基本相同，唯一不同之处在于MLP处残差和进行了归一化，且每一层的权重分布为可学习分布

---


# 训练结果

### 性能分类（性能从好到坏，按loss从低到高）：
三条参考线为：Baseline，Version2_Method1（CLT性能），Version2_Method3（微调CLT性能）

* __Version2_Method3__

* __Baseline__

* __Version2_Method1__

(全部训练完后进行分类排序)


