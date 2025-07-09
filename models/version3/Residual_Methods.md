# Idea推导

由论文中公式(27)可知：多组H node对Z node更新的消息量为$G=2\sum_n \sum_c {Q_{h,c}^{(t-1)}V_{n.c}U^{(c)T}}$。其中n为H node的组别，$V_c=Q_z^{(t-1)}V^{(c)}$

其中：$V_c$的信息来源于上一轮的$Q_z$而非更新该组时的轮次（类比Transformer中的第i层）。所以相当于Transformers中注意力模块的$V$需要重算。

由$V_c=Q_z^{(t-1)}V^{(c)}$可知：$V^{(c)}$没有发生改变，相当于Transformer层中的$W^V$没有发生改变，仅有输入嵌入$X$发生改变，即PT中的$Q_Z$轮次不再是当时的轮次。所以重算$V$的方法为：每层都重算，输入嵌入X改成当前Transformer层的X。