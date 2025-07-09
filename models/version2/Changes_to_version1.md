* __令每一层初始输入为$\vec{x}$，Multi-head Attention输出(未进行norm)为$\vec{x'}$，norm后结果/MLP输入为$\vec{y}$，MLP输出(未进行norm)为$\vec{y'}$__

Version1的MLP处残差连接方式为：第m层残差连接为$\dfrac{\sum_{i=1}^{m-1}\vec{y'_i}}{m-1}+\vec{y'_m}$，Attn处残差连接不变（相当于当时7中方法中的Method4）

Version2的残差连接方式为：
<br>

* 方法1：MLP处，第m层残差连接为$\dfrac{\sum_{i=1}^{m-1}\vec{y'_i}+\vec{y'_m}}{m}+\vec{y_m}$。Attention处残差不变
<br>
* 方法2：Attention处：第m层残差为$\dfrac{\sum_{i=1}^{m-1}\vec{x'_i}+\vec{x'_m}}{m}+\vec{x_m}$，MLP处残差不变。