* 方法0：原始Transformer
* 方法1：MLP处，第m层残差连接为$\dfrac{\sum_{i=1}^{m-1}\vec{y'_i}+\vec{y'_m}}{m}+\vec{y_m}$。Attention处残差不变
* 方法2：Attention处：第m层残差为$\dfrac{\sum_{i=1}^{m-1}\vec{x'_i}+\vec{x'_m}}{m}+\vec{x_m}$，MLP处残差不变。
* 方法3：微调方法1：将每一层权重$\dfrac{1}{m}$改为可学习参数并输出最后一层的参数列表。具体来说：除第0层外，每层都拥有一个独立的权重列表，第m层权重列表长度为m+1，涵盖了0-m层所有层的可学习权重。对权重做Softmax归一化后得到每层权重分布。


### 实验方法：
已验证方法0、1、2性能排序为：方法1>方法0>方法2。所以只对比方法2和方法3，并输出方法3的最终权重情况

# 实验数据

### Method1
```
***** eval metrics *****
  epoch                   =        5.0
  eval_accuracy           =     0.5151
  eval_loss               =     2.4321
  eval_perplexity         =    11.3829
  eval_runtime            = 0:00:03.59
  eval_samples            =        143
  eval_samples_per_second =     39.773
  eval_steps_per_second   =      5.006
```

### Method3
```
***** eval metrics *****
  epoch                   =        5.0
  eval_accuracy           =     0.5152
  eval_loss               =     2.4224
  eval_perplexity         =    11.2731
  eval_runtime            = 0:00:03.73
  eval_samples            =        143
  eval_samples_per_second =     38.293
  eval_steps_per_second   =       4.82
```


# Conclusion:

方法3（方法1的微调版本）训练效果略好于方法1。从可学习权重的训练情况看，普遍规律是当前层权重最高，第0层次之，中间层很低。但仍存在例外情况（如Method3_2的第7层最高，第6层次之，其他层很低）