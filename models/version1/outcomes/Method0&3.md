# 修改方法

---

* __令每一层初始输入为$\vec{x}$，Multi-head Attention输出(未进行norm)为$\vec{x'}$，norm后结果/MLP输入为$\vec{y}$，MLP输出(未进行norm)为$\vec{y'}$__

#### Method0：原始模型

#### Method3：MLP残差
* 只在MLP的$\vec{y'}$处进行相加，相加量改为$\sum_{i=1}^{m-1}\vec{y'_i}$

---

# 第一组训练

用较小的模型分别将Method0 Method3训练三次。

### Method0_1

```
  epoch                   =        5.0
  eval_accuracy           =     0.4905
  eval_loss               =     2.6249
  eval_perplexity         =    13.8028
```

### Method0_2

```
  epoch                   =        5.0
  eval_accuracy           =     0.4912
  eval_loss               =     2.6209
  eval_perplexity         =    13.7487
```

### Method0_3

```
  epoch                   =        5.0
  eval_accuracy           =     0.4906
  eval_loss               =     2.6249
  eval_perplexity         =    13.8037
```

### Method3_1

```
  epoch                   =        5.0
  eval_accuracy           =     0.4908
  eval_loss               =     2.6287
  eval_perplexity         =    13.8556
```

### Method3_2

```
  epoch                   =        5.0
  eval_accuracy           =     0.4909
  eval_loss               =     2.6303
  eval_perplexity         =    13.8784
```

### Method3_3

```
  epoch                   =        5.0
  eval_accuracy           =     0.4907
  eval_loss               =     2.6269
  eval_perplexity         =    13.8303
```

### Method1困惑度分析:
* 均值：13.7851
* 方差：0.0006614
* 标准差：0.02572

### Method3困惑度分析:
* 均值：13.8548
* 方差：0.0003859
* 标准差：0.01964

---

# 第二组训练

用100M模型分别将方法0、方法3训练三次

### Method0_1_100M

```
  eval_accuracy           =      0.516
  eval_loss               =     2.4246
  eval_perplexity         =    11.2983
```

### Method0_2_100M

```
  eval_accuracy           =     0.5158
  eval_loss               =     2.4254
  eval_perplexity         =    11.3063
```

### Method0_3_100M

```
  eval_accuracy           =     0.5157
  eval_loss               =      2.424
  eval_perplexity         =    11.2915
```

### Method3_1_100M

```
  eval_accuracy           =     0.5075
  eval_loss               =     2.4983
  eval_perplexity         =    12.1618
```

### Method3_2_100M

```
  eval_accuracy           =     0.5075
  eval_loss               =     2.5015
  eval_perplexity         =    12.2011
```

### Method3_3_100M

```
  eval_accuracy           =     0.5077
  eval_loss               =     2.4989
  eval_perplexity         =    12.1692
```

### Method1困惑度分析
* 均值：11.2987
* 方差：0.00003659
* 标准差：0.00605

### Method3困惑度分析
* 均值：12.1774
* 方差：0.000291
* 标准差：0.01706