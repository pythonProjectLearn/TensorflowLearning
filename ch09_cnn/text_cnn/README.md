# 说明
将整个项目分成：
1、 数据预处理
    1.1 数据清洗
    1.2 将数据处理成模型接收的数据格式
2、模型
    2.1 设置参数： 固定参数（迭代）、可调的参数
    2.2 构建模型结构
    2.3 损失函数
    2.4 优化方法
    2.5 评价方法(例如：准确率)
3、训练
    3.1 将数据分成训练集、测试集、验证集
    3.2 保留训练的中间结果（例如：每次迭代的残差，模型的参数，预测值）
    3.3 保存模型结构，保存训练好的模型参数
4、验证
    4.1 用非训练的数据验证模型的泛化能力
5、预测
    5.1 模型上线

## 0 utils.py辅助函数或者数据清洗

## 1 model.py创建模型结构

## 2 train.py 创建训练和评价过程
一.Graph

Ⅰ.介绍

一个TensorFlow的运算，被表示为一个数据流的图。
一幅图中包含一些操作（Operation）对象，这些对象是计算节点。前面说过的Tensor对象，则是表示在不同的操作（operation）间的数据节点
你一旦开始你的任务，就已经有一个默认的图已经创建好了,一种典型的用法就是要使用到Graph.as_default() 的上下文管理器（ context manager），它能够在这个上下文里面覆盖默认的图
例如:
```
import tensorflow as tf
import numpy as np

c=tf.constant(value=1)
#print(assert c.graph is tf.get_default_graph())
print(c.graph)
print(tf.get_default_graph())


with tf.Graph().as_default() as g:
    d=tf.constant(value=2)
    print(d.graph)
    #print(g)

g2=tf.Graph()
print("g2:",g2)
g2.as_default()
e=tf.constant(value=15)
print(e.graph)
```
上面的例子里面创创建了一个新的图g，然后把g设为默认，那么接下来的操作不是在默认的图中，而是在g中了。你也可以认为现在g这个图就是新的默认的图了。
要注意的是，最后一个量e不是定义在with语句里面的，也就是说，e会包含在最开始的那个图中。也就是说，要在某个graph里面定义量，要在with语句的范围里面定义。

一个Graph的实例支持任意多数量通过名字区分的“collections”。
为了方便，当构建一个大的图的时候，collection能够存储很多类似的对象。比如 tf.Variable就使用了一个collection（tf.GraphKeys.GLOBAL_VARIABLES），包含了所有在图创建过程中的变量。
也可以通过之指定新名称定义其他的collection

## 3 eval.py用重新构造的数据，检验一下模型的

## 4 predict.py 训练好模型之后用来预测

