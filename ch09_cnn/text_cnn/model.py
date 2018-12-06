# encoding:utf-8
"""
模型结构

tf.Variable与tf.get_variable()

使用tf.Variable时，如果检测到命名冲突，系统会自己处理。使用tf.get_variable()时，系统不会处理冲突而会报错
由于tf.Variable() 每次都在创建新对象，所有reuse=True 和它并没有什么关系。
对于get_variable()，如果已经创建的变量对象，就把那个对象返回，如果没有创建变量对象的话，就创建一个新的。
所以当我们需要共享变量的时候，需要使用tf.get_variable()
"""

import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size,
                       embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # 1 为X, y构建站位符
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 2 定义惩罚项
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # 3 定义词向量层
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # 4 创建卷积和池化
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(input=self.embedded_chars_expanded,  filter=W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # 卷积之后经过非线性转化
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 最大池化输出
                pooled = tf.nn.max_pool(value=h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID',  name="pool")
                pooled_outputs.append(pooled)

        # 5 把多层卷积+池化融合起来
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(values=pooled_outputs, axis=3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # 6 添加drop层防止过拟合
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # 7 全连接层
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            # 取得所有的连接权重，在每个权重上都添加正则化，只在全连接层的一个偏置上添加正则化
            W = tf.get_variable("W",  shape=[num_filters_total, num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # 8 损失函数
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # 9 准确率
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")