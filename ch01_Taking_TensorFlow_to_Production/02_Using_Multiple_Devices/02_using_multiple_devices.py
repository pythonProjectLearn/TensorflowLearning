# -*- coding: utf-8 -*-
# Using Multiple Devices
#----------------------------------
#
# 使用多个GPU
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# 通过log_device_placement参数来记录运行没一个运算的设备
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

# Runs the op.
print(sess.run(c))


# If we load a graph and want device placement to be forgotten,
#  we set a parameter in our session:
config = tf.ConfigProto()
config.allow_soft_placement = True
sess_soft = tf.Session(config=config)

# GPUs
#---------------------------------
# Note that the GPU must have a compute capability > 3.5 for TF to use.
# http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability


# Careful with GPU memory allocation, TF never releases it.  TF starts with almost
# all of the GPU memory allocated.  We can slowly grow to that limit with an
# option setting:
# 设置1：随着计算的增加，GPU内存也随之增加
config.gpu_options.allow_growth = True
sess_grow = tf.Session(config=config)

# Also, we can limit the size of GPU memory used, with the following option
# 设置2：限制使用GPU的内存大小
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess_limited = tf.Session(config=config)


# How to set placements on multiple devices.
# Here, assume we have three devies CPU:0, GPU:0, and GPU:1
# 当有多个GPU时，通过tf.device()将运算指定到特定设备上
if tf.test.is_built_with_cuda():
    with tf.device('/cpu:0'):
        a = tf.constant([1.0, 3.0, 5.0], shape=[1, 3])
        b = tf.constant([2.0, 4.0, 6.0], shape=[3, 1])
        
        with tf.device('/gpu:1'):
            c = tf.matmul(a,b)
            c = tf.reshape(c, [-1])
        
        with tf.device('/gpu:2'):
            d = tf.matmul(b,a)
            flat_d = tf.reshape(d, [-1])
        
        combined = tf.multiply(c, flat_d)
    print(sess.run(combined))