# encoding:utf-8

import utils
import tensorflow as tf
from tensorflow.contrib import learn

import numpy as np
import os

# 1 arameters 加载参数
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1481506269/checkpoints", "Checkpoint directory from training run")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
# 输入要判断的文本，根据保存的词向量，建立词向量
inp = input("type in today's news headline: ")
x_raw = [inp]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir,".." ,"vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# 2 Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
with tf.Graph().as_default() as graph:
    sess = tf.Session()
    with sess.as_default():
        # 1 加载训练出来的最佳参数
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # 2 提取模型中的  特征占位符、标签占位符 以及最后的预测值占位符
        # 2.1Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # 2.2 input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # 2.3Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # 3 批度预测，每次取一个样本
        # Generate batches for one epoch
        batches = utils.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # 4 把所有的预测值，放入all_predictions中
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

print(all_predictions)
