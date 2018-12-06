# encoding:utf-8
"""
加载训练参数和数据训练模型
"""
import os
import time
import datetime
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import utils
from model import TextCNN
from sklearn.model_selection import train_test_split

# 1 Parameters 加载参数
# ==================================================
# Data loading params


tf.app.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.app.flags.DEFINE_string("positive_data_file", "./data/rt-polarity/rt-polarity.pos", "Data source for the positive data.")
tf.app.flags.DEFINE_string("negative_data_file", "./data/rt-polarity/rt-polarity.neg", "Data source for the positive data.")

# Model Hyperparameters
tf.app.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.app.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.app.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.app.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("num_epochs", 500, "Number of training epochs (default: 500)")
tf.app.flags.DEFINE_integer("evaluate_every", 100, "Evaluate Model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 100, "Save Model after this many steps (default: 100)")
# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")



FLAGS = tf.app.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# 2 Data Preparatopn 加载训练数据
# ==================================================
# Load data
print("Loading data...")
x_text, y = utils.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation

#cross_validation

x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=FLAGS.dev_sample_percentage)

dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
sequence_length=x_train.shape[1]
num_classes=y_train.shape[1]
print("<SEQUENCE_LENGTH>", sequence_length)
print("<NUM_CLASSES>", num_classes)


# 3 Training 训练
# ==================================================
with tf.Graph().as_default():
    # 1 配置图的参数
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)

    # 2 建立图会话session
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 2.1 取得模型结构
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # 2.2 定义训练的方法
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # 2.3 保存训练的中间结果
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.histogram("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # 2.4 构建保存的路径
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # 2.5 保存训练的损失和精度
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # 2.6 将训练的中间结果和损失、精度融合到一起， 保存到train中
        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)


        # 2.7 将评价的损失和准确率，保存在dev中
        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # 2.8 创建checkpoint保存训练好的参数
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "Model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # 2.9 保存词向量
        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # 2.10初始化所有的变量
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # 2.11 训练模型
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        # 2.12 验证评价模型
        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates Model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # 2.13 生成批量迭代的数据
        # Generate batches
        batches = utils.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # 2.14 开始训练模型
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved Model checkpoint to {}\n".format(path))