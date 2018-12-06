# encoding:utf-8
import os
import pprint
import tensorflow as tf
from collections import Counter

from .model import MemN2N

pp = pprint.PrettyPrinter()

# 配置参数
flags = tf.app.flags
# 参数1:变量；参数2:值；参数3:变量描述
flags.DEFINE_integer("edim", 150, "internal state dimension [150]")
flags.DEFINE_integer("lindim", 75, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 6, "number of hops [6]")
flags.DEFINE_integer("mem_size", 100, "memory size [100]")
flags.DEFINE_integer("batch_size", 128, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 100, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")
flags.DEFINE_string("data_dir", "data", "data directory [data]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")
flags.DEFINE_string("data_name", "ptb", "data set name [ptb]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
flags.DEFINE_boolean("show", False, "print progress [False]")

FLAGS = flags.FLAGS


def read_data(fname, count, word2idx):
    """
    :param fname:文本路径
    :param count: [];词频
    :param word2idx:{}; {单词：索引}
    :return:
    """
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise("[!] Data %s not found" % fname)

    words = []
    for line in lines:
        words.extend(line.split())

    # <eos>表示空，有len(lines)行
    if len(count) == 0:
        count.append(['<eos>', 0])
    count[0][1] += len(lines)
    # 单词的频数
    count.extend(Counter(words).most_common())

    if len(word2idx) == 0:
        word2idx['<eos>'] = 0

    for word, _ in count:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    # data存储单词索引
    data = list()
    for line in lines:
        for word in line.split():
            index = word2idx[word]
            data.append(index)
        data.append(word2idx['<eos>'])

    print("Read %s words from %s" % (len(data), fname))
    return data


def main(_):
    count = []
    word2idx = {}

    if not os.path.exists(FLAGS.checkpoint_dir):
      os.makedirs(FLAGS.checkpoint_dir)

    train_data = read_data('%s/%s.train.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
    valid_data = read_data('%s/%s.valid.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
    test_data = read_data('%s/%s.test.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)

    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    FLAGS.nwords = len(word2idx)

    # 打印配置的所有参数
    pp.pprint(flags.FLAGS.__flags)

    with tf.Session() as sess:
        model = MemN2N(FLAGS, sess)
        model.build_model()

        if FLAGS.is_test:
            model.run(valid_data, test_data)
        else:
            model.run(train_data, valid_data)

if __name__ == '__main__':
    tf.app.run()
