# encoding:utf-8
"""
主要看一下别人是如何调参的
"""
import numpy as np

np.random.seed(456)
import tensorflow as tf

tf.set_random_seed(456)
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score


def eval_tox21_hyperparams(n_hidden=50, n_layers=1, learning_rate=.001,
                           dropout_prob=0.5, n_epochs=45, batch_size=100,
                           weight_positives=True):
    print("---------------------------------------------")
    print("Model hyperparameters")
    print("n_hidden = %d" % n_hidden)
    print("n_layers = %d" % n_layers)
    print("learning_rate = %f" % learning_rate)
    print("n_epochs = %d" % n_epochs)
    print("batch_size = %d" % batch_size)
    print("weight_positives = %s" % str(weight_positives))
    print("dropout_prob = %f" % dropout_prob)
    print("---------------------------------------------")

    d = 1024
    graph = tf.Graph()
    with graph.as_default():
        _, (train, valid, test), _ = dc.molnet.load_tox21()
        train_X, train_y, train_w = train.X, train.y, train.w
        valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
        test_X, test_y, test_w = test.X, test.y, test.w

        # Remove extra tasks
        train_y = train_y[:, 0]
        valid_y = valid_y[:, 0]
        test_y = test_y[:, 0]
        train_w = train_w[:, 0]
        valid_w = valid_w[:, 0]
        test_w = test_w[:, 0]

        # Generate tensorflow graph
        with tf.name_scope("placeholders"):
            x = tf.placeholder(tf.float32, (None, d))
            y = tf.placeholder(tf.float32, (None,))
            w = tf.placeholder(tf.float32, (None,))
            keep_prob = tf.placeholder(tf.float32)
        for layer in range(n_layers):
            with tf.name_scope("layer-%d" % layer):
                W = tf.Variable(tf.random_normal((d, n_hidden)))
                b = tf.Variable(tf.random_normal((n_hidden,)))
                x_hidden = tf.nn.relu(tf.matmul(x, W) + b)
                # Apply dropout
                x_hidden = tf.nn.dropout(x_hidden, keep_prob)
        with tf.name_scope("output"):
            W = tf.Variable(tf.random_normal((n_hidden, 1)))
            b = tf.Variable(tf.random_normal((1,)))
            y_logit = tf.matmul(x_hidden, W) + b
            # the sigmoid gives the class probability of 1
            y_one_prob = tf.sigmoid(y_logit)
            # Rounding P(y=1) will give the correct prediction.
            y_pred = tf.round(y_one_prob)
        with tf.name_scope("loss"):
            # Compute the cross-entropy term for each datapoint
            y_expand = tf.expand_dims(y, 1)
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
            # Multiply by weights
            if weight_positives:
                w_expand = tf.expand_dims(w, 1)
                entropy = w_expand * entropy
            # Sum all contributions
            l = tf.reduce_sum(entropy)

        with tf.name_scope("optim"):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)

        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", l)
            merged = tf.summary.merge_all()

        hyperparam_str = "d-%d-hidden-%d-lr-%f-n_epochs-%d-batch_size-%d-weight_pos-%s" % (
            d, n_hidden, learning_rate, n_epochs, batch_size, str(weight_positives))
        train_writer = tf.summary.FileWriter('/tmp/fcnet-func-' + hyperparam_str,
                                             tf.get_default_graph())
        N = train_X.shape[0]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            for epoch in range(n_epochs):
                pos = 0
                while pos < N:
                    batch_X = train_X[pos:pos + batch_size]
                    batch_y = train_y[pos:pos + batch_size]
                    batch_w = train_w[pos:pos + batch_size]
                    feed_dict = {x: batch_X, y: batch_y, w: batch_w, keep_prob: dropout_prob}
                    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
                    print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
                    train_writer.add_summary(summary, step)

                    step += 1
                    pos += batch_size

            # Make Predictions (set keep_prob to 1.0 for predictions)
            valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X, keep_prob: 1.0})

        weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
        print("Valid Weighted Classification Accuracy: %f" % weighted_score)
    return weighted_score


if __name__ == "__main__":
    scores = {}
    n_reps = 3
    hidden_sizes = [30, 60]
    epochs = [15, 30, 45]
    dropouts = [.5]
    num_layers = [1, 2]

    for rep in range(n_reps):
        for n_epochs in epochs:
            for hidden_size in hidden_sizes:
                for dropout in dropouts:
                    for n_layers in num_layers:
                        score = eval_tox21_hyperparams(n_hidden=hidden_size, n_epochs=n_epochs,
                                                       dropout_prob=dropout, n_layers=n_layers)
                        if (hidden_size, n_epochs, dropout, n_layers) not in scores:
                            scores[(hidden_size, n_epochs, dropout, n_layers)] = []
                        scores[(hidden_size, n_epochs, dropout, n_layers)].append(score)
    print("All Scores")
    print(scores)

    avg_scores = {}
    for params, param_scores in scores.iteritems():
        avg_scores[params] = np.mean(np.array(param_scores))
    print("Scores Averaged over %d repetitions" % n_reps)
    print(avg_scores)
