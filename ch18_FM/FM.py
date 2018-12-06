# encoding:utf-8
from itertools import count
from collections import defaultdict
from scipy.sparse import csr
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm


def vectorize_dic(dic,ix=None,p=None,n=0,g=0):
    """
    dic:pepole-item的字典组合，将它转化成one-hot的稀疏矩阵，行代表用户，列代表每个商品

    ix -- index generator (default None)
    p -- dimension of feature space (number of columns in the sparse matrix) (default None)
    n:表示有n个人
    """
    if ix==None:
        ix = dict()

    nz = n * g

    col_ix = np.empty(nz,dtype = int)

    i = 0
    for k,lis in dic.items():
        for t in range(len(lis)):
            ix[str(lis[t]) + str(k)] = ix.get(str(lis[t]) + str(k),0) + 1
            col_ix[i+t*g] = ix[str(lis[t]) + str(k)]
        i += 1

    row_ix = np.repeat(np.arange(0,n),g)
    data = np.ones(nz)
    if p == None:
        p = len(ix)

    ixx = np.where(col_ix < p)
    return csr.csr_matrix((data[ixx],(row_ix[ixx],col_ix[ixx])),shape=(n,p)),ix


def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)


cols = ['user','item','rating','timestamp']

train = pd.read_csv('/home/zt/Documents/Data_Tensorflow_keras_sklearn/ch18_FM/ua.base',delimiter='\t',names = cols)
test = pd.read_csv('/home/zt/Documents/Data_Tensorflow_keras_sklearn/ch18_FM/ua.test',delimiter='\t',names = cols)

x_train,ix = vectorize_dic({'users':train['user'].values,
                            'items':train['item'].values},n=len(train.index),g=2)


x_test,ix = vectorize_dic({'users':test['user'].values,
                           'items':test['item'].values},ix,x_train.shape[1],n=len(test.index),g=2)


print(x_train)
y_train = train['rating'].values
y_test = test['rating'].values

x_train = x_train.todense()
x_test = x_test.todense()

print(x_train)

print(x_train.shape)
print (x_test.shape)


n,p = x_train.shape # n个人，p个商品

k = 10

x = tf.placeholder('float',[None,p])

y = tf.placeholder('float',[None,1])

# 线性部分的偏执w0, 线性部分w
w0 = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.zeros([p]))

# 交互部分,p个特征k个交互
v = tf.Variable(tf.random_normal([k,p],mean=0,stddev=0.01))


# 线性部分
linear_terms = tf.add(w0,tf.reduce_sum(tf.multiply(w,x),1,keep_dims=True)) # n * 1
# 特征组合部分
pair_interactions = 0.5 * tf.reduce_sum(
    tf.subtract(
        tf.pow(tf.matmul(x,tf.transpose(v)),2),
        tf.matmul(tf.pow(x,2), tf.transpose(tf.pow(v,2)))
    ),axis = 1 , keep_dims=True)

# 将线性部分与特征组合部分结合
y_hat = tf.add(linear_terms,pair_interactions)

lambda_w = tf.constant(0.001,name='lambda_w')
lambda_v = tf.constant(0.001,name='lambda_v')

l2_norm = tf.reduce_sum(
    tf.add(
        tf.multiply(lambda_w,tf.pow(w,2)),
        tf.multiply(lambda_v,tf.pow(v,2))
    )
)

error = tf.reduce_mean(tf.square(y-y_hat))
loss = tf.add(error,l2_norm)


train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)


epochs = 10
batch_size = 1000

# Launch the graph
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in tqdm(range(epochs), unit='epoch'):
        perm = np.random.permutation(x_train.shape[0])
        # iterate over batches
        for bX, bY in batcher(x_train[perm], y_train[perm], batch_size):
            _,t = sess.run([train_op,loss], feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)})
            print(t)


    errors = []
    for bX, bY in batcher(x_test, y_test):
        errors.append(sess.run(error, feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)}))
        print(errors)
    RMSE = np.sqrt(np.array(errors).mean())
    print (RMSE)