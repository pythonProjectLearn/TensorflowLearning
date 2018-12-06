import tensorflow as tf

matrix = tf.constant([[1., 2.]])
negMatrix = tf.negative(matrix)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    result = sess.run(negMatrix)

print(result)
