
# https://github.com/rwth-i6/returnn/issues/703
# https://github.com/tensorflow/tensorflow/issues/52223


import tensorflow as tf
import numpy

tf.compat.v1.disable_eager_execution()


with tf.Graph().as_default() as graph:
    with tf.compat.v1.Session(graph=graph) as session:
        x = tf.compat.v1.placeholder(tf.float32, (None, None, 1, 40))  # [B,T,1,40]
        filters = tf.compat.v1.placeholder(tf.float32, (3, 3, None, 32))
        y = tf.compat.v1.nn.convolution(x, filter=filters, padding="SAME")

        session.run(
            y,
            feed_dict={
                x: numpy.zeros((3, 4, 1, 40)),
                filters: numpy.zeros((3, 3, 1, 32)),
                })
