#!/usr/bin/env python3

# https://github.com/tensorflow/tensorflow/issues/2652

import tensorflow as tf
from pprint import pprint


print("TensorFlow:", tf.__version__)


def run(on_gpu):
    tf.reset_default_graph()
    tf.set_random_seed(42)
    with tf.device('/gpu:0' if on_gpu else '/cpu:0'):
        a = tf.random_normal([100, 100])
        b = tf.get_variable('b', shape = [], initializer = tf.constant_initializer(value = 0.0))
        c = a*b
        grad = tf.gradients(c, [b], gate_gradients=tf.train.Optimizer.GATE_GRAPH)[0]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    grad_val = sess.run(grad)
    return grad_val


for i in range(20):
    print(repr(run(on_gpu=True)))
    if i == 0:
        ops = tf.get_default_graph().get_operations()
        pprint(ops)