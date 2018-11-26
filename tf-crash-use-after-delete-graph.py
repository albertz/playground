#!/usr/bin/env python3

# https://github.com/tensorflow/tensorflow/issues/22770

import gc
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow as c_api
from tensorflow.python.framework import c_api_util

print("TensorFlow:", tf.__version__, tf.__file__)

def graph_del(graph):
    print("del:", graph)

tf.Graph.__del__ = graph_del

graph = tf.Graph()
with graph.as_default():
    for i in range(1000):  # such that we fill up the memory a bit
        #x = tf.placeholder(tf.float32)
        x = tf.constant(42)
        if i == 0:
            x_c_op = x.op._c_op

# New graph, such that we remove traces to graph. Also to fill some memory.
with tf.Graph().as_default():
    for i in range(1000):  # such that we fill up the memory a bit
        x = tf.placeholder(tf.float32)

# Fill some more memory.
a = [bytes([255] * 10000000) for i in range(10)]

del graph
del x
gc.collect()
gc.collect()

print(c_api.TF_OperationName(x_c_op))
print(c_api.TF_OperationOpType(x_c_op))
print(c_api.TF_OperationDevice(x_c_op))
print(c_api.TF_OperationNumOutputs(x_c_op))

with c_api_util.tf_buffer() as buf:
    c_api.TF_OperationToNodeDef(x_c_op, buf)

