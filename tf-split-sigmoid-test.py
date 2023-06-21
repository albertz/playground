#!/usr/bin/env python3

"""
Via Eugen Beck.
"""

import math

import tensorflow as tf
import numpy as np

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def my_sigmoid(x):
    return 1 / (1 + tf.exp(-x))

def plus_zero(x):
  return x + 0.0

data = [[[2.0, 3.0]]]

tf.compat.v1.disable_eager_execution()

s = tf.compat.v1.Session()
p = tf.compat.v1.placeholder(dtype=tf.float32)
a, b = tf.split(p, 2, axis=2)
sig = tf.sigmoid(a)
m = sig * b

out = s.run([p, m], feed_dict={p: data})
print(out)

print('computed: ', out[-1][0,0,0])
print('expected: ', sigmoid(data[0][0][0]) * data[0][0][1])
