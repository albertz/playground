# https://stackoverflow.com/questions/51838390/tf-op-in-switch-context-why-is-switch-needed-on-external-inputs

import tensorflow as tf
from pprint import pprint

x = tf.constant(0.5)

def cond_branch(cond):
  y = tf.multiply(x, 2.0 if cond else 3.0)
  print("cond branch %r" % cond)
  print(y.op)
  print(y.op.inputs[0].op)
  return y

cond = tf.placeholder(tf.bool)

y = tf.cond(
  cond,
  lambda: cond_branch(True),
  lambda: cond_branch(False))


with tf.Session() as session:
  print("y:", session.run(y, feed_dict={cond: True}))
