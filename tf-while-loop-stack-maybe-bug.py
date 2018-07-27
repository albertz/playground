#!/usr/bin/env python3

"""
tf.while_loop uses Stack internally to accumulate gradients for loop invariants (e.g. vars),
end in the backprop loop it pops from the stack.
I'm not quite sure whether they do the ordering correct.
I'm testing this here.
"""

import better_exchook
better_exchook.install()
import tensorflow as tf
import numpy


with tf.Session().as_default() as session:

  n = 1000
  x = tf.constant(1.0)
  y_ta = tf.TensorArray(tf.float32, size=n, element_shape=())


  def body(i, y_ta):
    """
    :param tf.Tensor i: scalar
    :param tf.TensorArray y:
    :return: i + 1, new y
    """
    v = (-1.0 ** tf.to_float(i)) * x
    y_ta = y_ta.write(i, v)
    return i + 1, y_ta


  _, y_ta = tf.while_loop(
    cond=lambda i, *args: tf.less(i, n),
    body=body,
    loop_vars=(0, y_ta),
    parallel_iterations=n)
  assert isinstance(y_ta, tf.TensorArray)
  y = y_ta.stack()
  y.set_shape((n,))

  dx, = tf.gradients(y, x, grad_ys=[[-1.0 ** i for i in range(n)]])

  print(dx)
  np_dx = session.run(dx)
  print(np_dx)
  assert numpy.isclose(np_dx, n)
