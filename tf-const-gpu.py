
# https://github.com/tensorflow/tensorflow/issues/52200
# https://github.com/rwth-i6/returnn/issues/694

import tensorflow as tf


print("TF:", tf.__version__)
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_control_flow_v2()


with tf.compat.v1.Session() as session:
  x = tf.constant("foo")

  def body(i):
    with tf.control_dependencies([tf.print(x)]):
      return i + 1

  n = tf.while_loop(cond=lambda i: tf.less(i, 1), body=body, loop_vars=[0])
  session.run(n)
