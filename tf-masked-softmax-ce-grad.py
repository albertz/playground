
"""
Demonstrate how to get some non-zero grad with ``0.0 * loss``.
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import better_exchook


# noinspection PyUnusedLocal
@ops.RegisterGradient("IdentityWithPrint")
def _identity_with_print(op, grad):
  with tf.control_dependencies([tf.print([op.name, "grad:", grad])]):
    return [tf.identity(grad)]


def debug_grad(x):
  """
  :param tf.Tensor x:
  :return: x, but gradient will be printed
  :rtype: tf.Tensor
  """
  g = tf.compat.v1.get_default_graph()
  with g.gradient_override_map({"Identity": "IdentityWithPrint"}):
    return tf.identity(x, name=x.name.split("/")[-1].replace(":", "_"))


def main():
  max_seq_len = 15
  seq_len = 10

  logits = tf.zeros([max_seq_len])
  logits_ = debug_grad(logits)

  mask = tf.less(tf.range(max_seq_len), seq_len)
  logits_masked = tf.where(mask, logits_, float("-inf"))
  ce = -tf.reduce_sum(tf.where(mask, tf.nn.softmax(logits_masked) * tf.nn.log_softmax(logits_masked), 0.0))
  loss = 0.0 * ce

  d_logits, = tf.gradients(loss, [logits])

  with tf.compat.v1.Session() as session:
    print(session.run((ce, loss, d_logits)))


if __name__ == "__main__":
  better_exchook.install()
  tf.compat.v1.disable_eager_execution()
  main()

