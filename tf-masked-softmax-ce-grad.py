
"""
Demonstrate how to get some non-zero grad with ``0.0 * loss``.
"""

import tensorflow as tf
import better_exchook


def main():
  max_seq_len = 15
  seq_len = 10
  logits = tf.zeros([max_seq_len])
  mask = tf.less(tf.range(max_seq_len), seq_len)
  logits_masked = tf.where(mask, logits, float("-inf"))

  ce = -tf.reduce_sum(tf.where(mask, tf.nn.softmax(logits_masked) * tf.nn.log_softmax(logits_masked), 0.0))
  loss = 0.0 * ce

  d_logits, = tf.gradients(loss, [logits])

  with tf.compat.v1.Session() as session:
    print(session.run((ce, loss, d_logits)))


if __name__ == "__main__":
  better_exchook.install()
  tf.compat.v1.disable_eager_execution()
  main()

