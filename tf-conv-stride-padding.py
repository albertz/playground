import tensorflow as tf


n_batch = 1
n_time = 9
n_in_dim = 1
n_out_dim = 1

def _conv(*, strides):
    x = tf.ones((n_batch, n_time, n_in_dim), dtype=tf.float32)
    filters = tf.constant([1, 1, 1, 1], dtype=tf.float32)[:, None, None]  # (n_filter, n_in_dim, n_out_dim)
    filters = tf.tile(filters, [1, n_in_dim, n_out_dim])

    y = tf.nn.convolution(x, filters=filters, strides=strides, padding="SAME")
    print(y)


_conv(strides=3)
_conv(strides=1)

n_time = 7

_conv(strides=3)
_conv(strides=1)
