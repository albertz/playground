
import tensorflow as tf
from typing import Tuple

print("TensorFlow:", tf.__version__)


tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

session = tf.compat.v1.Session()


# https://www.tensorflow.org/guide/function

@tf.function
def f(x):
  while tf.reduce_sum(x) > 1:
    tf.print(x)
    x = tf.tanh(x)
  return x


session.run(f(tf.random.uniform([5])))


@tf.function
def f(x):
  seq_len = tf.shape(x)[0]
  for t in tf.range(seq_len):
    tf.print(t)
  return x


session.run(f(tf.random.uniform([5])))


chunk_step = 10
eoc_idx = 0
blank_idx = 9


@tf.function
def _f(in_: tf.Tensor, in_sizes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  batch_size = tf.shape(in_)[0]
  batched_ta = tf.TensorArray(dtype=tf.int32, size=batch_size, element_shape=[None])
  batched_ta_seq_lens = tf.TensorArray(dtype=tf.int32, size=batch_size, element_shape=[])
  for b in tf.range(batch_size):
    x = in_[b][:in_sizes[b]]
    ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, element_shape=[])
    i = 0
    for t in tf.range(in_sizes[b]):
      if t % chunk_step == 0 and t > 0:
        ta = ta.write(i, eoc_idx)
        i += 1
      if x[t] == blank_idx:
        continue
      ta = ta.write(i, x[t])
      i += 1
    ta = ta.write(i, eoc_idx)
    batched_ta = batched_ta.write(b, ta.stack())  # [i]
    batched_ta_seq_lens = batched_ta_seq_lens.write(b, i + 1)

  seq_lens = batched_ta_seq_lens.stack()
  # stack batched_ta using padding
  max_len = tf.reduce_max(seq_lens)
  batched_ta_ = tf.TensorArray(dtype=tf.int32, size=batch_size, element_shape=[None])
  for b in tf.range(batch_size):
    x = batched_ta.read(b)
    batched_ta_ = batched_ta_.write(b, tf.pad(x, [[0, max_len - tf.shape(x)[0]]]))
  return batched_ta_.stack(), seq_lens


session.run(_f(
  tf.random.uniform([5, 10], minval=0, maxval=10, dtype=tf.int32),
  tf.random.uniform([5], minval=1, maxval=10, dtype=tf.int32)))
