
import tensorflow as tf

print("TensorFlow:", tf.__version__)

tf.compat.v1.disable_eager_execution()

session = tf.compat.v1.Session()


# https://www.tensorflow.org/guide/function
# https://github.com/tensorflow/tensorflow/issues/59796


def f(x: tf.Tensor):
  from typing import Tuple

  @tf.function
  def local_func(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    while tf.reduce_sum(x) > 1:
      tf.print(x)
      x = tf.tanh(x)
    return x, x

  return local_func(x)


session.run(f(tf.random.uniform([5])))
