
import tensorflow as tf

print("TensorFlow:", tf.__version__)


tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

tf.compat.v1.reset_default_graph()

session = tf.compat.v1.Session()


# https://www.tensorflow.org/guide/function


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
