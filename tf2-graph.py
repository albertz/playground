#!/usr/bin/env python3

"""
Best practice graph-mode development with TF2?

https://www.tensorflow.org/guide/migrate

`import tensorflow.compat.v1 as tf`? (useful to not change too much...)

"""

import better_exchook
better_exchook.install()

import tensorflow as tf

if hasattr(tf, "compat"):
  tf_compat_v1 = tf.compat.v1
  tf_compat_v2 = tf.compat.v2
else:
  tf_compat_v1 = None
  tf_compat_v2 = None
  # noinspection PyUnresolvedReferences
  assert tf.VERSION.startswith("1.")


print("TF:", tf.version.VERSION)

if tf_compat_v1:
  tf_compat_v1.disable_eager_execution()
  tf_compat_v1.disable_v2_tensorshape()


def main():
  @tf.function
  def f(x):
    with tf.control_dependencies([tf.print(["f", x])]):
      return tf.identity(x)

  x = tf.constant(13.)
  y = f(x)

  v = tf.Variable([17.])

  # noinspection PyProtectedMember
  from tensorflow.python.data.ops.dataset_ops import _GeneratorDataset as GeneratorDataset
  from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter

  @tf.function
  def init_func(x):
    with tf.control_dependencies([tf.print(["init_func", x])]):
      return tf.identity(x)

  @tf.function
  def next_func(x):
    with tf.control_dependencies([tf.print(["next_func", x])]):
      return tf.identity(x)

  @tf.function
  def finalize_func(x):
    with tf.control_dependencies([tf.print(["finalize_func", x])]):
      return tf.identity(x)

  generator_dataset = GeneratorDataset(
    init_args=tf.constant("dummy_init_args"),
    init_func=init_func,
    next_func=next_func,
    finalize_func=finalize_func)
  generator_dataset_v1 = DatasetV1Adapter(generator_dataset)
  ds_iter = tf_compat_v1.data.make_initializable_iterator(generator_dataset_v1)
  ds_iter_init = ds_iter.make_initializer(generator_dataset_v1)

  with tf_compat_v1.Session() as session:
    session.run(y)
    session.run(ds_iter_init)

    for i in range(3):
      print(session.run(ds_iter.get_next()))


if __name__ == '__main__':
  main()
