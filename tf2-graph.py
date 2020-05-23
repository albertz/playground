#!/usr/bin/env python3

"""
Best practice graph-mode development with TF2?

https://www.tensorflow.org/guide/migrate

`import tensorflow.compat.v1 as tf`? (useful to not change too much...)

https://www.tensorflow.org/guide/function
https://github.com/rwth-i6/returnn/issues/292
https://stackoverflow.com/questions/61964379/tf-data-dataset-runs-on-cpu-except-of-prefetchdataset
https://stackoverflow.com/questions/61964754/is-there-a-queue-like-dataset
https://stackoverflow.com/questions/61964090/running-defun-in-graph-mode
https://stackoverflow.com/questions/61973237/parallel-execution-of-tf-ops-in-eager-code

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

  print(tf.autograph.to_code(f.python_function))

  x = tf.constant(13.)
  y = f(x)

  @tf.function
  def eager_func(x):
    while tf.reduce_sum(x) > 1:
      tf.print(x)
      tf.print("hello")
      x = tf.tanh(x)
    return x

  print(tf.autograph.to_code(eager_func.python_function))

  y2 = eager_func(tf.random.uniform([5]))

  v = tf.Variable(17)
  assert isinstance(v, tf.Variable)

  # noinspection PyProtectedMember
  from tensorflow.python.data.ops.dataset_ops import _GeneratorDataset as GeneratorDataset
  from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter

  def raise_out_of_range_error():
    empty_dataset = tf.data.Dataset.from_tensor_slices(tf.fill([0], 0))
    return DatasetV1Adapter(empty_dataset).make_one_shot_iterator().get_next()

  @tf.function(autograph=False)
  def init_func(x):
    with tf.control_dependencies([tf.print(["init_func", x]), v.assign(0)]):
      return tf.identity(x)

  @tf.function(autograph=False)
  def next_func(x):
    res = tf.identity(v)
    with tf.control_dependencies([res]):
      with tf.control_dependencies([tf.print(["next_func", x, res])]):
        end_check = tf.cond(
          pred=tf.greater_equal(res, 13),
          true_fn=raise_out_of_range_error,
          false_fn=lambda: tf.constant(0))
        with tf.control_dependencies([end_check]):
          with tf.control_dependencies([v.assign_add(1)]):
            return tf.identity(res)

  @tf.function(autograph=False)
  def finalize_func(x):
    with tf.control_dependencies([tf.print(["finalize_func", x, v])]):
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
    # session.run(y2)

    session.run(ds_iter_init)
    while True:
      try:
        print(session.run(ds_iter.get_next()))
      except tf.errors.OutOfRangeError:
        print("OutOfRangeError")
        break


if __name__ == '__main__':
  main()
