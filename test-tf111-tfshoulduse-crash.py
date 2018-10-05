
"""
Running this with TF 1.11, Python 3.6.3 on Ubuntu 16.04 results in this crash:

...
init vars
graph size: 658062
train
step 0, loss: 3.219436
step 1, loss: 4.317543
step 2, loss: 3.812533
step 3, loss: 9.656103
step 4, loss: 3.692919
step 5, loss: 3.183093
step 6, loss: 3.310433
step 7, loss: 3.173050
step 8, loss: 2.896871
step 9, loss: 3.313305
ERROR:tensorflow:==================================
Object was never used (type <class 'tensorflow.python.ops.tensor_array_ops.TensorArray'>):
<tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7f21d9740c50>
If you want to mark it as used call its "mark_used()" method.
It was originally created here:
  File "test-tf11-tfshoulduse-crash.py", line 328, in <module>    line: test()    locals:      test = <local> <function test at 0x7f21d9721b70>  File "test-tf11-tfshoulduse-crash.py", line 317, in test    line: print("step %i, loss: %f" % (s, loss_val))    locals:      print = <builtin> <built-in function print>      s = <local> 9      loss_val = <local> 3.3133047  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/util/tf_should_use.py", line 189, in wrapped    line: return _add_should_use_warning(fn(*args, **kwargs))    locals:      _add_should_use_warning = <global> <function _add_should_use_warning at 0x7f21e6a5d378>      fn = <local> <function TensorArray.unstack at 0x7f21e6a606a8>      args = <local> (<tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7f21e7bae160>, <tf.Tensor '@:0' shape=(?, ?, 6) dtype=float32>)      kwargs = <local> {}
==================================
ERROR:tensorflow:==================================
Object was never used (type <class 'tensorflow.python.ops.tensor_array_ops.TensorArray'>):
<tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7f21e70e35f8>
If you want to mark it as used call its "mark_used()" method.
It was originally created here:
  File "test-tf11-tfshoulduse-crash.py", line 328, in <module>    line: test()    locals:      test = <local> <function test at 0x7f21d9721b70>  File "test-tf11-tfshoulduse-crash.py", line 317, in test    line: print("step %i, loss: %f" % (s, loss_val))    locals:      print = <builtin> <built-in function print>      s = <local> 9      loss_val = <local> 3.3133047  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/util/tf_should_use.py", line 189, in wrapped    line: return _add_should_use_warning(fn(*args, **kwargs))    locals:      _add_should_use_warning = <global> <function _add_should_use_warning at 0x7f21e6a5d378>      fn = <local> <function TensorArray.unstack at 0x7f21e6a606a8>      args = <local> (<tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7f21e7bae160>, <tf.Tensor '@:0' shape=(?, ?, 6) dtype=float32>)      kwargs = <local> {}  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/ops/tensor_array_ops.py", line 907, in unstack    line: return self._implementation.unstack(value, name=name)    locals:      self = <local> <tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7f21e7bae160>      self._implementation = <local> <tensorflow.python.ops.tensor_array_ops._GraphTensorArray object at 0x7f21d971dcf8>      self._implementation.unstack = <local> <bound method should_use_result.<locals>.wrapped of <tensorflow.python.ops.tensor_array_ops._GraphTensorArray object at 0x7f21d971dcf8>>      value = <local> <tf.Tensor '@:0' shape=(?, ?, 6) dtype=float32>      name = <local> None  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/util/tf_should_use.py", line 189, in wrapped    line: return _add_should_use_warning(fn(*args, **kwargs))    locals:      _add_should_use_warning = <global> <function _add_should_use_warning at 0x7f21e6a5d378>      fn = <local> <function _GraphTensorArray.unstack at 0x7f21e6a5dd08>      args = <local> (<tensorflow.python.ops.tensor_array_ops._GraphTensorArray object at 0x7f21d971dcf8>, <tf.Tensor '@:0' shape=(?, ?, 6) dtype=float32>)      kwargs = <local> {'name': None}
==================================
Fatal Python error: Segmentation fault

Current thread 0x00007f22409ff700 (most recent call first):
  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1897 in name
  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 352 in name
  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 614 in __repr__
  File "/u/zeyer/code/playground/better_exchook.py", line 249 in pretty_print
  File "/u/zeyer/code/playground/better_exchook.py", line 485 in format_py_obj
  File "/u/zeyer/code/playground/better_exchook.py", line 565 in <lambda>
  File "/u/zeyer/code/playground/better_exchook.py", line 520 in _trySet
  File "/u/zeyer/code/playground/better_exchook.py", line 565 in format_tb
  File "/u/zeyer/.linuxbrew/opt/python3/lib/python3.6/traceback.py", line 37 in format_list
  File "/u/zeyer/.linuxbrew/opt/python3/lib/python3.6/traceback.py", line 193 in format_stack
  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/util/tf_should_use.py", line 60 in __del__
fish: Job 1, “python test-tf11-tfshoulduse-cr…” terminated by signal SIGSEGV (Address boundary error)

---

or just:

...
pure virtual method called
terminate called without an active exception
Fatal Python error: Aborted

Current thread 0x00007f1fb6d1b700 (most recent call first):
  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 2249 in node_def
  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 2095 in __str__
  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 353 in name
  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 614 in __repr__
  File "/u/zeyer/code/playground/better_exchook.py", line 249 in pretty_print
  File "/u/zeyer/code/playground/better_exchook.py", line 485 in format_py_obj
  File "/u/zeyer/code/playground/better_exchook.py", line 565 in <lambda>
  File "/u/zeyer/code/playground/better_exchook.py", line 520 in _trySet
  File "/u/zeyer/code/playground/better_exchook.py", line 565 in format_tb
  File "/u/zeyer/.linuxbrew/opt/python3/lib/python3.6/traceback.py", line 37 in format_list
  File "/u/zeyer/.linuxbrew/opt/python3/lib/python3.6/traceback.py", line 193 in format_stack
  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/util/tf_should_use.py", line 60 in __del__
fish: Job 1, “python test-tf111-tfshoulduse-c…” terminated by signal SIGABRT (Abort)

---

or:

ERROR:tensorflow:==================================
Object was never used (type <class 'tensorflow.python.ops.tensor_array_ops.TensorArray'>):
<tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7fbc2fc4b828>
If you want to mark it as used call its "mark_used()" method.
It was originally created here:
  File "test-tf111-tfshoulduse-crash.py", line 299, in <module>    line: test()    locals:      test = <local> <function test at 0x7fbc2229f7b8>  File "test-tf111-tfshoulduse-crash.py", line 288, in test    line: print("step %i, loss: %f" % (s, loss_val))    locals:      print = <builtin> <built-in function print>      s = <local> 9      loss_val = <local> 3.3501906  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/util/tf_should_use.py", line 189, in wrapped    line: return _add_should_use_warning(fn(*args, **kwargs))    locals:      _add_should_use_warning = <global> <function _add_should_use_warning at 0x7fbc2f5db0d0>      fn = <local> <function TensorArray.unstack at 0x7fbc2f5df400>      args = <local> (<tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7fbc307a0e48>, <tf.Tensor 'pe:0' shape=(?, ?, 6) dtype=float32>)      kwargs = <local> {}
==================================
ERROR:tensorflow:==================================
Object was never used (type <class 'tensorflow.python.ops.tensor_array_ops.TensorArray'>):
<tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7fbc2fc4b160>
If you want to mark it as used call its "mark_used()" method.
It was originally created here:
  File "test-tf111-tfshoulduse-crash.py", line 299, in <module>    line: test()    locals:      test = <local> <function test at 0x7fbc2229f7b8>  File "test-tf111-tfshoulduse-crash.py", line 288, in test    line: print("step %i, loss: %f" % (s, loss_val))    locals:      print = <builtin> <built-in function print>      s = <local> 9      loss_val = <local> 3.3501906  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/util/tf_should_use.py", line 189, in wrapped    line: return _add_should_use_warning(fn(*args, **kwargs))    locals:      _add_should_use_warning = <global> <function _add_should_use_warning at 0x7fbc2f5db0d0>      fn = <local> <function TensorArray.unstack at 0x7fbc2f5df400>      args = <local> (<tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7fbc307a0e48>, <tf.Tensor 'pe:0' shape=(?, ?, 6) dtype=float32>)      kwargs = <local> {}  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/ops/tensor_array_ops.py", line 907, in unstack    line: return self._implementation.unstack(value, name=name)    locals:      self = <local> <tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7fbc307a0e48>      self._implementation = <local> <tensorflow.python.ops.tensor_array_ops._GraphTensorArray object at 0x7fbc2229df60>      self._implementation.unstack = <local> <bound method should_use_result.<locals>.wrapped of <tensorflow.python.ops.tensor_array_ops._GraphTensorArray object at 0x7fbc2229df60>>      value = <local> <tf.Tensor 'pe:0' shape=(?, ?, 6) dtype=float32>      name = <local> None  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/util/tf_should_use.py", line 189, in wrapped    line: return _add_should_use_warning(fn(*args, **kwargs))    locals:      _add_should_use_warning = <global> <function _add_should_use_warning at 0x7fbc2f5db0d0>      fn = <local> <function _GraphTensorArray.unstack at 0x7fbc2f5dba60>      args = <local> (<tensorflow.python.ops.tensor_array_ops._GraphTensorArray object at 0x7fbc2229df60>, <tf.Tensor 'b:0' shape=(?, ?, 6) dtype=float32>)      kwargs = <local> {'name': None}
==================================
terminate called after throwing an instance of 'std::length_error'
  what():  basic_string::resize
Fatal Python error: Aborted

Current thread 0x00007fbc8958b700 (most recent call first):
  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 436 in _c_api_shape
  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 407 in shape
  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 492 in get_shape
  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 614 in __repr__
  File "/u/zeyer/code/playground/better_exchook.py", line 249 in pretty_print
  File "/u/zeyer/code/playground/better_exchook.py", line 485 in format_py_obj
  File "/u/zeyer/code/playground/better_exchook.py", line 565 in <lambda>
  File "/u/zeyer/code/playground/better_exchook.py", line 520 in _trySet
  File "/u/zeyer/code/playground/better_exchook.py", line 565 in format_tb
  File "/u/zeyer/.linuxbrew/opt/python3/lib/python3.6/traceback.py", line 37 in format_list
  File "/u/zeyer/.linuxbrew/opt/python3/lib/python3.6/traceback.py", line 193 in format_stack
  File "/u/zeyer/py-envs/py36-tf111/lib/python3.6/site-packages/tensorflow/python/util/tf_should_use.py", line 60 in __del__
fish: Job 1, “python test-tf111-tfshoulduse-c…” terminated by signal SIGABRT (Abort)

"""

import numpy
import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell
import contextlib
import better_exchook
import faulthandler


@contextlib.contextmanager
def make_scope():
  """
  :rtype: tf.Session
  """
  with tf.Graph().as_default() as graph:
    with tf.Session(graph=graph) as session:
      yield session


def test():
  random = numpy.random.RandomState(seed=1)
  num_inputs = 4
  num_outputs = 3
  seq_len = 10
  limit = 1.0

  with make_scope() as session:
    print("create graph")
    tf.set_random_seed(42)
    src_placeholder = tf.placeholder(tf.float32, (None, seq_len, num_inputs), name="src_placeholder")
    tgt_placeholder = tf.placeholder(tf.float32, (None, seq_len, num_outputs), name="tgt_placeholder")
    batch_size = tf.shape(src_placeholder)[0]

    def make_feed_dict():
      return {
        src_placeholder: random.uniform(-limit, limit, (1, seq_len, num_inputs)),
        tgt_placeholder: random.uniform(-limit, limit, (1, seq_len, num_outputs)),
      }

    with tf.variable_scope(tf.get_variable_scope()) as scope:
      x = tf.get_variable("b", shape=(seq_len, 1, num_outputs * 2))
      input_ta = tf.TensorArray(tf.float32, size=seq_len, element_shape=(None, num_outputs * 2))
      input_ta = input_ta.unstack(x)
      target_ta = tf.TensorArray(tf.float32, size=seq_len, element_shape=(None, num_outputs))
      target_ta = target_ta.unstack(tf.transpose(tgt_placeholder, [1, 0, 2]))
      loss_ta = tf.TensorArray(tf.float32, size=seq_len, element_shape=(None,))
      loss = tf.reduce_sum(x ** 2)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1, epsilon=1e-16, use_locking=False)
    minimize_op = optimizer.minimize(loss)
    check_op = tf.no_op()

    print('variables:')
    train_vars = (
            tf.trainable_variables() +
            tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
    print(train_vars)
    print('init vars')
    session.run(tf.global_variables_initializer())
    print('graph size:', session.graph_def.ByteSize())
    print('train')
    for s in range(10):
      loss_val, _, _ = session.run([loss, minimize_op, check_op], feed_dict=make_feed_dict())
      print("step %i, loss: %f" % (s, loss_val))


if __name__ == "__main__":
  better_exchook.install()
  better_exchook.replace_traceback_format_tb()
  faulthandler.enable()
  test()
