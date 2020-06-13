#!/usr/bin/env python3

"""
Trying to reproduce inf/nan bug in TensorFlow.
https://github.com/rwth-i6/returnn/issues/297

Following test_rec_subnet_train_t3d_simple from RETURNN.

"""

import sys
import numpy
import tensorflow as tf
import better_exchook
import argparse


# https://stackoverflow.com/questions/53581278/test-if-notebook-is-running-on-google-colab
IN_COLAB = 'google.colab' in sys.modules

# The checks could increase the memory usage a lot.
# Ignore some common ops which should not be able to introduce inf/nan.
CheckNumericsIgnoreOps = {
  "Add", "AddN", "Sum", "Mul", "MatMul", "Sub", "L2Loss", "Floor", "Neg", "UnsortedSegmentSum",
  "Switch", "Merge", "PreventGradient",
  "Select", "Maximum", "Minimum", "Abs", "Sign",
  "Const", "Identity", "Fill", "ZerosLike",
  "Reshape", "Tile", "ExpandDims", "ConcatV2", "Transpose",
  "Slice", "StridedSlice", "StridedSliceGrad", "Gather",
  "TruncatedNormal", "RandomUniform",
  "TensorArrayV3"}


def add_check_numerics_v1_ops(name="add_check_numerics_ops", verbose=False):
  """
  This is similar to :func:`tf.add_check_numerics_ops` and based on similar code.
  It adds some more logic and options.
  Copied from RETURNN, and simplified.

  :param str name: op-name for the final tf.group
  :param bool verbose:
  :return: operation which performs all the checks
  :rtype: tf.Operation
  """
  ops = tf.compat.v1.get_default_graph().get_operations()
  with tf.name_scope(name):
    check_op = []
    # This code relies on the ordering of ops in get_operations().
    # The producer of a tensor always comes before that tensor's consumer in
    # this list. This is true because get_operations() returns ops in the order
    # added, and an op can only be added after its inputs are added.
    for op in ops:
      assert isinstance(op, tf.Operation)
      if op.type in CheckNumericsIgnoreOps:
        continue
      # Frames from within a while-loop are partly broken.
      # https://github.com/tensorflow/tensorflow/issues/2211
      # noinspection PyProtectedMember
      if op._get_control_flow_context() != tf.compat.v1.get_default_graph()._get_control_flow_context():
        continue
      for output in op.outputs:
        if output.dtype not in [tf.float16, tf.float32, tf.float64]:
          continue
        message = op.name + ":" + str(output.value_index)
        with tf.control_dependencies(check_op):
          if verbose:
            print("add check for:", output, op.type)
          check_op = [tf.compat.v1.check_numerics(output, message=message, name=op.name + "_check_numerics")]
    return tf.group(*check_op)


def enable_check_numerics_v2():
  if tf.__version__.endswith("1."):
    return
  from tensorflow.python.debug.lib import check_numerics_callback
  # The flow value of TensorArray/TensorArrayV3 is uninitalized on GPU,
  # i.e. can be anything.
  # https://github.com/tensorflow/tensorflow/blob/v.2.2.0/tensorflow/core/kernels/tensor_array_ops.cc#L143
  check_numerics_callback.IGNORE_OP_OUTPUTS += (
    (b"TensorArrayV3", 1),)
  # Extend the safe/ignore list.
  for op_type in CheckNumericsIgnoreOps:
    check_numerics_callback.SAFE_OPS += (op_type.encode("utf8"),)
  tf.debugging.enable_check_numerics()


def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--nsteps", type=int, default=-1)
  arg_parser.add_argument("--reset_after_nsteps", type=int, default=-1)
  args = arg_parser.parse_args(
    ["--reset_after_nsteps", "100"] if IN_COLAB else sys.argv[1:])

  print("TF version:", tf.__version__)

  # tf.compat.v1.disable_eager_execution()
  tf.compat.v1.disable_v2_behavior()
  # If in Colab, and you run this repeatedly.
  tf.compat.v1.reset_default_graph()
  enable_check_numerics_v2()

  n_input_dim = 2
  n_classes_dim = 3
  x = tf.compat.v1.placeholder(tf.float32, shape=(None, None, n_input_dim), name="x")
  targets = tf.compat.v1.placeholder(tf.int32, shape=(None, None), name="targets")
  encoder = tf.keras.layers.Dense(units=5, activation="tanh", name="encoder")(x)
  batch = tf.shape(encoder)[0]
  size = tf.shape(encoder)[1]
  orth_embed = tf.keras.layers.Embedding(input_dim=n_classes_dim, output_dim=6)(targets)  # (B,T,D)
  orth_embed = tf.transpose(orth_embed, [1, 0, 2])  # (T,B,D)
  prev_orth_embed = tf.concat(
    [tf.zeros([1, batch, orth_embed.get_shape().as_list()[-1]]), orth_embed[:-1]], axis=0)  # (T,B,D)
  prev_orth_embed_ta = tf.TensorArray(
    tf.float32, name="prev_orth_embed_ta", dynamic_size=True, size=0,
    element_shape=(None, prev_orth_embed.get_shape().as_list()[-1]))
  prev_orth_embed_ta = prev_orth_embed_ta.unstack(prev_orth_embed)
  c_ta = tf.TensorArray(tf.float32, name="c_ta", dynamic_size=True, size=0)
  s_ta = tf.TensorArray(tf.float32, name="s_ta", dynamic_size=True, size=0)

  def loop_cond(t, *args):
    return tf.less(t, size)

  s_lstm = tf.keras.layers.LSTMCell(5, name="s")  # originally was LSTMBlockCell

  def loop_body(t, prev_c, prev_s_state, c_ta_, s_ta_):
    assert isinstance(prev_c, tf.Tensor)
    prev_c.set_shape((None, encoder.get_shape().as_list()[-1]))
    prev_orth_embed_t = prev_orth_embed_ta.read(t)  # (B,D)
    s_in_in = tf.concat([prev_c, prev_orth_embed_t], axis=-1)  # (B,D)
    s_in = tf.keras.layers.Dense(units=5, name="s_in", activation="tanh")(s_in_in)
    s, s_state = s_lstm(s_in, prev_s_state)
    c_in = s  # (B,D)

    # dot attention
    base = encoder  # (batch, base_time, n_out)
    base_ctx = encoder  # (batch, base_time, inner)
    source = tf.expand_dims(c_in, axis=2)  # (batch, inner, 1)
    energy = tf.matmul(base_ctx, source)  # (batch, base_time, 1)
    energy.set_shape(tf.TensorShape([None, None, 1]))
    energy = tf.squeeze(energy, axis=2)  # (batch, base_time)
    energy_mask = tf.sequence_mask(tf.fill([batch], size), maxlen=tf.shape(energy)[1])
    # NOTE: The following line seems to trigger it!
    energy = tf.where(energy_mask, energy, float("-inf") * tf.ones_like(energy))
    base_weights = tf.nn.softmax(energy)  # (batch, base_time)
    base_weights_bc = tf.expand_dims(base_weights, axis=1)  # (batch, 1, base_time)
    out = tf.matmul(base_weights_bc, base)  # (batch, 1, n_out)
    out.set_shape(tf.TensorShape([None, 1, base.get_shape().as_list()[-1]]))
    c = tf.squeeze(out, axis=1)  # (batch, n_out)

    assert isinstance(c_ta_, tf.TensorArray)
    assert isinstance(s_ta_, tf.TensorArray)
    c_ta_ = c_ta_.write(t, c)
    s_ta_ = s_ta_.write(t, s)

    return t + 1, c, s_state, c_ta_, s_ta_

  _, _, _, c_ta, s_ta = tf.while_loop(
    cond=loop_cond, body=loop_body,
    loop_vars=(
      0,  # t
      tf.zeros([batch, tf.shape(encoder)[-1]]),  # prev_c
      s_lstm.get_initial_state(batch_size=batch, dtype=tf.float32),  # prev_s_state
      c_ta, s_ta))

  assert isinstance(c_ta, tf.TensorArray)
  assert isinstance(s_ta, tf.TensorArray)
  c_ = c_ta.stack()  # (T,B,D)
  s_ = s_ta.stack()  # (T,B,D)
  cs = tf.concat([c_, s_], axis=-1)  # (T,B,D)
  cs = tf.transpose(cs, [1, 0, 2])  # (B,T,D)
  att = tf.keras.layers.Dense(units=6, name="att", activation="tanh")(cs)
  output_logits = tf.keras.layers.Dense(units=n_classes_dim, name="output_prob", activation=None)(att)
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=output_logits)  # (B,T)
  loss = tf.reduce_mean(loss)
  loss_eval = loss

  opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)  # originally was NadamOptimizer...
  minimize_op = opt.minimize(loss)

  check_op = add_check_numerics_v1_ops()
  with tf.control_dependencies([check_op, minimize_op]):
    loss = tf.identity(loss)
  vars_init_op = tf.compat.v1.global_variables_initializer()

  rnd = numpy.random.RandomState(42)
  n_batch = 2
  n_time = 5
  x_np = rnd.normal(size=(n_batch, n_time, n_input_dim))
  targets_np = rnd.randint(0, n_classes_dim, size=(n_batch, n_time))

  with tf.compat.v1.Session() as session:
    session.run(vars_init_op)
    step = 0
    while True:
      print("step %i, loss:" % step, session.run(loss, feed_dict={x: x_np, targets: targets_np}))
      print("step %i, loss (eval):" % step, session.run(loss_eval, feed_dict={x: x_np, targets: targets_np}))
      step += 1
      if 0 <= args.nsteps <= step:
        print("Stop after %i steps." % args.nsteps)
        break
      if args.reset_after_nsteps >= 0 and step % args.reset_after_nsteps == 0:
        print("Reset after %i steps." % args.reset_after_nsteps)
        session.run(vars_init_op)


if __name__ == '__main__':
  better_exchook.install()
  main()
