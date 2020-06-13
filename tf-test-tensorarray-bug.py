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
from tensorflow.python.ops import rnn_cell_impl
try:
  from tensorflow.contrib.rnn.ops import gen_lstm_ops as gen_rnn_ops
except ImportError:
  from tensorflow.python.ops import gen_rnn_ops


if not getattr(tf, "compat", None):
  class Dummy: pass
  tf.compat = Dummy()
if not getattr(tf.compat, "v1", None):
  tf.compat.v1 = tf

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
  if tf.__version__.startswith("1."):
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


# copy from: https://github.com/tensorflow/tensorflow/blob/v1.15.3/tensorflow/contrib/rnn/python/ops/lstm_ops.py
class LSTMBlockCell(rnn_cell_impl.LayerRNNCell):
  def __init__(self,
               num_units,
               forget_bias=1.0,
               cell_clip=None,
               use_peephole=False,
               dtype=None,
               reuse=None,
               name="lstm_cell"):

    super(LSTMBlockCell, self).__init__(_reuse=reuse, dtype=dtype, name=name)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._use_peephole = use_peephole
    self._cell_clip = cell_clip if cell_clip is not None else -1
    self._names = {
        "W": "kernel",
        "b": "bias",
        "wci": "w_i_diag",
        "wcf": "w_f_diag",
        "wco": "w_o_diag",
        "scope": "lstm_cell"
    }
    # Inputs must be 2-dimensional.
    from tensorflow.python.keras.engine import input_spec
    self.input_spec = input_spec.InputSpec(ndim=2)

  @property
  def state_size(self):
    return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if not inputs_shape.dims[1].value:
      raise ValueError(
          "Expecting inputs_shape[1] to be set: %s" % str(inputs_shape))
    input_size = inputs_shape.dims[1].value
    self._kernel = self.add_variable(
        self._names["W"], [input_size + self._num_units, self._num_units * 4])
    self._bias = self.add_variable(
        self._names["b"], [self._num_units * 4],
        initializer=tf.constant_initializer(0.0))
    if self._use_peephole:
      self._w_i_diag = self.add_variable(self._names["wci"], [self._num_units])
      self._w_f_diag = self.add_variable(self._names["wcf"], [self._num_units])
      self._w_o_diag = self.add_variable(self._names["wco"], [self._num_units])

    self.built = True

  def call(self, inputs, state):
    """Long short-term memory cell (LSTM)."""
    if len(state) != 2:
      raise ValueError("Expecting state to be a tuple with length 2.")

    if self._use_peephole:
      wci = self._w_i_diag
      wcf = self._w_f_diag
      wco = self._w_o_diag
    else:
      wci = wcf = wco = tf.zeros([self._num_units], dtype=self.dtype)

    (cs_prev, h_prev) = state
    (_, cs, _, _, _, _, h) = gen_rnn_ops.LSTMBlockCell(
        x=inputs,
        cs_prev=cs_prev,
        h_prev=h_prev,
        w=self._kernel,
        b=self._bias,
        wci=wci,
        wcf=wcf,
        wco=wco,
        forget_bias=self._forget_bias,
        cell_clip=self._cell_clip,
        use_peephole=self._use_peephole)

    new_state = rnn_cell_impl.LSTMStateTuple(cs, h)
    return h, new_state


def _LSTMBlockCellGrad(op, *grad):
  """Gradient for LSTMBlockCell."""
  (x, cs_prev, h_prev, w, wci, wcf, wco, b) = op.inputs
  (i, cs, f, o, ci, co, _) = op.outputs
  (_, cs_grad, _, _, _, _, h_grad) = grad

  from tensorflow.python.ops import nn_ops

  batch_size = x.get_shape().with_rank(2).dims[0].value
  if batch_size is None:
    batch_size = -1
  input_size = x.get_shape().with_rank(2).dims[1].value
  if input_size is None:
    raise ValueError("input_size from `x` should not be None.")
  cell_size = cs_prev.get_shape().with_rank(2).dims[1].value
  if cell_size is None:
    raise ValueError("cell_size from `cs_prev` should not be None.")

  (cs_prev_grad, dgates, wci_grad, wcf_grad,
   wco_grad) = gen_rnn_ops.LSTMBlockCellGrad(
       x=x,
       cs_prev=cs_prev,
       h_prev=h_prev,
       w=w,
       wci=wci,
       wcf=wcf,
       wco=wco,
       b=b,
       i=i,
       cs=cs,
       f=f,
       o=o,
       ci=ci,
       co=co,
       cs_grad=cs_grad,
       h_grad=h_grad,
       use_peephole=op.get_attr("use_peephole"))

  # Backprop from dgates to xh.
  xh_grad = tf.matmul(dgates, w, transpose_b=True)

  x_grad = tf.slice(xh_grad, (0, 0), (batch_size, input_size))
  x_grad.get_shape().merge_with(x.get_shape())

  h_prev_grad = tf.slice(xh_grad, (0, input_size),
                                (batch_size, cell_size))
  h_prev_grad.get_shape().merge_with(h_prev.get_shape())

  # Backprop from dgates to w.
  xh = tf.concat([x, h_prev], 1)
  w_grad = tf.matmul(xh, dgates, transpose_a=True)
  w_grad.get_shape().merge_with(w.get_shape())

  # Backprop from dgates to b.
  b_grad = nn_ops.bias_add_grad(dgates)
  b_grad.get_shape().merge_with(b.get_shape())

  return (x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad,
          wco_grad, b_grad)

try:
  tf.RegisterGradient("LSTMBlockCell")(_LSTMBlockCellGrad)
except KeyError:
  pass  # maybe already registered... (earlier TF versions)


def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--nsteps", type=int, default=-1)
  arg_parser.add_argument("--reset_after_nsteps", type=int, default=100)
  arg_parser.add_argument("--input_dim", type=int, default=2)
  arg_parser.add_argument("--classes_dim", type=int, default=3)
  arg_parser.add_argument("--batch_size", type=int, default=2)
  arg_parser.add_argument("--seq_len", type=int, default=11)
  arg_parser.add_argument("--unroll", action="store_true")
  arg_parser.add_argument("--add_check_numerics", action="store_true")
  args = arg_parser.parse_args([] if IN_COLAB else sys.argv[1:])

  print("TF version:", tf.__version__)

  if tf.__version__.startswith("2."):
    # tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
  # If in Colab, and you run this repeatedly.
  tf.compat.v1.reset_default_graph()
  # enable_check_numerics_v2()

  rnd = numpy.random.RandomState(42)
  n_input_dim = args.input_dim
  n_classes_dim = args.classes_dim
  n_batch = args.batch_size
  n_time = args.seq_len
  x_np = rnd.normal(size=(n_batch, n_time, n_input_dim))
  targets_np = rnd.randint(0, n_classes_dim, size=(n_batch, n_time))

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

  # from tensorflow.python.ops import rnn_cell
  # s_lstm = rnn_cell.LSTMCell(5, name="s")  # originally was LSTMBlockCell
  # s_lstm = tf.keras.layers.LSTMCell(5, name="s")  # originally was LSTMBlockCell
  # from tensorflow.contrib.rnn.python.ops.lstm_ops import LSTMBlockCell
  s_lstm = LSTMBlockCell(num_units=5, name="s")

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

  values = (
    0,  # t
    tf.zeros([batch, tf.shape(encoder)[-1]]),  # prev_c
    s_lstm.zero_state(batch_size=batch, dtype=tf.float32),  # prev_s_state
    c_ta, s_ta)

  if args.unroll:
    for t in range(n_time):
      values = loop_body(*values)
  else:
    values = tf.while_loop(
      cond=loop_cond, body=loop_body,
      loop_vars=values)
  _, _, _, c_ta, s_ta = values

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

  check_op = add_check_numerics_v1_ops() if args.add_check_numerics else tf.no_op()
  with tf.control_dependencies([check_op, minimize_op]):
    loss = tf.identity(loss)
  vars_init_op = tf.compat.v1.global_variables_initializer()

  count_errors = 0
  loss_np = float("inf")
  with tf.compat.v1.Session() as session:
    session.run(vars_init_op)
    step = 0
    while True:
      print("step %i, loss:" % step, session.run(loss, feed_dict={x: x_np, targets: targets_np}))
      loss_np_ = session.run(loss_eval, feed_dict={x: x_np, targets: targets_np})
      # print("step %i, loss (eval):" % step, loss_np_)
      if not numpy.isfinite(loss_np_):  # or loss_np_ > loss_np * 2.
        print("ERR, loss invalid:", loss_np_)
        count_errors += 1
        if count_errors >= 10:
          sys.exit(1)
      loss_np = loss_np_
      step += 1
      if 0 <= args.nsteps <= step:
        print("Stop after %i steps." % args.nsteps)
        break
      if args.reset_after_nsteps >= 0 and step % args.reset_after_nsteps == 0:
        print("Reset after %i steps." % args.reset_after_nsteps)
        session.run(vars_init_op)
        loss_np = float("inf")


if __name__ == '__main__':
  better_exchook.install()
  try:
    main()
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
