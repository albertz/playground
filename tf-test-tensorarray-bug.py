#!/usr/bin/env python3

"""
Trying to reproduce inf/nan bug in TensorFlow.
https://github.com/rwth-i6/returnn/issues/297

Following test_rec_subnet_train_t3d_simple from RETURNN.

"""

import numpy
import tensorflow as tf
import better_exchook


def add_check_numerics_ops(
      ignore_ops=None, use_check_numerics=True, debug_print_added_checks=True,
      name="add_check_numerics_ops"):
  """
  This is similar to :func:`tf.add_check_numerics_ops` and based on similar code.
  It adds some more logic and options.
  Copied from RETURNN, and simplified.

  :param list[tf.Operation|tf.Tensor]|None fetches: in case this is given, will only look at these and dependent ops
  :param list[str] ignore_ops: e.g. ""
  :param bool use_check_numerics: if False, instead of :func:`tf.check_numerics`,
    it does the check manually (via :func:`tf.is_finite`) and in case there is inf/nan,
    it will also print the tensor (while `tf.check_numerics` does not print the tensor).
    Note that this can be about 50 times slower.
  :param bool debug_print_added_checks: prints info about each added check
  :param str name: op-name for the final tf.group
  :return: operation which performs all the checks
  :rtype: tf.Operation
  """
  ops = tf.get_default_graph().get_operations()
  if ignore_ops is None:
    # The checks could increase the memory usage a lot.
    # Ignore some common ops which should not be able to introduce inf/nan.
    ignore_ops = {
      "Add", "AddN", "Sum", "Mul", "MatMul", "Sub", "L2Loss", "Floor", "Neg", "UnsortedSegmentSum",
      "Switch", "Merge", "PreventGradient",
      "Select", "Maximum", "Minimum", "Abs", "Sign",
      "Const", "Identity", "Fill", "ZerosLike",
      "Reshape", "Tile", "ExpandDims", "ConcatV2", "Transpose",
      "Slice", "StridedSlice", "StridedSliceGrad", "Gather",
      "TruncatedNormal", "RandomUniform"}
  with tf.name_scope(name):
    check_op = []
    # This code relies on the ordering of ops in get_operations().
    # The producer of a tensor always comes before that tensor's consumer in
    # this list. This is true because get_operations() returns ops in the order
    # added, and an op can only be added after its inputs are added.
    for op in ops:
      assert isinstance(op, tf.Operation)
      if op.type in ignore_ops:
        continue
      # Frames from within a while-loop are partly broken.
      # https://github.com/tensorflow/tensorflow/issues/2211
      # noinspection PyProtectedMember
      if op._get_control_flow_context() != tf.get_default_graph()._get_control_flow_context():
        continue
      for output in op.outputs:
        if output.dtype not in [tf.float16, tf.float32, tf.float64]:
          continue
        message = op.name + ":" + str(output.value_index)
        with tf.control_dependencies(check_op):
          if debug_print_added_checks:
            print("add check for:", output, op.type)
          if use_check_numerics:
            check_op = [tf.check_numerics(output, message=message, name=op.name + "_check_numerics")]
          else:
            is_finite = tf.reduce_all(tf.is_finite(output))
            check_op = [tf.Assert(is_finite, [message, "Tensor had inf or nan values:", output])]
    return tf.group(*check_op)


def main():
  print("TF version:", tf.__version__)

  # From test_rec_subnet_train_t3d_simple:
  """
  beam_size = 2
  network = {
    "encoder": {"class": "linear", "activation": "tanh", "n_out": 5},
    "output": {"class": "rec", "from": [], "unit": {
      'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["output_prob"]},
      "end": {"class": "compare", "from": ["output"], "value": 0},
      'orth_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6},
      "s_in": {"class": "linear", "activation": "tanh", "from": ["prev:c", "prev:orth_embed"], "n_out": 5},
      "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["s_in"], "n_out": 5},
      "c_in": {"class": "copy", "from": ["s"]},
      "c": {"class": "dot_attention", "from": ["c_in"], "base": "base:encoder", "base_ctx": "base:encoder"},
      "att": {"class": "linear", "activation": "tanh", "from": ["c", "s"], "n_out": 6},
      "output_prob": {"class": "softmax", "from": ["att"], "target": "classes", "loss": "ce"}
    }, "target": "classes", "max_seq_len": 75},
  }

  from GeneratingDataset import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": network,
    "start_epoch": 1,
    "num_epochs": 200,
    "batch_size": 10,
    "nadam": True,
    "learning_rate": 0.01,
    "debug_add_check_numerics_ops": True
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()
  """

  """
  Rec layer 'output' (search False, train 'globals/train_flag:0') sub net:
    Input layers moved out of loop: (#: 2)
      output
      orth_embed
    Output layers moved out of loop: (#: 2)
      output_prob
      att                                                                                                                     
    Layers in loop: (#: 4)
      c
      c_in
      s
      s_in
    Unused layers: (#: 1)
      end  
  """

  n_input_dim = 2
  n_classes_dim = 3
  x = tf.placeholder(tf.float32, shape=(None, None, n_input_dim), name="x")
  targets = tf.placeholder(tf.int32, shape=(None, None), name="targets")
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

  # s_lstm = tf.keras.layers.LSTMCell(units=5, name="s")
  # s_lstm = tf.compat.v1.nn.rnn_cell.LSTMBlockCell(num_units=5, name="s")
  from tensorflow.contrib.rnn.python.ops.lstm_ops import LSTMBlockCell
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

  # opt = tf.train.AdamOptimizer(learning_rate=0.01)
  from tensorflow.contrib.opt.python.training.nadam_optimizer import NadamOptimizer
  opt = NadamOptimizer(learning_rate=0.01)
  minimize_op = opt.minimize(loss)

  check_op = add_check_numerics_ops()
  with tf.control_dependencies([check_op, minimize_op]):
    loss = tf.identity(loss)

  rnd = numpy.random.RandomState(42)
  n_batch = 2
  n_time = 5
  x_np = rnd.normal(size=(n_batch, n_time, n_input_dim))
  targets_np = rnd.randint(0, n_classes_dim, size=(n_batch, n_time))

  with tf.compat.v1.Session() as session:
    session.run(tf.global_variables_initializer())
    while True:
      print("loss:", session.run(loss, feed_dict={x: x_np, targets: targets_np}))
      print("loss:", session.run(loss_eval, feed_dict={x: x_np, targets: targets_np}))


if __name__ == '__main__':
  better_exchook.install()
  main()
