
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


def xavier_initializer(uniform=True, seed=None, dtype=tf.float32):
  """
  Alias for tf.glorot_uniform_initializer or tf.glorot_normal_initializer.

  :param bool uniform: uniform or normal distribution
  :param int seed:
  :param tf.DType dtype:
  :return: ((tuple[int]) -> tf.Tensor) | tf.Initializer
  """
  from tensorflow.python.ops import init_ops
  return init_ops.variance_scaling_initializer(
    scale=1.0, mode='fan_avg', distribution="uniform" if uniform else "normal", seed=seed, dtype=dtype)


@contextlib.contextmanager
def var_creation_scope():
  """
  If you create a variable inside of a while-loop, you might get the following error:
    InvalidArgumentError: The node 'while/w/Assign' has inputs from different frames.
    The input 'while/j' is in frame 'while/while/'. The input 'while/w' is in frame ''.
  Also see tests/test_TFUtil.py:test_loop_var_creation().
  Related TF bugs:
    https://github.com/tensorflow/tensorflow/issues/3114
    https://github.com/tensorflow/tensorflow/issues/4478
    https://github.com/tensorflow/tensorflow/issues/8604
  The solution is to reset the current frame.
  Resetting all control dependencies has this effect.
  See also :func:`same_context`
  """
  with tf.control_dependencies(None) as dep:
    yield dep


def cond(pred, fn1, fn2, name=None):
  """
  This is a wrapper around tf.control_flow_ops.cond().
  This will be a branched execution, i.e. either fn1() or fn2() will be executed,
  or at least the resulting graph will be evaluated.
  If pred can is constant at the call, only the corresponding fn will be called.
  This is similar as the TF internal _smart_cond().
  And similar as tf.contrib.framework.smart_cond.

  :param tf.Tensor|bool pred:
  :param ()->(tf.Tensor|list[tf.Tensor]) fn1:
  :param ()->(tf.Tensor|list[tf.Tensor]) fn2:
  :param str name:
  :return: fn1() if pred else fn2()
  :rtype: tf.Tensor|list[tf.Tensor]
  """
  if not callable(fn1):
    raise TypeError("fn1 must be callable.")
  if not callable(fn2):
    raise TypeError("fn2 must be callable.")
  if pred is True:
    return fn1()
  if pred is False:
    return fn2()
  from tensorflow.python.framework import tensor_util
  pred_const = tensor_util.constant_value(pred)
  if pred_const is not None:
    if pred_const:
      return fn1()
    else:
      return fn2()
  from tensorflow.python.ops import control_flow_ops
  return control_flow_ops.cond(pred, fn1, fn2, name=name)


def dot(a, b):
  """
  :param tf.Tensor a: shape [...da...,d]
  :param tf.Tensor b: shape [d,...db...]
  :return: tensor of shape [...da...,...db...]
  :rtype: tf.Tensor
  """
  with tf.name_scope("dot"):
    a_ndim = a.get_shape().ndims
    b_ndim = b.get_shape().ndims
    assert a_ndim is not None
    if a_ndim == 0:
      return tf.scalar_mul(a, b)
    assert b_ndim is not None
    if b_ndim == 0:
      return tf.scalar_mul(b, a)
    if a_ndim == b_ndim == 1:
      return tf.reduce_sum(a * b)
    d = tf.shape(b)[0]
    assert a_ndim >= 2 and b_ndim >= 2
    res_shape = None
    if a_ndim > 2 or b_ndim > 2:
      res_shape = [tf.shape(a)[i] for i in range(0, a_ndim - 1)] + [tf.shape(b)[i] for i in range(1, b_ndim)]
    if a_ndim > 2:
      a = tf.reshape(a, (-1, d))
    if b_ndim > 2:
      b = tf.reshape(b, (d, -1))
    res = tf.matmul(a, b)
    if a_ndim > 2 or b_ndim > 2:
      res = tf.reshape(res, res_shape)
    return res


class RHNCell(rnn_cell.RNNCell):
  """
  Recurrent Highway Layer.
  With optional dropout for recurrent state (fixed over all frames - some call this variational).

  References:
    https://github.com/julian121266/RecurrentHighwayNetworks/
    https://arxiv.org/abs/1607.03474
  """

  def __init__(self, num_units, is_training=None, depth=5, dropout=0.0, dropout_seed=None, transform_bias=None,
               batch_size=None):
    """
    :param int num_units:
    :param bool|tf.Tensor|None is_training:
    :param int depth:
    :param float dropout:
    :param int dropout_seed:
    :param float|None transform_bias:
    :param int|tf.Tensor|None batch_size:
    """
    super(RHNCell, self).__init__()
    self._num_units = num_units
    if is_training is None:
      is_training = True
    self.is_training = is_training
    self.depth = depth
    self.dropout = dropout
    if dropout_seed is None:
      dropout_seed = 13
    self.dropout_seed = dropout_seed
    self.transform_bias = transform_bias or 0.0
    self.batch_size = batch_size
    self._dropout_mask = None

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  @staticmethod
  def _linear(x, output_dim):
    """
    :param tf.Tensor x:
    :param int output_dim:
    :rtype: tf.Tensor
    """
    input_dim = x.get_shape().dims[-1].value
    assert input_dim is not None, "%r shape unknown" % (x,)
    with var_creation_scope():
      weights = tf.get_variable("W", shape=(input_dim, output_dim))
    x = dot(x, weights)
    return x

  def _get_dropout_mask(self):
    if self._dropout_mask is not None:
      return self._dropout_mask

    # Create the dropout masks outside the loop:
    with var_creation_scope():
      def get_mask():
        if self.batch_size is not None:
          batch_size = self.batch_size
        else:
          from TFNetworkLayer import LayerBase
          batch_size = LayerBase.get_recent_layer().get_batch_dim()
        keep_prob = 1.0 - self.dropout
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += tf.random_uniform((batch_size, self._num_units), seed=self.dropout_seed, dtype=tf.float32)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = tf.floor(random_tensor)
        return binary_tensor * (1.0 / keep_prob)
      self._dropout_mask = cond(self.is_training, get_mask, lambda: 1.0)
    return self._dropout_mask

  def _optional_dropout(self, state):
    if not self.dropout:
      return state
    if self.is_training is False:
      return state
    state *= self._get_dropout_mask()
    state.set_shape((None, self._num_units))
    return state

  def get_input_transformed(self, x, batch_dim=None):
    """
    :param tf.Tensor x: (time, batch, dim)
    :return: (time, batch, num_units * 2)
    :rtype: tf.Tensor
    """
    x = self._linear(x, self._num_units * 2)
    with var_creation_scope():
      bias = tf.get_variable(
        "b", shape=(self._num_units * 2,),
        initializer=tf.constant_initializer(
          [0.0] * self._num_units + [self.transform_bias] * self._num_units))
    x += bias
    return x

  def call(self, inputs, state):
    """
    :param tf.Tensor inputs:
    :param tf.Tensor state:
    :return: (output, state)
    :rtype: (tf.Tensor, tf.Tensor)
    """
    inputs.set_shape((None, self._num_units * 2))
    state.set_shape((None, self._num_units))

    # Carry-gate coupled with transform gate: C = 1 - T
    current_state = state
    for i in range(self.depth):
      current_state_masked = self._optional_dropout(current_state)
      with tf.variable_scope('depth_%i' % i):
        state_transformed = self._linear(current_state_masked, self._num_units * 2)
      if i == 0:
        state_transformed += inputs
      h, t = tf.split(state_transformed, 2, axis=-1)
      h = tf.tanh(h)
      t = tf.sigmoid(t)
      # Simplified equation for better numerical stability.
      # The current_state here should be without the dropout applied,
      # because dropout would divide by keep_prop, which can blow up the state.
      current_state += t * (h - current_state)

    return current_state, current_state


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

    default_var_initializer = xavier_initializer(seed=13)
    with tf.variable_scope(tf.get_variable_scope(), initializer=default_var_initializer) as scope:
      rhn = RHNCell(num_units=num_outputs, is_training=True, dropout=0.9, dropout_seed=1, batch_size=batch_size)
      state = rhn.zero_state(batch_size, tf.float32)
      x = tf.transpose(src_placeholder, [1, 0, 2])
      x = rhn.get_input_transformed(x)
      input_ta = tf.TensorArray(tf.float32, size=seq_len, element_shape=(None, num_outputs * 2))
      input_ta = input_ta.unstack(x)
      target_ta = tf.TensorArray(tf.float32, size=seq_len, element_shape=(None, num_outputs))
      target_ta = target_ta.unstack(tf.transpose(tgt_placeholder, [1, 0, 2]))
      loss_ta = tf.TensorArray(tf.float32, size=seq_len, element_shape=(None,))

      def loop_iter(i, state, loss_ta):
        output, state = rhn(inputs=input_ta.read(i), state=state)
        frame_loss = tf.reduce_mean(tf.squared_difference(target_ta.read(i), output), axis=1)
        assert frame_loss.get_shape().ndims == 1  # (batch,)
        # frame_loss = tf.Print(frame_loss, ["frame", i, "loss", tf.reduce_sum(frame_loss)])
        loss_ta = loss_ta.write(i, frame_loss)
        return i + 1, state, loss_ta

      loss = 0.0
      for i in range(seq_len):
        output, state = rhn(inputs=x[i], state=state)
        frame_loss = tf.reduce_mean(tf.squared_difference(tgt_placeholder[:, i], output), axis=1)
        # frame_loss = tf.Print(frame_loss, ['frame', i, 'loss', frame_loss, 'SE of', tgt_placeholder[:, i], output])
        assert frame_loss.get_shape().ndims == 1  # (batch,)
        loss += tf.reduce_sum(frame_loss)
      #loss = tf.reduce_sum(tf.get_variable("x", shape=(3, 3)) ** 2)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1, epsilon=1e-16, use_locking=False)
    minimize_op = optimizer.minimize(loss)
    check_op = tf.no_op()

    print('variables:')
    train_vars = (
            tf.trainable_variables() +
            tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
    print(train_vars)
    var_norms = [tf.nn.l2_loss(v) for v in train_vars]
    print('init vars')
    session.run(tf.global_variables_initializer())
    print('graph size:', session.graph_def.ByteSize())
    print('train')
    for s in range(10):
      loss_val, _, _ = session.run([loss, minimize_op, check_op], feed_dict=make_feed_dict())
      print("step %i, loss: %f" % (s, loss_val))
      # var_norm_vals = session.run(var_norms)
      # print('var norms:')
      # for (v, x) in zip(train_vars, var_norm_vals):
      #  print(' ', v, ':', x)


if __name__ == "__main__":
  better_exchook.install()
  better_exchook.replace_traceback_format_tb()
  faulthandler.enable()
  test()
