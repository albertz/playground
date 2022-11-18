


# https://github.com/google/learned_optimization/blob/main/learned_optimization/research/general_lopt/prefab.py
# https://github.com/google/learned_optimization/blob/main/learned_optimization/research/general_lopt/pretrained_optimizers.py
# https://github.com/google/learned_optimization/blob/main/learned_optimization/checkpoints.py

# requirements:
# absl-py==0.12.0
# numpy>=1.18
# jax>=0.2.6
# jaxlib>=0.1.68
# pytest
# tqdm>=4.62.3
# flax
# dm-haiku==0.0.5
# optax>=0.0.9
# tensorflow>=2.7.0
# tensorflow-datasets>=4.4.0
# tensorflow-metadata==1.5.0
# tensorflow-probability>=0.16.0
# tensorboard>=2.7.0
# gin-config>=0.5.0
# seqio>=0.0.7
# oryx

from __future__ import annotations
import collections
from concurrent import futures
import os
import time
from typing import Any, Callable, Mapping, Optional, TypeVar, Union
from typing import Any, Optional, Sequence, Tuple
import abc

from absl import logging
import tensorflow as tf
from flax import serialization
import flax.struct
import jax
import functools
import os
import uuid

import chex
import flax
import gin
import haiku as hk
from jax import lax
import jax.numpy as jnp
import numpy as onp


T = TypeVar("T")


_pretrain_root = 'gs://gresearch/learned_optimization/pretrained_lopts/'
_name = "aug12_continue_on_bigger_2xbs_200kstep_bigproblem_v2_5620"


@gin.configurable
def opt_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    extra_bindings=tuple([])
) -> Optimizer:
  """Load an optimizer from a checkpoint path, and gin config.
  Args:
    checkpoint_path: Path to `ParameterCheckpoint` saved to disk.
    config_path: Optional path to operative gin config for this checkpoint. If
      not provided, we look in the same folder for a config.gin
    extra_bindings: Optional extra gin bindings to load with this optimizer.
  Returns:
    Optimizer instance created from the learned optimizer + weights.
  """

  if config_path is None:
    config_path = "/".join(checkpoint_path.split("/")[:-1]) + "/config.gin"

  logging.info("Restoring configs from: %s", config_path)
  with gin.unlock_config():
    scope = f"opt_from_checkpoint__{str(uuid.uuid4()).replace('-', '_')}"
    with gin.config_scope(None):
      with gin.config_scope(scope):
        if config_path:
          with file_open(config_path, "rb") as f:
            content = bytes(f.read()).decode("utf-8")

          # gin writes out multi line sometimes, undo this.
          content = content.replace("\\\n", "")

          def maybe_add_scope(c):
            # filter out train as this overlaps with outer_training.
            if c.startswith("#"):
              return None
            if "=" in c:
              return scope + "/" + c
            return c

          bindings = [maybe_add_scope(c) for c in content.split("\n")]
          bindings = [b for b in bindings if b]
          bindings = bindings + [maybe_add_scope(c) for c in extra_bindings]

          logging.info("Parsing bindings")
          for b in bindings:
            logging.info(b)
            print(b)
          gin.parse_config(bindings, skip_unknown=True)

        configurable = gin.query_parameter(f"{scope}/run_train.lopt")
        if isinstance(configurable, gin.config._UnknownConfigurableReference):  # pylint: disable=protected-access
          raise ValueError("Gin couldn't find the learned optimizer in current"
                           " imports. Did you forget to import the module?")

        # with summary.summary_scope("opt_from_checkpoint"):
        lopt = configurable.configurable.wrapped()
        theta = lopt.init(jax.random.PRNGKey(0))
        logging.info(f"Restoring checkpoint {checkpoint_path}")  # pylint: disable=logging-fstring-interpolation
        ckpt = ParameterCheckpoint(theta, "", 0)
        ckpt = load_state(checkpoint_path, ckpt)
        opt = lopt.opt_fn(ckpt.params)
        return opt
        # wrapped = _GinScopeClass(opt, scope)
        # For now, just add the lopt to the returned class.
        # TODO(lmetz) change this api to return a more structured class?
        # wrapped.lopt = lopt
        # return wrapped  # type: ignore


def load_opt(path: str):
  lopt = HyperV2(
    lstm_hidden_size=512, param_inits=256, use_bugged_next_lstm_state=True)
  state = (lopt.init(jax.random.PRNGKey(0)), '', 0)
  theta, _, _ = load_state(path, state)
  return lopt.opt_fn(theta)


def load_state(path: str, state: T) -> T:
  """Load a pytree state directly from a file.
  Args:
    path: path to load pytree state from.
    state: pytree whose structure should match that of the stucture saved in the
      path. The values of this pytree are not used.
  Returns:
    The restored pytree matching the pytree structure of state.
  """
  logging.info("Restoring state %s", path)
  with file_open(path, "rb") as fp:
    state_new = serialization.from_bytes(state, fp.read())
  tree = jax.tree_util.tree_structure(state)
  leaves_new = jax.tree_util.tree_leaves(state_new)
  return jax.tree_util.tree_unflatten(tree, leaves_new)


def file_open(path: str, mode: str):
  """Open a file, returning a file object."""
  if _path_on_gcp(path):
    return tf.io.gfile.GFile(path, mode)
  return open(path, mode)


def _path_on_gcp(path: str) -> bool:
  prefixes = ["gs://"]
  return any([path.startswith(p) for p in prefixes])


def _fractional_tanh_embed(x):

  def one_freq(timescale):
    return jnp.tanh((x - (jnp.float32(timescale))) * 10)

  timescales = jnp.asarray([0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1],
                           dtype=jnp.float32)
  return jax.vmap(one_freq)(timescales)


def factored_dims(shape: Sequence[int]) -> Optional[Tuple[int, int]]:
  """Whether to use a factored second moment estimator.
  If there are not two dimensions of size >= min_dim_size_to_factor, then we
  do not factor. If we do factor the accumulator, then this function returns a
  tuple of the two largest axes to reduce over.
  Args:
    shape: a Shape
  Returns:
    None or a tuple of ints
  """
  if len(shape) < 2:
    return None
  sorted_dims = onp.argsort(shape)
  return int(sorted_dims[-2]), int(sorted_dims[-1])


def _clip_log_abs(v, scale=1.0):
  mag = jnp.log(1e-8 + jnp.abs(v * scale))
  return jnp.clip(mag, -5, 5) * 0.5


def _sorted_values(dd):
  return list(zip(*sorted(dd.items(), key=lambda x: x[0])))[1]


class BufferLossAccumulators:
  """Rolling accumulator for loss values."""

  def __init__(self):
    pass

  def init(self, num_steps):
    halflife = jnp.logspace(1, jnp.log10(num_steps), 10)
    decays = jnp.exp(-1. / halflife)
    return {
        "means":
            jnp.zeros((len(decays),), dtype=jnp.float32),
        "iteration":
            jnp.asarray(0, dtype=jnp.int32),
        "running_min":
            999999999999. * jnp.ones((len(decays),), dtype=jnp.float32),
        "decays":
            decays,
    }

  @functools.partial(jax.jit, static_argnums=(0,))
  def update(self, state, loss):
    """Update the state with a new loss."""
    # wana clip the losses so it doesn't go absolutely insane.
    jdecays = state["decays"]
    cor_mean = state["means"] / (1 - jdecays**(state["iteration"] + 1))
    approx_max = jnp.max(cor_mean)
    approx_max = jnp.where(state["iteration"] == 0, loss, approx_max)
    loss = jnp.minimum(jnp.abs(approx_max) * 2, loss)

    means = state["means"] * jdecays + loss * (1. - jdecays)

    cor_mean = means / (1 - jdecays**(state["iteration"] + 1))
    running_min = jnp.minimum(state["running_min"], cor_mean)

    return {
        "means": means,
        "iteration": state["iteration"] + 1,
        "running_min": running_min,
        "decays": state["decays"],
    }

  @functools.partial(jax.jit, static_argnums=(0,))
  def features(self, state):
    """Compute features to pass to NN from state."""
    jdecays = state["decays"]
    cor_mean = state["means"] / (1 - jdecays**(state["iteration"]))
    # longest running decay
    approx_max = cor_mean[1:]
    cor_mean = cor_mean[0:-1]
    running_min = state["running_min"][0:-1]

    den = jnp.maximum(1e-8, (approx_max - running_min))
    pre_center = (cor_mean - running_min) / den
    feature1 = (pre_center - 1.0)
    feature1 = jnp.clip(feature1, -1, 1)
    # first couple features are bad.
    return jnp.where(state["iteration"] <= 2, feature1 * 0, feature1)


@flax.struct.dataclass
class State:
  """Inner state of learned optimizer."""
  params: chex.ArrayTree
  rms_rolling: chex.ArrayTree
  mom_rolling: chex.ArrayTree
  fac_rolling: chex.ArrayTree
  iteration: jnp.ndarray
  state: chex.ArrayTree
  num_steps: jnp.ndarray
  lstm_hidden_state: chex.ArrayTree
  loss_buffer: chex.ArrayTree


def _safe_rsqrt(x):
  return lax.rsqrt(jnp.maximum(x, 1e-9))


def _second_moment_normalizer(x, axis, eps=1e-5):
  return x * jax.lax.rsqrt(eps +
                           jnp.mean(jnp.square(x), axis=axis, keepdims=True))


# pytree containing jax types
ModelState = Any
Params = Any
Gradient = Params
OptState = Any


@flax.struct.dataclass
class StatelessState:
  params: chex.ArrayTree
  state: chex.ArrayTree


class Optimizer(abc.ABC):
  """Baseclass for the Optimizer interface."""

  def get_params(self, state: OptState) -> Params:
    return state.params

  def get_state(self, state: OptState) -> ModelState:
    return state.state

  def get_params_state(self, state: OptState) -> Tuple[Params, ModelState]:
    return self.get_params(state), self.get_state(state)

  def init(self,
           params: Params,
           state: Optional[ModelState] = None,
           num_steps: Optional[int] = None,
           key: Optional[chex.PRNGKey] = None,
           **kwargs) -> OptState:
    raise NotImplementedError

  def set_params(self, state: OptState, params: Params) -> OptState:
    return state.replace(params=params)

  def update(
      self,
      opt_state: OptState,
      grad: Gradient,
      model_state: Optional[ModelState] = None,
      key: Optional[chex.PRNGKey] = None,
      **kwargs,
  ) -> OptState:
    raise NotImplementedError()

  @property
  def name(self) -> str:
    """Name of optimizer.
    This property is used when serializing results / baselines. This should
    lead with the class name, and follow with all parameters used to create
    the object. For example: "<ClassName>_<param1><value>_<param2><value>"
    """
    return "UnnamedOptimizer"


MetaParamOpt = collections.namedtuple("MetaParamOpt", ["init", "opt_fn"])

PRNGKey = jnp.ndarray
Params = Any
MetaParams = Any


class LearnedOptimizer(abc.ABC):
  """Base class for learned optimizers."""

  @abc.abstractmethod
  def init(self, key: PRNGKey) -> MetaParams:
    raise NotImplementedError()

  @abc.abstractmethod
  def opt_fn(self,
             theta: MetaParams,
             is_training: bool = False) -> Optimizer:
    raise NotImplementedError()

  @property
  def name(self):
    return None


@flax.struct.dataclass
class ParameterCheckpoint:
  """State that we write out to disk for using the optimizer."""
  params: MetaParams
  gen_id: str
  step: int


@gin.configurable
class HyperV2(LearnedOptimizer):
  """Experimental hypernetwork based learned optimizer."""

  def __init__(
      self,
      lstm_hidden_size=128,
      ff_hidden_size=4,
      ff_hidden_layers=2,
      initial_momentum_decays=(0.9, 0.99, 0.999),
      initial_rms_decays=(0.999,),
      initial_adafactor_decays=(0.9, 0.99, 0.999),
      param_inits=64,
      mix_layers=True,
      exp_mult=0.001,
      step_mult=0.001,
      validation_mode=False,
      with_validation_feature_dim=False,

      # ablation flags.
      with_g=True,
      with_m=True,
      with_m_feat=True,
      with_rms=True,
      with_rms_feat=True,
      with_rms_norm_g=True,
      with_rsqrt_rms=True,
      with_p=True,
      with_fac_norm_g=True,
      with_fac_rms=True,
      with_fac_rsqrt=True,
      with_grad_clip_feat=True,
      with_fac_mom_mult=True,
      with_rms_only_norm_g=True,
      adafactor_accumulator=True,
      param_scale_mult=True,
      use_bugged_next_lstm_state=False,
      use_bugged_loss_features=True,
      precondition_output=False,
      reparam_decay=10.,
      rnn_state_decay=0.0,

      # more summaries
      summarize_each_layer=False,
      summarize_all_control=False,

      # Modify the lopt to probe behavior
      constant_loss=False,
      clip_param_scale_amount=None,
  ):
    """Initializer.
    Args:
      lstm_hidden_size: size of the per tensor LSTM.
      ff_hidden_size: hidden size of the per-parameter MLP.
      ff_hidden_layers: number of layers in per-parameter mlp.
      initial_momentum_decays: The values of momentum accumulators to use
      initial_rms_decays: The values of the second moment gradient accumulators
        to use.
      initial_adafactor_decays: The values of the adafactor style accumulators
        to use.
      param_inits: Number of parameter inputs with which to linearly interpolate
        to create each per-parameter MLP.
      exp_mult: setting to rescale output of lopt
      step_mult: setting to rescale output of lopt  validation model: optionally
        add an additional input to LSTM to denote targeting train or valid loss.
      with_validation_feature: Set the above feature on or off.   <many ablation
        flags>
    """
    # TODO(lmetz): Remove reparam_decay -- is not being used.
    super().__init__()
    self.lstm_hidden_size = lstm_hidden_size
    self.ff_hidden_size = ff_hidden_size
    self.ff_hidden_layers = ff_hidden_layers
    self.initial_momentum_decays = initial_momentum_decays
    self.initial_rms_decays = initial_rms_decays
    self.initial_adafactor_decays = initial_adafactor_decays
    self.param_inits = param_inits
    self.mix_layers = mix_layers
    self.with_g = with_g
    self.with_m = with_m
    self.with_m_feat = with_m_feat
    self.with_rms = with_rms
    self.with_rms_feat = with_rms_feat
    self.with_rms_norm_g = with_rms_norm_g
    self.with_rsqrt_rms = with_rsqrt_rms
    self.with_p = with_p
    self.with_fac_norm_g = with_fac_norm_g
    self.with_fac_rms = with_fac_rms
    self.with_fac_rsqrt = with_fac_rsqrt
    self.with_grad_clip_feat = with_grad_clip_feat
    self.with_fac_mom_mult = with_fac_mom_mult
    self.with_rms_only_norm_g = with_rms_only_norm_g
    self.adafactor_accumulator = adafactor_accumulator
    self.param_scale_mult = param_scale_mult
    self.exp_mult = exp_mult
    self.step_mult = step_mult
    self.use_bugged_next_lstm_state = use_bugged_next_lstm_state
    self.use_bugged_loss_features = use_bugged_loss_features
    self.summarize_each_layer = summarize_each_layer
    self.precondition_output = precondition_output
    self.reparam_decay = reparam_decay
    self.rnn_state_decay = rnn_state_decay
    self.with_validation_feature_dim = with_validation_feature_dim
    self.validation_mode = validation_mode
    self.constant_loss = constant_loss
    self.summarize_all_control = summarize_all_control
    self.clip_param_scale_amount = clip_param_scale_amount

    if self.use_bugged_loss_features:
      logging.warning("You are using bugged loss features! If you are using a"
                      "pretrained optimizer, otherwise this is an error.")

    logging.info(
        f"Validation mode: {self.validation_mode} (with valid feature dim: {with_validation_feature_dim})"
    )

    self.rnn_to_controls = hk.without_apply_rng(
        hk.transform(lambda x: hk.Linear(  # pylint: disable=unnecessary-lambda, g-long-lambda
            param_inits,
            name="rnn_to_controls",
            w_init=hk.initializers.Constant(0.),
        )(x)))

    self.lstm_fn = lambda: hk.LSTM(lstm_hidden_size, name="rnn")

    self.rnn = hk.without_apply_rng(hk.transform(self._rnn_forward))
    self.ff_mod = hk.transform(self._ff_mod)
    self.buffer_loss_fns = BufferLossAccumulators()

  def _decay_to_param(self, x):
    return jnp.log(1 - x) / self.reparam_decay

  def _param_to_decay(self, x):
    return 1 - jnp.exp(x * self.reparam_decay)

  def accumulators_for_decays(self,
                              mom_param=None,
                              rms_param=None,
                              adafactor_param=None):
    if mom_param is None:
      mom_decay = jnp.asarray(self.initial_momentum_decays)
    else:
      mom_decay = self._param_to_decay(
          self._decay_to_param(jnp.asarray(self.initial_momentum_decays)) +
          mom_param)
    if rms_param is None:
      rms_decay = jnp.asarray(self.initial_rms_decays)
    else:
      rms_decay = self._param_to_decay(
          self._decay_to_param(jnp.asarray(self.initial_rms_decays)) +
          rms_param)

    if adafactor_param is None:
      adafactor_decay = jnp.asarray(self.initial_adafactor_decays)
    else:
      adafactor_decay = self._param_to_decay(
          self._decay_to_param(jnp.asarray(self.initial_adafactor_decays)) +
          adafactor_param)

    mom_roll = vec_rolling_mom(mom_decay)
    rms_roll = vec_rolling_rms(rms_decay)
    fac_vec_roll = vec_factored_rolling(adafactor_decay)
    return mom_roll, rms_roll, fac_vec_roll

  def _rnn_forward(self, x, state):
    if self.mix_layers:
      mix_layer = hk.Linear(self.lstm_hidden_size)(x)
      mix_layer = jax.nn.relu(mix_layer)
      mix_layer = hk.Linear(self.lstm_hidden_size)(x)
      mix_layer = jax.nn.relu(mix_layer)
      v = jnp.max(mix_layer, axis=0, keepdims=True)
      x = hk.Linear(self.lstm_hidden_size)(x) + v

    rnn_out, state = self.lstm_fn()(x, state)

    controls = hk.Linear(
        self.param_inits,
        name="rnn_to_controls",
    )(
        rnn_out)
    lr_mult = jnp.squeeze(hk.Linear(1, name="step_size")(rnn_out), -1)
    return controls, lr_mult, state

  def _ff_mod(self, global_feat, extra_step_mult, p, g, m, rms, fac_g,
              fac_vec_col, fac_vec_row, fac_vec_v):
    # this doesn't work with scalar parameters, so instead lets just reshape.
    if len(p.shape) == 0:  # pylint: disable=g-explicit-length-test
      p = jnp.expand_dims(p, 0)
      g = jnp.expand_dims(g, 0)
      m = jnp.expand_dims(m, 0)
      rms = jnp.expand_dims(rms, 0)
      fac_g = jnp.expand_dims(fac_g, 0)
      fac_vec_v = jnp.expand_dims(fac_vec_v, 0)
      fac_vec_col = jnp.expand_dims(fac_vec_col, 0)
      fac_vec_row = jnp.expand_dims(fac_vec_row, 0)
      did_reshape = True
    else:
      did_reshape = False
    inps = []

    if self.with_g:
      batch_g = jnp.expand_dims(g, axis=-1)
      inps.append(batch_g)

    if self.with_grad_clip_feat:
      clip_batch_g = jnp.expand_dims(jnp.clip(g, -0.1, 0.1), axis=-1)
      inps.append(clip_batch_g)

    if self.with_p:
      batch_p = jnp.expand_dims(p, axis=-1)
      inps.append(batch_p)

    # grads and params features
    if self.with_m and self.with_m_feat:
      inps.append(m)

    if self.with_rms and self.with_rms_feat:
      inps.append(rms)

    if self.with_rms_norm_g or self.with_rsqrt_rms and self.with_rms_only_norm_g:
      rsqrt = lax.rsqrt(rms + 1e-6)

    if self.with_rms_norm_g:
      norm_g = m * rsqrt
      inps.append(norm_g)

    if self.with_rsqrt_rms:
      inps.append(rsqrt)

    if self.with_fac_norm_g:
      inps.append(fac_g)

    if self.with_rms_only_norm_g:
      rms_norm_g = jnp.expand_dims(g, axis=-1) * rsqrt
      inps.append(rms_norm_g)

    if self.adafactor_accumulator:
      factored_dim = factored_dims(g.shape)
      if factored_dim is not None:
        d1, d0 = factored_dim

        # add 2 dims. 1 for batch of decay. one because low rank
        to_tile = [1] * (1 + len(g.shape))
        # offset here because of vectorization over decays.
        to_tile[d0] = g.shape[d0]
        row_feat = jnp.tile(jnp.expand_dims(fac_vec_row, axis=d0), to_tile)

        to_tile = [1] * (1 + len(g.shape))
        to_tile[d1] = g.shape[d1]
        col_feat = jnp.tile(jnp.expand_dims(fac_vec_col, axis=d1), to_tile)
        # goal: <feat, n1, n2>

        if self.with_fac_rms:
          inps.append(row_feat)
          inps.append(col_feat)

        if self.with_fac_rsqrt:
          inps.append(lax.rsqrt(row_feat + 1e-8))
          inps.append(lax.rsqrt(col_feat + 1e-8))

        reduced_d1 = d1 - 1 if d1 > d0 else d1
        # reduced_d1:1, d0:0, d1:1, g.shape:(784, 32),
        # fac_vec_row.shape:(6, 784), fac_vec_col.shape:(6, 32)
        row_col_mean = jnp.mean(fac_vec_row, axis=reduced_d1, keepdims=True)

        row_factor = _safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9))
        col_factor = _safe_rsqrt(fac_vec_col)

        if self.with_fac_mom_mult:
          fac_mom_mult = (
              m * jnp.expand_dims(row_factor, axis=d0) *
              jnp.expand_dims(col_factor, axis=d1))
          inps.append(fac_mom_mult)

      else:
        if self.with_fac_rms:
          inps.append(fac_vec_v)
          inps.append(fac_vec_v)

        if self.with_fac_rsqrt:
          inps.append(lax.rsqrt(fac_vec_v + 1e-8))
          inps.append(lax.rsqrt(fac_vec_v + 1e-8))

        if self.with_fac_mom_mult:
          fac_mom_mult = m * (fac_vec_v)**-0.5
          inps.append(fac_mom_mult)

    # Inline / unrolled MLP implementation. We found this to be faster than
    # doing the more standard implementation with matmuls.

    # First, we build the weights of the NN
    last_size = sum([i.shape[-1] for i in inps])

    weights = []
    biases = []

    for wi, w in enumerate([self.ff_hidden_size] * (self.ff_hidden_layers) +
                           [3]):
      stddev = 1. / onp.sqrt(last_size)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)
      if wi == 0:
        w1 = []
        for ii, i in enumerate(inps):
          w1.append(
              hk.get_parameter(
                  f"w{wi}__{ii}",
                  shape=(i.shape[-1], w),
                  dtype=jnp.float32,
                  init=w_init))
        weights.append(w1)
      else:
        weights.append(
            hk.get_parameter(
                f"w{wi}", shape=(last_size, w), dtype=jnp.float32, init=w_init))

      biases.append(
          hk.get_parameter(
              f"b{wi}", shape=(w,), dtype=jnp.float32, init=jnp.zeros))
      last_size = w

    axis = list(range(len(p.shape)))

    inp_stack = [_second_moment_normalizer(i, axis=axis) for i in inps]

    # Next we apply our MLP.
    o = inp_stack
    for wi, (w, b) in enumerate(zip(weights, biases)):
      if wi == 0:
        o_tmp = jnp.zeros(o[0].shape[:-1] + w[0].shape[1:])
        for oi, oo in enumerate(o):
          o_tmp = o_tmp + oo @ w[oi]
      else:
        o_tmp = o @ w  # pytype: disable=unsupported-operands

      o = o_tmp + jnp.broadcast_to(b,
                                   list(o_tmp.shape[0:-1]) + [o_tmp.shape[-1]])

      if wi != len(weights) - 1:
        o = jax.nn.relu(o)

    # extract outputs from MLP to construct a step.
    direction = o[..., 0]
    magnitude_param = o[..., 1]

    mag_param = jnp.exp(magnitude_param * self.exp_mult)
    param_scale = jnp.sqrt(jnp.mean(jnp.square(p)) + 1e-9)
    summary.summary("hyperv2/param_scale", param_scale)

    if self.clip_param_scale_amount is not None:
      max_scale = self.clip_param_scale_amount * onp.sqrt(onp.prod(p.shape))
      param_scale = jnp.minimum(param_scale,
                                jnp.asarray(max_scale, dtype=jnp.float32))
      summary.summary("hyperv2/post_param_scale", param_scale)

    avg_step_size = jnp.mean(
        jnp.abs(direction * mag_param * self.step_mult * extra_step_mult))
    summary.summary("hyperv2/no_parammag_mult_avg_step_size", avg_step_size)

    if self.param_scale_mult:
      step = direction * (param_scale * mag_param) * self.step_mult
    else:
      step = direction * mag_param * self.step_mult
    step = extra_step_mult * step

    avg_step_size = jnp.mean(jnp.abs(step))
    summary.summary("hyperv2/pre_precondition_avg_step_size", avg_step_size)

    step = step.reshape(p.shape)
    if self.precondition_output:
      # extract out the last rms.
      norms = jax.tree_util.tree_map(lambda x: x[..., -1], rms)
      assert norms.shape == step.shape
      step = step * lax.rsqrt(norms + 1e-6)

    avg_step_size = jnp.mean(jnp.abs(step))
    summary.summary("hyperv2/avg_step_size", avg_step_size)
    summary.summary("hyperv2/extra_step_mult", extra_step_mult)

    new_p = p - step
    if did_reshape:
      new_p = jnp.squeeze(new_p, 0)

    return new_p

  def lstm_features_for_tensor(self, p, g, m, rms, summary_prefix,
                               fraction_trained, loss_features):
    norm_mult = jax.lax.rsqrt(jnp.maximum(1e-9, jnp.mean(p**2)))
    g = g * norm_mult
    p = p * norm_mult
    m = m * norm_mult
    rms = rms * norm_mult

    inputs = {}

    fraction_left = _fractional_tanh_embed(fraction_trained)
    inputs["fraction_left"] = fraction_left
    inputs["loss_features"] = loss_features

    leading_axis = list(range(0, len(p.shape)))
    mean_m = jnp.mean(m, axis=leading_axis, keepdims=True)
    var_m = jnp.mean(jnp.square(m - mean_m), axis=leading_axis)
    inputs["var_m"] = _clip_log_abs(var_m, scale=10.)

    mean_rms = jnp.mean(rms, axis=leading_axis, keepdims=True)
    var_rms = jnp.mean(jnp.square(rms - mean_m), axis=leading_axis)
    inputs["mean_rms"] = _clip_log_abs(
        jnp.reshape(mean_rms, [mean_rms.shape[-1]]), scale=10.)
    inputs["var_rms"] = _clip_log_abs(var_rms, scale=10.)

    # rank
    n_rank = onp.sum(onp.asarray(p.shape) > 1)
    inputs["rank"] = hk.one_hot(n_rank, 5)

    # TODO(lmetz) turn this off when we want more speed???
    for k, v in inputs.items():
      if len(v.shape) > 0:  # pylint: disable=g-explicit-length-test
        for vi, vv in enumerate(v):
          summary.summary(
              f"per_tensor_feat/{k}__{vi}", vv, aggregation="sample")
      else:
        summary.summary(f"per_tensor_feat/{k}", v, aggregation="sample")

    if self.summarize_each_layer:
      for k, v in inputs.items():
        if len(v.shape) > 0:  # pylint: disable=g-explicit-length-test
          for vi, vv in enumerate(v):
            summary.summary(
                f"per_tensor_feat/{summary_prefix}/{k}__{vi}",
                vv,
                aggregation="sample")
        else:
          summary.summary(
              f"per_tensor_feat/{summary_prefix}/{k}", v, aggregation="sample")

    values = _sorted_values(inputs)
    values = [v if len(v.shape) == 1 else jnp.expand_dims(v, 0) for v in values]

    # add the validation features at the end of the feature vector to make it
    # easier to do surgery into it.
    if self.with_validation_feature_dim:
      values.append(jnp.ones([1], dtype=jnp.float32) * self.validation_mode)

    return jnp.concatenate(values, axis=0)

  def init(self, key) -> MetaParams:
    r = 10
    c = 10
    p = jnp.ones([r, c])
    g = jnp.ones([r, c])

    m = jnp.ones([r, c, len(self.initial_momentum_decays)])
    rms = jnp.ones([r, c, len(self.initial_rms_decays)])
    fac_g = jnp.ones([r, c, len(self.initial_adafactor_decays)])
    fac_vec_row = jnp.ones([r, len(self.initial_adafactor_decays)])
    fac_vec_col = jnp.ones([c, len(self.initial_adafactor_decays)])
    fac_vec_v = jnp.ones([len(self.initial_adafactor_decays)])

    def ffmod_init(key):
      global_features = {
          "iterations": 0,
          "num_steps": 10,
      }
      mod_theta = self.ff_mod.init(key, global_features, 1.0, p, g, m, rms,
                                   fac_g, fac_vec_col, fac_vec_row, fac_vec_v)
      return mod_theta

    key1, key = jax.random.split(key)
    per_param_thetas = jax.vmap(ffmod_init)(
        jax.random.split(key1, self.param_inits))

    lstm_inital_state = hk.transform(
        lambda: self.lstm_fn().initial_state(1))[1](None, key1)

    loss_features = self.buffer_loss_fns.features(self.buffer_loss_fns.init(10))

    # figure out how may m and rms features there are by getting an opt state.
    output_shape = jax.eval_shape(
        self.lstm_features_for_tensor,
        p,
        p,
        m,
        rms,
        0,  # no prefix!,
        fraction_trained=1.0,
        loss_features=loss_features)

    assert len(output_shape.shape) == 1

    rnn_input_features = output_shape.shape[0]

    key1, key = jax.random.split(key)
    return {
        "lstm_init_state":
            lstm_inital_state,
        "rnn_params":
            self.rnn.init(key1, jnp.zeros([1, rnn_input_features]),
                          lstm_inital_state),
        "ff_mod_stack":
            per_param_thetas,
    }

  def opt_fn(self, theta, is_training=True) -> Optimizer:
    parent = self

    class _Opt(Optimizer):
      """Inner optimizer."""

      def __init__(self, theta):
        super().__init__()
        self.theta = theta

      @functools.partial(jax.jit, static_argnums=(0,))
      def init(self,
               params: Any,
               model_state=None,
               num_steps=None,
               key=None) -> State:
        mom_roll, rms_roll, adafac_roll = parent.accumulators_for_decays()
        if parent.use_bugged_loss_features:
          loss_buffer = parent.buffer_loss_fns.init(10)
        else:
          loss_buffer = parent.buffer_loss_fns.init(num_steps)

        n_states = len(jax.tree_util.tree_leaves(params))
        lstm_hidden_state = jax.tree_util.tree_map(
            lambda x: jnp.tile(x, [n_states] + [1] * len(x.shape[1:])),
            theta["lstm_init_state"])

        return State(
            params=params,
            state=model_state,
            rms_rolling=rms_roll.init(params),
            mom_rolling=mom_roll.init(params),
            fac_rolling=adafac_roll.init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            num_steps=jnp.asarray(num_steps, dtype=jnp.int32),
            lstm_hidden_state=lstm_hidden_state,
            loss_buffer=loss_buffer)

      @functools.partial(jax.jit, static_argnums=(0,))
      def update(self,
                 opt_state,
                 grads,
                 loss=None,
                 model_state=None,
                 is_valid=False,
                 key=None) -> State:
        if parent.constant_loss:
          loss = 1.0
        assert loss is not None
        summary.summary("validation_mode", parent.validation_mode)

        next_loss_buffer = parent.buffer_loss_fns.update(
            opt_state.loss_buffer, loss)
        to_lstm_from_loss = parent.buffer_loss_fns.features(next_loss_buffer)

        grads = jax.tree_util.tree_map(lambda x: jnp.clip(x, -1000., 1000.),
                                       grads)
        # Run the LSTM to get params for ff.

        fraction_trained = opt_state.iteration / jnp.asarray(
            opt_state.num_steps, dtype=jnp.float32)
        ff = functools.partial(
            parent.lstm_features_for_tensor,
            fraction_trained=fraction_trained,
            loss_features=to_lstm_from_loss)

        m = opt_state.mom_rolling.m
        rms = opt_state.rms_rolling.rms
        if parent.summarize_each_layer:
          summary_prefix = map_named(lambda k, v: k,
                                                opt_state.params)
        else:
          summary_prefix = jax.tree_util.tree_map(lambda x: "None",
                                                  opt_state.params)

        rnn_inputs = jax.tree_util.tree_map(ff, opt_state.params, grads, m, rms,
                                            summary_prefix)

        stack = jnp.asarray(jax.tree_util.tree_leaves(rnn_inputs))

        lstm_hidden_state = opt_state.lstm_hidden_state

        control_params, lr_mult, next_lstm_hidden_state = parent.rnn.apply(
            theta["rnn_params"], stack, lstm_hidden_state)

        # This bug was accidentally introduced, never the less we would like
        # to be able to make use of old checkpoints which don't propogate
        # lstm state forward. As such we leave a setting here.
        if not parent.use_bugged_next_lstm_state:
          lstm_hidden_state = next_lstm_hidden_state

        if parent.rnn_state_decay > 0.0:
          lstm_hidden_state = tree_mul(
              lstm_hidden_state, (1.0 - parent.rnn_state_decay))

        # one per param.
        control_params = [d for d in control_params]
        if parent.summarize_all_control:
          for pi, p in enumerate(control_params):
            summary.summary(f"control_param/{pi}", p, "tensor")
        struct = jax.tree_util.tree_structure(grads)

        control_params = struct.unflatten(control_params)
        lr_mult = struct.unflatten([lr for lr in lr_mult])

        # Run the FF
        mom_roll, rms_roll, adafac_roll = parent.accumulators_for_decays()
        next_mom_rolling = mom_roll.update(opt_state.mom_rolling, grads)
        next_rms_rolling = rms_roll.update(opt_state.rms_rolling, grads)
        next_adafac_rolling, fac_g = adafac_roll.update(opt_state.fac_rolling,
                                                        grads)

        global_features = {
            "iterations": opt_state.iteration,
            "num_steps": opt_state.num_steps,
        }

        def apply_one(control_param, key, lr_mult, p, g, m, rms, fac_g, v_col,
                      v_row, v):

          def interpolate_theta(ff_p):
            target = [ff_p.shape[0]] + [1] * (len(ff_p.shape) - 1)
            c = jnp.reshape(control_param, target)
            return 100. * jnp.mean(ff_p * c, axis=0)

          ff_param = jax.tree_util.tree_map(interpolate_theta,
                                            theta["ff_mod_stack"])
          next_p = parent.ff_mod.apply(
              ff_param,
              key,
              global_features,
              lr_mult,
              p,
              g,
              m=m,
              rms=rms,
              fac_g=fac_g,
              fac_vec_col=v_col,
              fac_vec_row=v_row,
              fac_vec_v=v)
          return next_p

        l, struct = jax.tree_util.tree_flatten(control_params)
        key, key1 = jax.random.split(key)
        keys = struct.unflatten([k for k in jax.random.split(key1, len(l))])
        next_params = jax.tree_util.tree_map(
            apply_one, control_params, keys, lr_mult, opt_state.params, grads,
            next_mom_rolling.m, next_rms_rolling.rms, fac_g,
            next_adafac_rolling.v_col, next_adafac_rolling.v_row,
            next_adafac_rolling.v_diag)

        ss = State(
            params=next_params,
            state=model_state,
            mom_rolling=next_mom_rolling,
            rms_rolling=next_rms_rolling,
            fac_rolling=next_adafac_rolling,
            iteration=opt_state.iteration + 1,
            num_steps=opt_state.num_steps,
            lstm_hidden_state=lstm_hidden_state,
            loss_buffer=next_loss_buffer,
        )
        return match_type(ss, opt_state)

    return _Opt(theta)


MomAccumulator = collections.namedtuple("MomAccumulator", ["m", "t"])
RMSAccumulator = collections.namedtuple("RMSAccumulator", ["rms", "t"])
_InitUpdate = collections.namedtuple("_InitUpdate", ["init", "update"])


def rolling_mom(decay: float) -> _InitUpdate:
  """Acculator to keep track of momentum."""

  def init_fn(p: Any) -> MomAccumulator:
    return MomAccumulator(
        m=jax.tree_util.tree_map(jnp.zeros_like, p),
        t=jnp.asarray(0, dtype=jnp.int32))

  def update_fn(state: MomAccumulator, grad: Any) -> MomAccumulator:
    m = jax.tree_util.tree_map(lambda a, b: decay * a + (1 - decay) * b,
                               state.m, grad)
    return MomAccumulator(m=m, t=state.t + 1)

  return _InitUpdate(init_fn, update_fn)


def rolling_rms(decay: float) -> _InitUpdate:
  """Acculator to keep track of second moment accumulators."""

  def init_fn(p: Any) -> RMSAccumulator:
    return RMSAccumulator(
        rms=jax.tree_util.tree_map(jnp.zeros_like, p),
        t=jnp.asarray(0, dtype=jnp.int32))

  def update_fn(state: RMSAccumulator, grad: Any) -> RMSAccumulator:
    clip_decay = jnp.clip(decay, 0.0, 1.0)
    rms = jax.tree_util.tree_map(
        lambda a, b: clip_decay * a + (1 - clip_decay) * (b * b), state.rms,
        grad)
    return RMSAccumulator(rms=rms, t=state.t + 1)

  return _InitUpdate(init_fn, update_fn)


def _vmap_accumulator(accumulator: Callable[[float], _InitUpdate],
                      decays: jnp.ndarray) -> _InitUpdate:
  """Helper function that vmaps an accumulator fn to run on multiple decays."""

  def init_fn(p):
    return jax.vmap(lambda d: accumulator(d).init(p), out_axes=-1)(decays)

  def update(state, grads):
    return jax.vmap(
        lambda s, d: accumulator(d).update(s, grads), in_axes=-1,
        out_axes=-1)(state, decays)

  return _InitUpdate(init=init_fn, update=update)


def vec_rolling_mom(decays: jnp.ndarray) -> _InitUpdate:
  """Vectorized accumulator to keep track of multiple momentum decays."""
  return _vmap_accumulator(rolling_mom, decays)


def vec_rolling_rms(decays: jnp.ndarray) -> _InitUpdate:
  """Vectorized accumulator to keep track of multiple second moment decays."""
  return _vmap_accumulator(rolling_rms, decays)


def safe_rsqrt(x: jnp.ndarray) -> jnp.ndarray:
  return jax.lax.rsqrt(jnp.maximum(x, 1e-9))


@flax.struct.dataclass
class FactoredAccum:
  v_col: jnp.ndarray
  v_row: jnp.ndarray
  v_diag: jnp.ndarray


def factored_rolling(decay_rate: float, epsilon: float = 1e-30) -> _InitUpdate:
  """Gradient statistics accumulator based on factored gradients.
  This calculates accumulators similar to that of AdaFactor.
  Args:
    decay_rate: accumulator decay
    epsilon: numerical stability
  Returns:
    functions to initialize and update the adafactor style accumulators.
  """

  def init_fn(params: Any) -> FactoredAccum:

    def _init_one(param):
      shape = param.shape
      f_dims = factored_dims(shape)
      # If factored, set v_row, v_col. Otherwise set v_full
      if f_dims is not None:
        d1, d0 = f_dims
        vr_shape = onp.delete(shape, d0)
        vc_shape = onp.delete(shape, d1)
        v_row = jnp.zeros(vr_shape, dtype=jnp.float32)
        v_col = jnp.zeros(vc_shape, dtype=jnp.float32)
        return v_row, v_col, jnp.asarray([], dtype=jnp.float32)

      else:
        v = jnp.zeros(param.shape, dtype=jnp.float32)
        return jnp.asarray([],
                           dtype=jnp.float32), jnp.asarray([],
                                                           dtype=jnp.float32), v

    leaves, tree = jax.tree_util.tree_flatten(params)
    v_rows, v_cols, v_fulls = zip(*[_init_one(l) for l in leaves])
    return FactoredAccum(
        v_row=jax.tree_util.tree_unflatten(tree, v_rows),
        v_col=jax.tree_util.tree_unflatten(tree, v_cols),
        v_diag=jax.tree_util.tree_unflatten(tree, v_fulls))

  def update_fn(state: FactoredAccum, grad: Any) -> Tuple[FactoredAccum, Any]:

    def update_one(v_col: Any, v_row: Any, v_full: Any,
                   g: Any) -> Tuple[Any, Any, Any, Any]:
      clip_decay_rate = jnp.clip(decay_rate, 0.0, 1.0)
      mixing_rate = 1.0 - clip_decay_rate

      grad_sqr = g * g + epsilon
      f_dims = factored_dims(g.shape)

      if f_dims is not None:
        # precondition with factored dimensions.
        d1, d0 = f_dims
        new_v_row = (
            clip_decay_rate * v_row + mixing_rate * jnp.mean(grad_sqr, axis=d0))
        new_v_col = (
            clip_decay_rate * v_col + mixing_rate * jnp.mean(grad_sqr, axis=d1))

        reduced_d1 = d1 - 1 if d1 > d0 else d1
        row_col_mean = jnp.mean(new_v_row, axis=reduced_d1, keepdims=True)

        row_factor = safe_rsqrt(new_v_row / (row_col_mean + 1e-9))
        col_factor = safe_rsqrt(new_v_col)
        y = (
            g * jnp.expand_dims(row_factor, axis=d0) *
            jnp.expand_dims(col_factor, axis=d1))
        return new_v_col, new_v_row, jnp.asarray([], jnp.float32), y

      else:
        # otherwise precondition with diagonal style preconditioner
        new_v = clip_decay_rate * v_full + mixing_rate * grad_sqr
        y = g * safe_rsqrt(new_v + 1e-9)
        return jnp.asarray([], jnp.float32), jnp.asarray([],
                                                         jnp.float32), new_v, y

    f_v_col, tree = jax.tree_util.tree_flatten(state.v_col)
    f_v_row = jax.tree_util.tree_leaves(state.v_row)
    f_v = jax.tree_util.tree_leaves(state.v_diag)
    f_g = jax.tree_util.tree_leaves(grad)
    assert len(f_g) == len(f_v_col)
    assert len(f_g) == len(f_v)
    assert len(f_g) == len(f_v_row)
    f_v_col, f_v_row, f_v, outs = zip(
        *[update_one(*args) for args in zip(f_v_col, f_v_row, f_v, f_g)])

    next_state = FactoredAccum(
        v_col=jax.tree_util.tree_unflatten(tree, f_v_col),
        v_row=jax.tree_util.tree_unflatten(tree, f_v_row),
        v_diag=jax.tree_util.tree_unflatten(tree, f_v))

    return next_state, jax.tree_util.tree_unflatten(tree, outs)

  return _InitUpdate(init_fn, update_fn)


def vec_factored_rolling(decays: jnp.ndarray) -> _InitUpdate:
  """Vectorized accumulator to keep track of factored accumulators."""
  return _vmap_accumulator(factored_rolling, decays)


class summary:
  # from learned_optimization import summary
  @classmethod
  def summary(cls, name, val, *args, **kwargs) -> jnp.ndarray:
    """Create a summary.
    This is for use exclusivly inside jax functions.
    Args:
      name: name of summary
      val: scalar value to write
      aggregation: How to aggregate duplicate names. Currently supported are mean,
        sample, and collect.
    Returns:
      val which has the summary in the computation graph
    """
    if not isinstance(name, str):
      raise ValueError("First argument must be a string. The order of arguments "
                       " was changed Q1 2022.")
    return val  # dummy




def _is_scalar(x):
  try:
    jnp.asarray(x)
    return True
  except Exception:  # pylint: disable=broad-except
    return False


@jax.jit
def tree_add(treea, treeb):
  return jax.tree_util.tree_map(lambda a, b: a + b, treea, treeb)


@jax.jit
def tree_sub(treea, scalar_or_treeb):
  if _is_scalar(scalar_or_treeb):
    return jax.tree_util.tree_map(lambda a: a - scalar_or_treeb, treea)
  else:
    return jax.tree_util.tree_map(lambda a, b: a - b, treea, scalar_or_treeb)


@jax.jit
def tree_mean_abs(val):
  num_entry = sum(
      map(lambda x: onp.prod(x.shape), jax.tree_util.tree_leaves(val)))
  sum_abs = sum(
      map(lambda x: jnp.sum(jnp.abs(x)), jax.tree_util.tree_leaves(val)))
  return sum_abs / num_entry


@jax.jit
def tree_mean(val):
  num_entry = sum(
      map(lambda x: onp.prod(x.shape), jax.tree_util.tree_leaves(val)))
  return sum(map(jnp.sum, jax.tree_util.tree_leaves(val))) / num_entry


@jax.jit
def tree_norm(val):
  sum_squared = sum(
      map(lambda x: jnp.sum(jnp.square(x)), jax.tree_util.tree_leaves(val)))
  return jnp.sqrt(sum_squared)


@jax.jit
def tree_div(treea, scalar_or_treeb):
  if _is_scalar(scalar_or_treeb):
    return jax.tree_util.tree_map(lambda a: a / scalar_or_treeb, treea)
  else:
    return jax.tree_util.tree_map(lambda a, b: a / b, treea, scalar_or_treeb)


@jax.jit
def tree_mul(treea, scalar_or_treeb):
  if _is_scalar(scalar_or_treeb):
    return jax.tree_util.tree_map(lambda a: a * scalar_or_treeb, treea)
  else:
    return jax.tree_util.tree_map(lambda a, b: a * b, treea, scalar_or_treeb)


@jax.jit
def tree_dot(treea, treeb):
  mult = jax.tree_util.tree_map(lambda a, b: a * b, treea, treeb)
  return sum(map(jnp.sum, jax.tree_util.tree_leaves(mult)))


def tree_zip_onp(xs):
  xs = list(xs)
  _, tree_def = jax.tree_util.tree_flatten(xs[0])
  ys = map(onp.asarray,
           zip(*map(lambda x: jax.tree_util.tree_flatten(x)[0], xs)))
  return jax.tree_util.tree_unflatten(tree_def, ys)


@jax.jit
def tree_zip_jnp(xs):
  xs = list(xs)
  _, tree_def = jax.tree_util.tree_flatten(xs[0])
  ys = map(jnp.asarray,
           zip(*map(lambda x: jax.tree_util.tree_flatten(x)[0], xs)))
  return jax.tree_util.tree_unflatten(tree_def, ys)


def first_dim(a):
  return jax.tree_util.tree_flatten(a)[0][0].shape[0]


def match_type(struct1, struct2):
  leaves = jax.tree_util.tree_leaves(struct2)
  for l in leaves:
    if not hasattr(l, "dtype"):
      raise ValueError("The target struct doesn't have dtype specified?"
                       f" Value found: {l}")
  return jax.tree_util.tree_map(lambda a, b: jnp.asarray(a, dtype=b.dtype),
                                struct1, struct2)


def map_named(function: Callable[[str, Any], Any],
              val: Any,
              key: Optional[str] = "") -> Any:
  """Map a given function over pytree with a string path name.
  For example:
  ```
  a = {"a": 1, "b": {"c": 1}}
  map_named(lambda k,v: v*2 if "a/b/c"==k else v, a)
  ```
  Will be `{"a": 1, "b": {"c": 2}}`.
  Args:
    function: Callable with 2 inputs: key, value
    val: Pytree to map over
    key: Optional initial key
  Returns:
    Struct with the same pytree.
  """
  if isinstance(val, Mapping):
    return type(val)(
        **{k: map_named(function, v, key + "/" + k) for k, v in val.items()})
  elif isinstance(val, tuple) or isinstance(val, list):
    return type(val)(
        *
        [map_named(function, v, key + "/" + str(i)) for i, v in enumerate(val)])
  # check if it's a flax dataclass
  elif hasattr(val, "__dataclass_fields__"):
    classname = repr(val).split("(")[0]
    return type(val)(**{
        k: map_named(function, v, f"{key}/{classname}.{k}")
        for k, v in val.__dataclass_fields__.items()
    })
  else:
    return function(key, val)


def strip_weak_type(pytree):

  def maybe_remove_weak(x):
    if not isinstance(x, jnp.ndarray):
      x = jnp.asarray(x)
    return x

  return jax.tree_util.tree_map(maybe_remove_weak, pytree)


FilterFN = Callable[[str, chex.Array], bool]


@flax.struct.dataclass
class PartitionUnflatten:
  data: Any

  def __call__(self, partitioned_vals):
    return partition_unflatten(self, partitioned_vals)


def partition(functions: Sequence[FilterFN],
              values: chex.ArrayTree,
              strict: bool = False):
  """Split a pytree up into disjoint lists of values.
  The resulting data can then be manipulated and combined again by either
    calling the unflattener, or `partition_unflatten`.
  Args:
    functions: list of boolean functions which to filter. We always partition
      based on the first true function if more than one returns true.
    values: The pytree to be partitioned.
    strict: If set to False, an additional partition is returned.
  Returns:
    partitions: List of lists containing partitioned values
    unflattener: A pytree which can be used to unflatten values.
  """

  vals, struct = jax.tree_util.tree_flatten(values)

  def get_name(k, v):
    del v
    return k

  keys = jax.tree_util.tree_leaves(map_named(get_name, "", values))
  keys = [str(i) for i, v in enumerate(vals)]
  if not strict:
    functions = list(functions) + [lambda k, v: True]

  partitions = [[] for _ in functions]
  names = [[] for _ in functions]

  for k, v in zip(keys, vals):
    has_got = False
    for fi, f in enumerate(functions):
      if f(k, v):
        partitions[fi].append(v)
        names[fi].append(k)
        has_got = True
        break
    assert has_got, f"No matching found for: {k}"
  data_to_restore = (tuple(keys), tuple(names), struct)
  return partitions, PartitionUnflatten(data_to_restore)


def partition_unflatten(unflattener: PartitionUnflatten,
                        part_values: Sequence[jnp.ndarray]) -> Any:
  """Unflatten the paritioned values from `partition`.
  Args:
    unflattener: The unflattener object from `partition`.
    part_values: The partitioned values.
  Returns:
    tree: The original pytree of values.
  """

  keys, names, struct = unflattener.data
  unmap = {k: i for i, k in enumerate(keys)}
  to_fill = [None for _ in keys]
  for name, part in zip(names, part_values):
    for n, p in zip(name, part):
      to_fill[unmap[n]] = p

  return jax.tree_util.tree_unflatten(struct, to_fill)


@gin.configurable
def run_train(
    train_log_dir: str,
    lopt: None = gin.REQUIRED,
    outer_learner_fn: None = gin.REQUIRED,
    num_estimators: int = 2,
    is_trainer: bool = True,
    is_worker: bool = True,
    worker_id: int = 0,
    summary_every_n: int = 10,
    num_steps: int = 10000,
    num_seconds: float = 0.,
    trainer_batch_size: int = 1,
    staleness: int = 1,
    stochastic_resample_frequency: int = 200,
    sample_estimator_fn=None,
    population_worker_id: int = 0,
    population_root_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
    learner_mode="async",
    run_num_estimators_per_gradient: Optional[int] = None,
):
  pass


def main():
  import better_exchook
  better_exchook.install()
  import argparse
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--path", default=f"{_pretrain_root}{_name}/params")
  args = arg_parser.parse_args()
  opt = opt_from_checkpoint(args.path)
  print(opt)


if __name__ == "__main__":
  main()
