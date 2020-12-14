
import torch
import numpy
import tensorflow as tf
import numba

N = 3
T = 100000
num_groups = 1
num_channels = 512

inputs_np = numpy.random.randn(N, num_channels, T).astype("float32")
inputs = torch.from_numpy(inputs_np)

model = torch.nn.GroupNorm(num_groups, num_channels)
out = model(inputs)
out = out.detach().numpy()

gn_mean = torch.mean(inputs, dim=(1, 2), keepdim=True)
gn_var = torch.var(inputs, dim=(1, 2), keepdim=True, unbiased=False)
gn_out = (inputs - gn_mean) / torch.sqrt(gn_var + model.eps)
print((gn_out / out)[0, :2, :2] - 1.)


@numba.jit(nopython=True)
def mean_inexact_np(x: numpy.ndarray) -> numpy.ndarray:
  # return torch.sum(x, dim=dim, keepdim=keepdim) / numpy.prod([x.shape[i] for i in dim])
  x_ = x.reshape(x.shape[0], -1)
  n = x_.shape[1]
  acc_ = numpy.zeros_like(x_[:, 0])
  for i in range(n):
    acc_ += x_[:, i]
  res = acc_ / n
  return res


gn_mean = mean_inexact_np(inputs_np)
gn_var = mean_inexact_np(inputs_np * inputs_np) - gn_mean * gn_mean
gn_out = (inputs_np - gn_mean[:, None, None]) / numpy.sqrt(gn_var[:, None, None] + model.eps)
print((gn_out / out)[0, :2, :2] - 1.)

inputs_tf = tf.constant(inputs)
tf_mean, tf_var = tf.nn.moments(inputs_tf, axes=[1, 2], keepdims=True)
tf_out = (inputs_tf - tf_mean) / tf.sqrt(tf_var + model.eps)
print((tf_out / out)[0, :2, :2] - 1.)
