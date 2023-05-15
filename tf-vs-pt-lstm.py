"""
Comparing TensorFlow vs PyTorch, naive LSTM implementation
"""

import tensorflow as tf
import torch
import numpy
import numpy.testing


torch.set_float32_matmul_precision("highest")
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


rnd = numpy.random.RandomState(42)
n_time = 20
n_batch = 5
n_in = 7
n_dim = 1024
forget_bias = 1.


kernel = rnd.normal(scale=0.5, size=(n_in + n_dim, 4 * n_dim))
bias = rnd.normal(scale=0.5, size=(4 * n_dim,))
kernel_tf = tf.constant(kernel, dtype=tf.float32)
bias_tf = tf.constant(bias, dtype=tf.float32)
kernel_pt = torch.tensor(kernel, dtype=torch.float32)
bias_pt = torch.tensor(bias, dtype=torch.float32)


c_tf = tf.zeros((n_batch, n_dim))
h_tf = tf.zeros((n_batch, n_dim))
c_pt = torch.zeros((n_batch, n_dim))
h_pt = torch.zeros((n_batch, n_dim))


def _cmp(v_tf: tf.Tensor, v_pt: torch.Tensor, name: str):
    v_tf_np = v_tf.numpy()
    v_pt_np = v_pt.detach().numpy()
    print(f"{name} diff:", numpy.abs(v_pt_np - v_tf_np).max())


for t in range(n_time):
    print("*** t =", t)
    x = rnd.uniform(-1., 1., size=(n_batch, n_in))
    x_tf = tf.constant(x, dtype=tf.float32)

    parts0_tf = tf.matmul(
        tf.concat([x_tf, h_tf], 1), kernel_tf)
    parts_tf = tf.nn.bias_add(parts0_tf, bias_tf)  # (B, 4*D)
    i, j, f, o = tf.split(value=parts_tf, num_or_size_splits=4, axis=1)
    c_tf = (
        tf.sigmoid(f + forget_bias) * c_tf +
        tf.sigmoid(i) * tf.tanh(j))
    h_tf = tf.sigmoid(o) * tf.tanh(c_tf)

    x_pt = torch.tensor(x, dtype=torch.float32)
    parts0_pt = torch.matmul(torch.concat([x_pt, h_pt], 1), kernel_pt)
    parts_pt = parts0_pt + bias_pt
    i, j, f, o = torch.split(parts_pt, [n_dim] * 4, dim=1)
    c_pt = (
        torch.sigmoid(f + forget_bias) * c_pt +
        torch.sigmoid(i) * torch.tanh(j))
    h_pt = torch.sigmoid(o) * torch.tanh(c_pt)

    _cmp(parts0_tf, parts0_pt, "parts0")
    _cmp(parts_tf, parts_pt, "parts")
    _cmp(c_tf, c_pt, "c")
    _cmp(h_tf, h_pt, "h")

