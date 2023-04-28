import torch
import tensorflow as tf

n_batch = 1
n_time = 4
n_in_dim = 1

pool_size = 3
stride = 3

x_pt = -torch.ones(n_batch, n_in_dim, n_time, dtype=torch.float32)
x_tf = -tf.ones((n_batch, n_in_dim, n_time), dtype=tf.float32)

y_pt = torch.max_pool1d(x_pt, kernel_size=pool_size, stride=stride, ceil_mode=True, padding=1)
print(y_pt)
y_tf = tf.nn.pool(x_tf, window_shape=[pool_size], strides=[stride], pooling_type="MAX", padding="SAME", data_format="NCW")
print(y_tf)
