#!/usr/bin/env python3

"""
Via: https://github.com/rwth-i6/returnn/issues/300
By: Akshat Dewan <https://github.com/akshatdewan>
"""

import tensorflow as tf
from numpy.random import default_rng
import time

tf.compat.v1.disable_eager_execution()


rng = default_rng()
vals = rng.standard_normal((100,100))
more_vals = rng.standard_normal((100,100))

# Initialize two constants
x1 = tf.constant(vals)
x2 = tf.constant(more_vals)

# Multiply
result = tf.multiply(x1, x2)

# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)

with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
# Initialize Session and run `result`
  output = sess.run(result)
  print(output)

  print("press Ctrl+C to stop")
  while True:
    time.sleep(1)
