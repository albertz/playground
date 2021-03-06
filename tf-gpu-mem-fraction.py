#!/usr/bin/env python3

"""
Via:
https://github.com/rwth-i6/returnn/issues/300
https://github.com/tensorflow/tensorflow/issues/9374
https://github.com/tensorflow/tensorflow/issues/8021
https://github.com/tensorflow/tensorflow/issues/8136

Reported by Akshat Dewan <https://github.com/akshatdewan>.
"""

import tensorflow as tf
from tensorflow.python.client import device_lib
import subprocess

print("TF version:", tf.__version__)
tf.compat.v1.disable_eager_execution()

v = tf.compat.v1.get_variable("var", shape=(100, 100))
result = v

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
pre_ops = {}
pre_ops["device_count"] = {"GPU": 0}  # this will ignore the gpu_options for now...

with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, **pre_ops)) as sess:
  pass

# The following line will cause the full GPU mem allocation.
print(list(device_lib.list_local_devices()))


with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
  sess.run(tf.compat.v1.global_variables_initializer())
  output = sess.run(result)
  print(output)

  subprocess.check_call(["nvidia-smi"])
