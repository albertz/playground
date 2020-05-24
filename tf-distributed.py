#!/usr/bin/env python3

"""
https://github.com/rwth-i6/returnn/wiki/Distributed-TensorFlow
https://www.tensorflow.org/api_docs/python/tf/distribute/Server
https://stackoverflow.com/questions/61986820/tf-distribute-server-pipe-creation-failed-too-many-open-files-segfault
"""

import tensorflow as tf
import better_exchook

better_exchook.install()
tf.compat.v1.disable_eager_execution()
print("TF:", tf.version.VERSION)


def main():
  cluster_def = {
    'worker': ["localhost:12345", "localhost:23456"]
  }
  # task_type in {"chief", "worker"}
  # task_type == job_name (?)
  # worker_device = "/job:%s/task:%d" % (task_type, task_id)
  s1 = tf.distribute.Server(cluster_def, task_index=0)
  s2 = tf.distribute.Server(cluster_def, task_index=1)
  s1.join()


if __name__ == '__main__':
  main()
