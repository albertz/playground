#!/usr/bin/env python3

"""
https://github.com/rwth-i6/returnn/wiki/Distributed-TensorFlow
https://www.tensorflow.org/api_docs/python/tf/distribute/Server
https://stackoverflow.com/questions/61986820/tf-distribute-server-pipe-creation-failed-too-many-open-files-segfault
https://stackoverflow.com/questions/61988958/tf-python-remote-call-vs-tf-device-when-do-you-need-remote-call
"""

import tensorflow as tf
import better_exchook
import multiprocessing
import time

better_exchook.install()
tf.compat.v1.disable_eager_execution()

cluster_def = {
  'worker': ["localhost:12345", "localhost:23456"]
}
# task_type in {"chief", "worker"}
# task_type == job_name (?)
# worker_device = "/job:%s/task:%d" % (task_type, task_id)


def proc_server(task_index: int):
  server = tf.distribute.Server(cluster_def, task_index=task_index)
  server.join()


def main():
  print("TF:", tf.version.VERSION)

  multiprocessing.set_start_method("spawn")
  proc = multiprocessing.Process(target=proc_server, args=(0,), daemon=True)
  proc.start()
  proc = multiprocessing.Process(target=proc_server, args=(1,), daemon=True)
  proc.start()

  target = "grpc://%s" % cluster_def["worker"][0]

  # Remote session.
  with tf.compat.v1.Session(target) as session:
    session.run(tf.print("Hey"))
    # Do not try this at home.
    # session.run(tf.raw_ops.Abort(error_msg="Exit", exit_without_error=True))

  from tensorflow.python.eager import function

  #@tf.function(autograph=False)
  @function.defun(input_signature=[tf.TensorSpec([], tf.int32)])
  def f(x):
    with tf.control_dependencies([tf.print("Hey from f")]):
      return tf.constant([1]) + x

  # Local session.
  with tf.compat.v1.Session() as session:
    session.run(tf.python.remote_call(target, f=f._get_concrete_function_internal(), Tout=[tf.int32], args=[1]))

  while True:
    time.sleep(1)


if __name__ == '__main__':
  main()
