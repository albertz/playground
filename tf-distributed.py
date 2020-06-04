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
import os
from returnn import TFUtil

better_exchook.install()
tf.compat.v1.disable_eager_execution()

cluster_def = {
  'worker': ["localhost:12345", "localhost:23456"]
}
# task_type in {"chief", "worker"}
# task_type == job_name (?)
# worker_device = "/job:%s/task:%d" % (task_type, task_id)


class DeviceNameMod:

  _tf_mod = None

  @classmethod
  def get_mod(cls, verbose=False):
    """
    :param bool verbose:
    :return: module
    """
    if cls._tf_mod:
      return cls._tf_mod

    # also see:
    # https://github.com/tensorflow/tensorflow/blob/4b628e8a154c4fbd74ed13d3284fb887a5103e41/tensorflow/core/kernels/data/experimental/prefetching_kernels.cc

    src_code = """
    #include "tensorflow/core/framework/common_shape_fns.h"
    #include "tensorflow/core/framework/op.h"
    #include "tensorflow/core/framework/op_kernel.h"
    #include "tensorflow/core/framework/device_attributes.pb.h"
    #include "tensorflow/core/framework/resource_mgr.h"

    using namespace tensorflow;

    REGISTER_OP("GetDeviceName")
      .Output("out: string")
      .SetShapeFn(shape_inference::ScalarShape);

    class GetDeviceNameOp : public OpKernel {
    public:
      explicit GetDeviceNameOp(OpKernelConstruction* context) : OpKernel(context) {}

      void Compute(OpKernelContext* context) override {
        const DeviceAttributes& attribs = context->device()->attributes();
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(0, TensorShape({}), &output_tensor));
        output_tensor->scalar<string>()() = 
          context->device()->name();
          // attribs.device_type(); 
          // attribs.physical_device_desc();
          // context->session_handle();
          // context->resource_manager()->DebugString();
      }
    };

    REGISTER_KERNEL_BUILDER(Name("GetDeviceName").Device(DEVICE_CPU), GetDeviceNameOp);
    REGISTER_KERNEL_BUILDER(Name("GetDeviceName").Device(DEVICE_GPU).HostMemory("out"), GetDeviceNameOp);
    """

    compiler = TFUtil.OpCodeCompiler(
      base_name="GetDeviceName", code_version=1, code=src_code,
      is_cpp=True, use_cuda_if_available=True,
      # This would lead to a get_tf_list_local_devices call, which we might not want at this point.
      cuda_auto_min_compute_capability=False,
      verbose=verbose)
    tf_mod = compiler.load_tf_module()
    assert hasattr(tf_mod, "get_device_name"), "content of mod: %r" % (dir(tf_mod),)
    cls._tf_mod = tf_mod
    return tf_mod

  @classmethod
  def get_device_name(cls):
    """
    :return: scalar string
    :rtype: tf.Tensor
    """
    return cls.get_mod().get_device_name()


class OsPidMod:

  _tf_mod = None

  @classmethod
  def get_mod(cls, verbose=False):
    """
    :param bool verbose:
    :return: module
    """
    if cls._tf_mod:
      return cls._tf_mod

    src_code = """
    #include <sys/types.h>
    #include <unistd.h>
    #include "tensorflow/core/framework/common_shape_fns.h"
    #include "tensorflow/core/framework/op.h"
    #include "tensorflow/core/framework/op_kernel.h"

    using namespace tensorflow;

    REGISTER_OP("GetOsPid")
      .Output("out: int64")
      .SetShapeFn(shape_inference::ScalarShape);

    class GetOsPidOp : public OpKernel {
    public:
      explicit GetOsPidOp(OpKernelConstruction* context) : OpKernel(context) {}

      void Compute(OpKernelContext* context) override {
        const DeviceAttributes& attribs = context->device()->attributes();
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(0, TensorShape({}), &output_tensor));
        output_tensor->scalar<int64>()() = ::getpid();  
      }
    };

    REGISTER_KERNEL_BUILDER(Name("GetOsPid").Device(DEVICE_CPU), GetOsPidOp);
    REGISTER_KERNEL_BUILDER(Name("GetOsPid").Device(DEVICE_GPU).HostMemory("out"), GetOsPidOp);
    """

    compiler = TFUtil.OpCodeCompiler(
      base_name="GetOsPid", code_version=1, code=src_code,
      is_cpp=True, use_cuda_if_available=True,
      # This would lead to a get_tf_list_local_devices call, which we might not want at this point.
      cuda_auto_min_compute_capability=False,
      verbose=verbose)
    tf_mod = compiler.load_tf_module()
    assert hasattr(tf_mod, "get_os_pid"), "content of mod: %r" % (dir(tf_mod),)
    cls._tf_mod = tf_mod
    return tf_mod

  @classmethod
  def get_os_pid(cls):
    """
    :return: scalar int64
    :rtype: tf.Tensor
    """
    return cls.get_mod().get_os_pid()


def get_py_res():
  def f():
    return "Hello, pid %i" % os.getpid()
  return tf.numpy_function(f, [], tf.string, name="py_res")


def proc_server(task_index: int):
  print("Proc server, task index %i, pid %i" % (task_index, os.getpid()))
  DeviceNameMod.get_mod()  # make op available
  OsPidMod.get_mod()  # make op available
  get_py_res()  # make available
  server = tf.distribute.Server(cluster_def, task_index=task_index)
  server.join()


def main():
  print("TF:", tf.version.VERSION)
  DeviceNameMod.get_mod()
  OsPidMod.get_mod()

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

  from tensorflow.python.eager.function import defun
  from tensorflow.python.framework.function import Defun

  def get_current_device():
    """
    :rtype: tf.Tensor
    :return: string, e.g. "/job:worker/replica:0/task:1/device:CPU:0"
    """
    dummy_iter = tf.data.Iterator.from_structure(tf.int32)
    # noinspection PyProtectedMember
    return tf.raw_ops.IteratorGetDevice(resource=dummy_iter._iterator_resource)

  #@tf.function(autograph=False)
  #@defun(input_signature=[tf.TensorSpec([], tf.int32)])
  @Defun(tf.int32)
  def f(x):
    with tf.control_dependencies([
          tf.print(
            "Hey from f,",
            "x:", x,
            "device:", get_current_device(),
            "dev name:", DeviceNameMod.get_device_name(),
            "os pid:", OsPidMod.get_os_pid(),
            "py res:", get_py_res()
          )]):
      return tf.constant([1]) + x

  #f_ = f._get_concrete_function_internal()  # via defun
  f_ = f

  # Local session.
  with tf.compat.v1.Session(target) as session:
    session.run(
      tf.python.remote_call("/task:1", f=f_, Tout=[tf.int32], args=[1]))
    with tf.device("/task:0"):
      session.run(f(1))
    with tf.device("/task:1"):
      session.run(f(1))

  #while True:
  #  time.sleep(1)


if __name__ == '__main__':
  main()
