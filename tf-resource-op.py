
# https://stackoverflow.com/questions/58774526/tensorflow-custom-c-op-with-resource-handle
# https://stackoverflow.com/questions/55958530/custom-resource-in-tensorflow
# https://github.com/cyberfire/tensorflow-mtcnn/issues/8
# https://github.com/tensorflow/tensorflow/pull/21072
# https://github.com/tensorflow/tensorflow/issues/19747
# https://github.com/tensorflow/tensorflow/issues/17316

import os
import sys
from subprocess import check_call
import tensorflow as tf


CC_NAME = "tf-resource-op.cc"
SO_NAME = "tf-resource-op.so"


def compile_so():
  use_cxx11_abi = hasattr(tf, 'CXX11_ABI_FLAG') and tf.CXX11_ABI_FLAG
  common_opts = ["-shared", "-O2"]
  common_opts += ["-std=c++11"]
  if sys.platform == "darwin":
    common_opts += ["-undefined", "dynamic_lookup"]
  tf_include = tf.sysconfig.get_include()  # e.g. "...python2.7/site-packages/tensorflow/include"
  tf_include_nsync = tf_include + "/external/nsync/public"  # https://github.com/tensorflow/tensorflow/issues/2412
  include_paths = [tf_include, tf_include_nsync]
  for include_path in include_paths:
    common_opts += ["-I", include_path]
  common_opts += ["-fPIC", "-v"]
  common_opts += ["-D_GLIBCXX_USE_CXX11_ABI=%i" % (1 if use_cxx11_abi else 0)]
  common_opts += ["-g"]
  opts = common_opts + [CC_NAME, "-o", SO_NAME]
  ld_flags = ["-L%s" % tf.sysconfig.get_lib(), "-ltensorflow_framework"]
  opts += ld_flags
  cmd_bin = "g++"
  cmd_args = [cmd_bin] + opts
  print("$ %s" % " ".join(cmd_args))
  check_call(cmd_args)


def main():
  if not os.path.exists(SO_NAME):
    compile_so()
  mod = tf.load_op_library(SO_NAME)
  handle = mod.open_fst_load(filename="foo.bar")
  new_states, scores = mod.open_fst_transition(handle=handle, inputs=[0], states=[0])

  with tf.Session() as session:
    # InternalError: ndarray was 1 bytes but TF_Tensor was 98 bytes
    # print("fst:", session.run(handle))

    out_new_states, out_scores = session.run((new_states, scores))
    print("output new states:", out_new_states)
    print("output scores:", out_scores)

    # When session unloads, crashes with assertion:
    # F .../site-packages/tensorflow/include/tensorflow/core/lib/core/refcount.h:79] Check failed: ref_.load() == 0 (1 vs. 0)  # nopep8


if __name__ == '__main__':
  import better_exchook
  better_exchook.install()
  main()
