#!/usr/bin/env python3

import subprocess
import distutils.sysconfig
import tensorflow as tf

py_compile_vars = distutils.sysconfig.get_config_vars()
py_compile_flags = py_compile_vars["CFLAGS"].split(" ")
py_compile_flags += ["-I", py_compile_vars["INCLUDEPY"]]
py_link_flags = py_compile_vars["LDFLAGS"].split(" ")
if py_compile_vars["LIBDIR"]:
    py_link_flags += [
        "-L", py_compile_vars["LIBDIR"],
        "-Wl,-rpath," + py_compile_vars["LIBDIR"],
    ]
py_link_flags += ["-lpython" + py_compile_vars["LDVERSION"]]

tf_compile_flags = tf.sysconfig.get_compile_flags()
tf_link_flags = tf.sysconfig.get_link_flags()
tf_link_flags += ["-Wl,-rpath," + tf.sysconfig.get_lib()]

cpp_bin = "g++"

compile_args = (
    [cpp_bin] + py_compile_flags + tf_compile_flags +
    ["cpp-link-py-tf.cpp", "-o", "cpp-link-py-tf.bin"] +
    py_link_flags + tf_link_flags)
print("$", " ".join(map(repr, compile_args)))
subprocess.check_call(compile_args)

