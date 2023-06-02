#!/usr/bin/env python3

import subprocess
import distutils.sysconfig
import tensorflow as tf
import sys
import re

py_compile_vars = distutils.sysconfig.get_config_vars()
py_compile_flags = py_compile_vars["CFLAGS"].split() if py_compile_vars["CFLAGS"] else []
py_compile_flags += ["-I", py_compile_vars["INCLUDEPY"]]
py_link_flags = py_compile_vars["LDFLAGS"].split() if py_compile_vars["LDFLAGS"] else []
if py_compile_vars["LIBDIR"]:
    py_link_flags += [
        "-L", py_compile_vars["LIBDIR"],
        "-Wl,-rpath," + py_compile_vars["LIBDIR"],
    ]
py_link_flags += ["-lpython" + py_compile_vars["LDVERSION"]]

if sys.platform.startswith('linux'):
    # There is some symbol conflict between libtensorflow_framework.so and the Python hashlib module
    # on some OpenSSL symbols.
    # The Python hashlib module assumes the symbols to come from libcrypto.so
    # but libtensorflow_framework.so overwrites them,
    # which can lead to segfaults when the Python hashlib module is used
    # (which happens all the time, e.g. while importing Numpy).
    # https://github.com/numpy/numpy/issues/21588
    # https://github.com/pybind/pybind11/issues/3543
    # https://unix.stackexchange.com/questions/594790/tool-for-finding-shared-library-symbol-conflicts/747804#747804
    import _hashlib
    out = subprocess.check_output(["ldd", _hashlib.__file__])
    # Lines like:
    # "        libcrypto.so.1.1 => /work/tools/users/zeyer/linuxbrew/lib/libcrypto.so.1.1 (0x00007f1a3fa01000)"
    path = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith(b"libcrypto.so."):
            line = line.decode("utf8")
            m = re.match(r"libcrypto\.so.* => (.*)/libcrypto\.so.*", line)
            path = m.group(1)
            break
    assert path, "Could not find libcrypto.so path in ldd output:\n" + out.decode("utf8")
    py_link_flags += ["-L", path, "-Wl,-rpath," + path, "-Wl,-no-as-needed", "-lcrypto"]

tf_compile_flags = tf.sysconfig.get_compile_flags()
tf_link_flags = tf.sysconfig.get_link_flags()
tf_link_flags += ["-Wl,-rpath," + tf.sysconfig.get_lib()]

cpp_bin = "g++"

compile_args = (
    [cpp_bin] + py_compile_flags + tf_compile_flags +
    ["cpp-link-py-tf.cpp", "-o", "cpp-link-py-tf.bin"] +
    py_link_flags + tf_link_flags)
print("$", " ".join(compile_args))
subprocess.check_call(compile_args)

