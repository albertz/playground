#!/usr/bin/env python3

# related: http://faulthandler.readthedocs.io
# here we want to print the C stack, not Python stack

import sys
import os
import ctypes
import ctypes.util

def crash():
    print("Crash now")
    ctypes.string_at(0)

#libsigsegv_filename = ctypes.util.find_library("sigsegv")
#assert libsigsegv_filename
#libsigsegv = ctypes.CDLL(libsigsegv_filename)

# http://git.savannah.gnu.org/cgit/libsigsegv.git/tree/src/sigsegv.h.in
#print(libsigsegv)

try:
  # libSegFault on Unix/Linux, not on MacOSX
  libfn = ctypes.util.find_library("SegFault")
  assert libfn
  # Nothing more needed than loading it, it will automatically register itself.
  libSegFault = ctypes.CDLL(libfn)

except Exception as exc:
  print("Error loading libSegFault.so: %s" % exc)

# see signal_handler.c
lib = ctypes.CDLL("./signal_handler.so")
print(lib)

lib.install_signal_handler.return_type = None
lib.install_signal_handler()

print("signal_handler.so ok")



crash()

