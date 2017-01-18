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

# libSegFault on Unix/Linux, not on MacOSX
#libfn = ctypes.util.find_library("libSegFault")
#assert libfn
#libSegFault = ctypes.CDLL(libfn)

# see signal_handler.c
lib = ctypes.CDLL("signal_handler.so")
print(lib)

lib.install_signal_handler.return_type = None
lib.install_signal_handler()

print("Ok")


crash()

