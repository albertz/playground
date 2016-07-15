#!/usr/bin/env python

import numpy
import ctypes
import sys
import better_exchook
better_exchook.install()

libc_so = {"darwin": "libc.dylib"}[sys.platform]
libc = ctypes.CDLL(libc_so, use_errno=True, use_last_error=True)
shm_key_t = ctypes.c_int
IPC_PRIVATE = 0
IPC_RMID = 0

# int shmget(key_t key, size_t size, int shmflg);
shmget = libc.shmget
shmget.restype = ctypes.c_int
shmget.argtypes = (shm_key_t, ctypes.c_size_t, ctypes.c_int)
# void* shmat(int shmid, const void *shmaddr, int shmflg);
shmat = libc.shmat
shmat.restype = ctypes.c_void_p
shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
# int shmdt(const void *shmaddr);
shmdt = libc.shmdt
shmdt.restype = ctypes.c_int
shmdt.argtypes = (ctypes.c_void_p,)
# int shmctl(int shmid, int cmd, struct shmid_ds *buf);
shmctl = libc.shmctl
shmctl.restype = ctypes.c_int
shmctl.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
# void* memcpy( void *dest, const void *src, size_t count );
memcpy = libc.memcpy
memcpy.restype = ctypes.c_void_p
memcpy.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)

# PyObject_HEAD expands to:
#   Py_ssize_t ob_refcnt;
#   PyTypeObject *ob_type;
# typedef struct PyArrayObject {
#     PyObject_HEAD
#     char *data;
#     int nd;
#     npy_intp *dimensions;
#     npy_intp *strides;
#     PyObject *base;
#     PyArray_Descr *descr;
#     int flags;
#     PyObject *weakreflist;
# } PyArrayObject;

class PyArrayObject(ctypes.Structure):
    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("data", ctypes.c_void_p),
        ("nd", ctypes.c_int),
        ("dimensions", ctypes.POINTER(ctypes.c_long)),
        ("strides", ctypes.POINTER(ctypes.c_long)),
        ("base", ctypes.c_void_p),
        ("descr", ctypes.c_void_p),
        ("flags", ctypes.c_int),
        ("weakreflist", ctypes.c_void_p)
    ]


class SharedMem:
    def __init__(self, size):
        self.size = size
        self.shmid = shmget(IPC_PRIVATE, self.size, 0o600)
        assert self.shmid > 0
        self.ptr = shmat(self.shmid, 0, 0)
        assert self.ptr

    def remove(self):
        shmdt(self.ptr)
        self.ptr = None
        shmctl(self.shmid, IPC_RMID)
        self.shmid = None

    def __del__(self):
        self.remove()


def demo():
    m = numpy.arange(100, dtype="float32").reshape(10, 10)
    m_ptr = ctypes.addressof(ctypes.py_object(m))
    m_c = ctypes.cast(m_ptr, ctypes.POINTER(PyArrayObject))
    print(m_c.value.nd)


if __name__ == "__main__":
    demo()


