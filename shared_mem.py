#!/usr/bin/env python

import numpy
import subprocess
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
        shmctl(self.shmid, IPC_RMID, 0)
        self.shmid = None

    def __del__(self):
        self.remove()


class SharedMemClient:
    def __init__(self, size, shmid):
        self.size = size
        self.shmid = shmid
        self.ptr = None
        assert self.shmid > 0
        self.ptr = shmat(self.shmid, 0, 0)
        assert self.ptr

    def remove(self):
        shmdt(self.ptr)
        self.ptr = None

    def __del__(self):
        self.remove()


def demo():
    p = subprocess.Popen([__file__, "--client"], stdin=subprocess.PIPE)

    m = numpy.arange(100, dtype="float32").reshape(10, 10)
    assert m.itemsize * numpy.prod(m.shape) == m.nbytes
    mem = SharedMem(m.nbytes)
    memcpy(mem.ptr, m.ctypes.data, m.nbytes)

    p.stdin.write("%i\n" % mem.size)
    p.stdin.write("%i\n" % mem.shmid)
    p.stdin.write("%r\n" % m.__array_interface__)
    p.wait()
    print("Done. Return code %i" % p.returncode)


def demo_client():
    size = int(sys.stdin.readline())
    shmid = int(sys.stdin.readline())
    array_intf = eval(sys.stdin.readline())

    mem = SharedMemClient(size, shmid)
    array_intf["data"] = (mem.ptr, False)
    class A: __array_interface__ = array_intf
    m2 = numpy.array(A, copy=False)
    assert not m2.flags.owndata

    m = numpy.arange(100, dtype="float32").reshape(10, 10)
    assert numpy.isclose(m, m2).all()
    print("Yup!")


if __name__ == "__main__":
    if sys.argv[1:] == ["--client"]:
        demo_client()
    else:
        demo()


