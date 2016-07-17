#!/usr/bin/env python

import sys
import subprocess
import numpy
import ctypes
import atexit
import gc
import better_exchook
better_exchook.install()


libc_so = {"darwin": "libc.dylib", "linux2": ""}[sys.platform]
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


def _darwin_get_sysctl(key):
    res = subprocess.check_output(["sysctl", key]).split()
    assert len(res) == 2
    assert res[0] == "%s:" % key
    return int(res[1])

def _darwin_set_sysctl(key, value):
    cmd = ["sudo", "sysctl", "-w", "%s=%s" % (key, value)]
    print("Calling %r" % cmd)
    subprocess.check_call(cmd)

def _darwin_check_sysctl(key, minvalue):
    value = _darwin_get_sysctl(key)
    if value < minvalue:
        print("Value %s = %s < %s" % (key, value, minvalue))
        _darwin_set_sysctl(key, minvalue)

ShmmaxWanted = 1024 ** 3

def check_shmmax():
    if sys.platform == "darwin":
        # https://support.apple.com/en-us/HT4022
        _darwin_check_sysctl("kern.sysv.shmmax", ShmmaxWanted)
        _darwin_check_sysctl("kern.sysv.shmall", ShmmaxWanted / 4048)
        #_darwin_check_sysctl("kern.sysv.shmmni", 256)  # Operation not permitted?
        #_darwin_check_sysctl("kern.sysv.shmseg", 256)
        # Maybe call cleanup_shared_mem.py?
    else:
        print("check_shmmax not implemented for platform %r" % sys.platform)


class SharedMem:
    def __init__(self, size, shmid=None):
        self.size = size
        self.shmid = None
        self.ptr = None
        if shmid is None:
            self.is_creator = True
            self.shmid = shmget(IPC_PRIVATE, self.size, 0o600)
            assert self.shmid > 0
            atexit.register(self.remove)
        else:
            self.is_creator = False
            self.shmid = shmid
            assert self.shmid > 0
        self.ptr = shmat(self.shmid, 0, 0)
        assert self.ptr > 0

    def remove(self):
        if self.ptr:
            shmdt(self.ptr)
            self.ptr = None
        if self.shmid and self.is_creator:
            shmctl(self.shmid, IPC_RMID, 0)
            self.shmid = None

    def __del__(self):
        self.remove()

    def __getstate__(self):
        return {"size": self.size, "shmid": self.shmid}

    def __setstate__(self, state):
        self.__init__(**state)


def next_power_of_two(n):
    return 2 ** ((n - 1).bit_length())


class SharedNumpyArray:
    # cls members
    server_instances = set()
    extra_space = 4048
    # local members
    is_server = False
    mem = None
    shape, strides, typestr = None, None, None
    array = None

    @classmethod
    def needed_mem_size(cls, shape, typestr):
        itemsize = int(typestr[2:])
        mem_size = cls.extra_space + itemsize * numpy.prod(shape)
        return mem_size

    @classmethod
    def create_copy(cls, array):
        assert isinstance(array, numpy.ndarray)
        array_intf = array.__array_interface__
        shape = array_intf["shape"]
        strides = array_intf["strides"]
        typestr = array_intf["typestr"]
        inst = cls.create_new(shape=shape, strides=strides, typestr=typestr)
        inst.array[...] = array
        return inst

    @classmethod
    def create_new(cls, shape, strides, typestr):
        needed_mem_size = cls.needed_mem_size(shape=shape, typestr=typestr)
        for inst in cls.server_instances:
            assert isinstance(inst, SharedNumpyArray)
            if inst.is_in_use(): continue
            if inst.mem.size < needed_mem_size: continue
            inst._set_is_used(1)
            inst._create_numpy(shape=shape, strides=strides, typestr=typestr)
            return inst
        return cls(shape=shape, strides=strides, typestr=typestr)

    @classmethod
    def create_from_shared(cls, shape, strides, typestr, mem):
        return cls(shape=shape, strides=strides, typestr=typestr, mem=mem)

    def __init__(self, shape, strides, typestr, mem=None):
        if not mem:
            self.is_server = True
            mem_size = next_power_of_two(self.needed_mem_size(shape=shape, typestr=typestr))
            self.mem = SharedMem(size=mem_size)
            self._set_is_used(1)
            self.server_instances.add(self)
        else:
            self.is_server = False
            mem_size = self.needed_mem_size(shape=shape, typestr=typestr)
            assert isinstance(mem, SharedMem)
            assert mem.size >= mem_size
            assert mem.shmid > 0
            assert mem.ptr > 0
            self.mem = mem
        self._create_numpy(shape=shape, strides=strides, typestr=typestr)

    def _create_numpy(self, shape, strides, typestr):
        self.shape = shape
        self.strides = strides
        self.typestr = typestr
        array_intf = {
            "data": (self.mem.ptr + self.extra_space, False),
            "shape": shape,
            "strides": strides,
            'typestr': typestr,
            "version": 3
        }
        class A:
            _base = self
            __array_interface__ = array_intf
        a = numpy.array(A, copy=False)
        assert not a.flags.owndata
        self.array = a

    def _get_in_use_flag_ref(self):
        assert self.mem.ptr > 0
        return ctypes.cast(ctypes.c_void_p(self.mem.ptr), ctypes.POINTER(ctypes.c_uint64)).contents

    def _set_is_used(self, n):
        self._get_in_use_flag_ref().value = n

    def is_in_use(self):
        return self._get_in_use_flag_ref().value > 0

    def set_unused(self):
        if self.mem:
            self._set_is_used(0)
            self.mem.remove()
            self.mem = None

    def __getstate__(self):
        return {
            "shape": self.shape, "strides": self.strides, "typestr": self.typestr,
            "mem": self.mem
        }

    def __setstate__(self, state):
        print("setstate %r" % state)
        self.__init__(**state)

    def __del__(self):
        # On the server side, we will get deleted at program end
        # because we are referenced in the global SharedNumpyArray.server_instances.
        # On the client side, we will get deleted once we are not used anymore.
        # Note that self.array holds a reference to self.
        self.set_unused()

    def __repr__(self):
        return "<%s is_server=%r state=%r>" % (self.__class__.__name__, self.is_server, self.__getstate__())


def pickle_std(s, v):
    import pickle
    pickler = pickle.Pickler(file=s, protocol=-1)
    pickler.dump(v)
    s.flush()

def pickle_ext(s, v):
    import extpickle
    pickler = extpickle.Pickler(file=s)
    pickler.dump(v)
    s.flush()

def pickle_shm(s, v):
    import pickle
    assert isinstance(v, tuple)
    if len(v) == 2:
        assert isinstance(v[1], numpy.ndarray)
        shared = SharedNumpyArray.create_copy(v[1])
        v = (v[0], shared)
    pickler = pickle.Pickler(file=s, protocol=-1)
    pickler.dump(v)
    s.flush()

pickle = pickle_std

def unpickle(s):
    import pickle
    unpickler = pickle.Unpickler(file=s)
    v = unpickler.load()
    assert isinstance(v, tuple)
    if len(v) == 2 and isinstance(v[1], SharedNumpyArray):
        a = v[1]
        assert a.array is not None
        return (v[0], a.array)
    return v


LoopCount = 10
MatrixSize = 1000

def demo():
    if pickle is pickle_shm:
        check_shmmax()
    p = subprocess.Popen([__file__] + sys.argv[1:] + ["--client"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    for i in range(LoopCount):
        m = numpy.random.randn(MatrixSize, MatrixSize)
        pickle(p.stdin, ("ping", m))
        out, m2 = unpickle(p.stdout)
        assert out == "pong"
        assert isinstance(m2, numpy.ndarray)
        assert m.shape == m2.shape
        assert numpy.isclose(m, m2).all()
        gc.collect()
    print("Copying done, exiting.")
    pickle(p.stdin, ("exit",))
    out, = unpickle(p.stdout)
    assert out == "exit"
    p.wait()
    print("Done. Return code %i" % p.returncode)


def demo_client():
    in_stream = sys.stdin
    out_stream = sys.stdout
    sys.stdout = sys.stderr
    print("Hello from client!")
    while True:
        cmd = unpickle(in_stream)
        assert isinstance(cmd, tuple)
        if cmd[0] == "exit":
            pickle(out_stream, ("exit",))
            break
        elif cmd[0] == "ping":
            a = cmd[1]
            assert isinstance(a, numpy.ndarray)
            pickle(out_stream, ("pong", a))
            gc.collect()
        else:
            assert False, "unknown: %r" % cmd
    print("Exit from client!")


if __name__ == "__main__":
    if sys.argv[1] == "--shared_mem":
        pickle = pickle_shm
    elif sys.argv[1] == "--pickle":
        pickle = pickle_std
    elif sys.argv[1] == "--extpickle":
        pickle = pickle_ext
    else:
        assert False, "unknown args: %r" % sys.argv[1:]
    if sys.argv[2:] == ["--client"]:
        demo_client()
    elif sys.argv[2:] == []:
        demo()
    else:
        assert False, "unknown args: %r" % sys.argv[1:]

