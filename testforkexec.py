#!/usr/bin/python

import sys, os, time
import pickle

print sys.argv, os.getpid()

if "--fork" in sys.argv:
	print "in fork"
	argidx = sys.argv.index("--fork")
	writeend = int(sys.argv[argidx + 1])
	writeend = os.fdopen(writeend, "w")
	readend = int(sys.argv[argidx + 2])
	readend = os.fdopen(readend, "r")
	unpickler = pickle.Unpickler(readend)
	
	print "read:", unpickler.load()
	time.sleep(1)
	sys.exit(0)

def pipeOpen():
	readend,writeend = os.pipe()
	readend = os.fdopen(readend, "r")
	writeend = os.fdopen(writeend, "w")
	return readend,writeend
pipe_c2p = pipeOpen()
pipe_p2c = pipeOpen()
pid = os.fork()

if pid == 0: # child
	pipe_c2p[0].close()
	pipe_p2c[1].close()
	print "child", os.getpid()
	os.execv(sys.argv[0], sys.argv + [
		"--fork",
		str(pipe_c2p[1].fileno()),
		str(pipe_p2c[0].fileno())
		])
	
else: # parent
	pipe_c2p[1].close()
	pipe_p2c[0].close()
	
	pickler = pickle.Pickler(pipe_p2c[1])
	pickler.dump("foo")
	pipe_p2c[1].flush()
	print "parent"
	time.sleep(1)
	