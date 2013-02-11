#!/usr/bin/python

import sys, os
import pickle

print sys.argv, os.getpid()

if "--fork" in sys.argv:
	print "in fork"
	argidx = sys.argv.index("--fork")
	readend = int(sys.argv[argidx + 1])
	readend = os.fdopen(readend, "r")
	unpickler = pickle.Unpickler(readend)
	
	print "read:", unpickler.load()
	sys.exit(0)

def pipeOpen():
	readend,writeend = os.pipe()
	readend = os.fdopen(readend, "r")
	writeend = os.fdopen(writeend, "w")
	return readend,writeend
readend,writeend = pipeOpen()
pid = os.fork()

if pid == 0: # child
	writeend.close()
	print "child", os.getpid()
	os.execv(sys.argv[0], sys.argv + ["--fork", str(readend.fileno())])
	
else: # parent
	readend.close()
	
	pickler = pickle.Pickler(writeend)
	pickler.dump("foo")
	print "parent"
	