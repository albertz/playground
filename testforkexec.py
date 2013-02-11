#!/usr/bin/python

import sys, os

print sys.argv, os.getpid()

if "--fork" in sys.argv:
	print "in fork"
	argidx = sys.argv.index("--fork")
	readend = int(sys.argv[argidx + 1])
	readend = os.fdopen(readend, "r")
	
	print "read:", readend.read()
	sys.exit(0)

readend,writeend = os.pipe()
readend = os.fdopen(readend, "r")
writeend = os.fdopen(writeend, "w")
pid = os.fork()

if pid == 0: # child
	writeend.close()
	print "child", os.getpid()
	os.execv(sys.argv[0], sys.argv + ["--fork", str(readend.fileno())])
	
else: # parent
	readend.close()
	
	writeend.write("foo")
	print "parent"
	