#!/usr/bin/python

import sys, os

print sys.argv, os.getpid()

if "--fork" in sys.argv:
	print "in fork"
	argidx = sys.argv.index("--fork")
	readend = int(sys.argv[argidx + 1])
	
	print "read:", os.read(readend, 100)
	sys.exit(0)

readend,writeend = os.pipe()
pid = os.fork()

if pid == 0: # child
	os.close(writeend)
	print "child", os.getpid()
	os.execv(sys.argv[0], sys.argv + ["--fork", str(readend)])
	
else: # parent
	os.close(readend)
	
	os.write(writeend, "foo")
	print "parent"
	