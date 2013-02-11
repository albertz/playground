#!/usr/bin/python

import sys, os

print sys.argv, os.getpid()

if "--fork" in sys.argv:
	print "in fork"
	sys.exit(0)

pid = os.fork()

if pid == 0: # child
	print "child", os.getpid()
	os.execv(sys.argv[0], sys.argv + ["--fork"])
	
else: # parent
	
	print "parent"