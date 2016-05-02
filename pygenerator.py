#!/usr/bin/env python

def gen():
	print "gen1"
	yield 1
	print "gen2"
	yield 2
	print "gen3"


print "main1"
g = gen()
print "main2"
for x in g:
	print "main iter:", x

print "main exit"
