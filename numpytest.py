import random
import numpy
from itertools import *

# inspired by http://empslocal.ex.ac.uk/people/staff/rjchapma/etc/evildet.pdf

def zeromat(n):
	M = []
	for i in xrange(n): M.append([0]*n)
	return M

def rndmatrix(n, numset = [-1,1]):
	M = zeromat(n)
	for i in xrange(0,n):
		for j in xrange(0,n):
			M[i][j] = random.choice(numset)
	return M

def setdiagtozero(M):
	for i in xrange(len(M)):
		M[i][i] = 0

def makeschief(M):
	for i in xrange(len(M) - 1):
		for j in xrange(i + 1, len(M)):
			M[j][i] = - M[i][j]

def test(n):
	M = rndmatrix(n)
	setdiagtozero(M)
	makeschief(M)
	return M, int(round(numpy.linalg.det(M))), det(M)

def rndtest():
	return test(random.choice(range(2, 10, 2)))

def makeidiffm(n):
	M = zeromat(n)
	for i in xrange(n):
		for j in xrange(n):
			M[i][j] = i - j
	return M

def diag(M, permut = lambda x: x):
	d = []
	for i in xrange(len(M)):
		d.append(M[i][permut(i)])
	return d

def sign(p):
	s = 1.0
	for i in xrange(len(p) - 1):
		for j in xrange(i + 1, len(p)):
			s *= p[i] - p[j]
			s /= i - j
	s = int(round(s))
	return s

def diagpermuts(M):
	for p in permutations(range(0,len(M))):
		d = diag(M, lambda i: p[i])
		if not 0 in d: yield d, sign(p)

def prod(l):
	y = 1
	for x in l: y *= x
	return y

def det(M):
	x = 0
	for d,s in diagpermuts(M):
		x += prod(d) * s
	return x

def interestingfacs(n):
	l = [ s * prod(d) for (d,s) in diagpermuts(makeidiffm(n)) ]
	ll = {} # abs fac -> num
	for x in l:
		key = abs(x)
		if not key in ll: ll[key] = 0
		if x > 0: ll[key] += 1
		elif x < 0: ll[key] -= 1
		if ll[key] == 0: ll.pop(key)
	return ll

def legendresymbol(a, p):
	n = ((p - 1) / 2)
	x = 1
	for _ in xrange(n):
		x *= a
		x %= p
	if x > 1: x -= p
	return x

def evildet(n):
	x = 0
	for k,v in interestingfacs(n).iteritems():
		lg = legendresymbol(k, 2 * n - 1)
		x += lg * v
	return x

def evilm(n):
	M = makeidiffm(n)
	p = 2 * n - 1
	for i in xrange(n):
		for j in xrange(n):
			M[i][j] = legendresymbol(M[i][j], p)
	return M

def evildet2(n):
	p = 2 * n - 1
	M = evilm(n)
	return numpy.linalg.det(M)

def primes(n):
    # http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    """ Returns  a list of primes < n """
    sieve = [True] * (n/2)
    for i in xrange(3,int(n**0.5)+1,2):
        if sieve[i/2]:
            sieve[i*i/2::i] = [False] * ((n-i*i-1)/(2*i)+1)
    return [2] + [2*i+1 for i in xrange(1,n/2) if sieve[i]]
