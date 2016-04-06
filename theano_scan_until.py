#!/usr/bin/env python

import theano
import theano.tensor as T
import numpy

def step(i, s_p):
	return [i], {}, theano.scan_module.until(T.eq(i, 5))

s, _ = theano.scan(step, sequences=T.arange(10), outputs_info=[numpy.int64(0)])
print s.eval()

