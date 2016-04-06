#!/usr/bin/env python

import sys
import theano
import theano.tensor as T
import numpy
import better_exchook
better_exchook.install()

def step(i, s_p):
	return [i], {}, theano.scan_module.until(T.eq(i, 5))

theano_scan = theano.scan

if int((sys.argv + [0])[1]):
	# Path to RETURNN. See homepage of RWTH i6 chair.
	sys.path += ["/u/zeyer/setups/switchboard/2016-01-28--crnn/crnn"]
	import TheanoUtil
	theano_scan = TheanoUtil.unroll_scan

s, _ = theano_scan(step, n_steps=10, sequences=T.arange(10), outputs_info=[numpy.int64(0)])
print s.eval()

