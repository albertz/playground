"""
Code via Jan-Thorsten Peter (JTP)
"""

import sys
import os
import glob
import time
from multiprocessing.pool import ThreadPool
pool = ThreadPool(20)

def check_file(filename):
    try:
        os.stat(filename)
        glob.glob(filename+'/*')
    except FileNotFoundError:
        pass

def single_thread():
    for i in sys.argv[1:]:
        check_file(i)

def multi_thread():
    pool.map(check_file, sys.argv[1:])

start = time.time()
single_thread()
print('Single:', time.time()-start)

start = time.time()
multi_thread()
print('Multi', time.time()-start)
