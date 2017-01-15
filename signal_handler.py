import signal
import time
import os

print("My pid: %i" % os.getpid())

def signal_handler(signum, frame):
	print("Signal %i" % signum)
	print("Frame %r" % frame)

print("Installing signal handler for SIGUSR1.")
signal.signal(signal.SIGUSR1, signal_handler)

print("Looping.")
while True:
	time.sleep(1)

