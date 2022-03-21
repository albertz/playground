
from threading import Thread, Condition
from signal import signal, SIGINT
import sys
import time


def main():
  def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(1)

  signal(SIGINT, signal_handler)

  def background():
    for i in range(5):
      print('background step %i' % i)
      time.sleep(1)

  thread = Thread(target=background)
  thread.daemon = True
  thread.start()
  # In Python 2, this blocks the signal handler. Try to press Ctrl+C.
  # In Python 3, the signal handler should still work.
  thread.join()

  while True:
    time.sleep(1)


if __name__ == '__main__':
  main()
