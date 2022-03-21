
from threading import Thread, Condition
from signal import signal, SIGINT
import sys
import time

# concurrent.futures._base:
"""
class Future(object):
    def result(self, timeout=None):
        try:
            with self._condition:
                if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:
                    raise CancelledError()
                elif self._state == FINISHED:
                    return self.__get_result()

                self._condition.wait(timeout)

                if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:
                    raise CancelledError()
                elif self._state == FINISHED:
                    return self.__get_result()
                else:
                    raise TimeoutError()
        finally:
            # Break a reference cycle with the exception in self._exception
            self = None
"""


def main():
  def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(1)

  signal(SIGINT, signal_handler)

  cond = Condition()

  class _Locals:
    stop = False

  def background():
    for i in range(5):
      print('background step %i' % i)
      time.sleep(1)
    with cond:
      _Locals.stop = True
      cond.notify()

  thread = Thread(target=background)
  thread.daemon = True

  # Like Future.result.
  with cond:
    thread.start()

    # In Python 2, this blocks the signal handler. Press Ctrl+C to test.
    # In Python 3, this should be interrupted by the signal handler.
    cond.wait()

  while True:
    time.sleep(1)


if __name__ == '__main__':
  main()
