"""
https://youtrack.jetbrains.com/issue/PY-63226/PyCharm-debugger-hangs-when-process-is-forking
"""

import os
import time
import multiprocessing


def main():
    mp_manager = multiprocessing.Manager()

    print("step 1")
    print("step 2")
    time.sleep(3)


def main2():

    pid = os.fork()

    if pid == 0:
        print('child')
        time.sleep(3)
        os._exit(0)

    print("parent. step 1")
    print("parent. step 2")
    os.waitpid(pid, 0)


if __name__ == '__main__':
    main2()
