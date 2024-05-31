"""
subprocess communicate dies with SIGPIPE. Are the file handles closed?

https://github.com/rwth-i6/sisyphus/issues/190
"""

import gc
import subprocess
import sys
import argparse
import time
import psutil


def test():
    print("Starting...")
    _status()

    try:
        out, err = _test_popen_timeout()
        print(out)
        print(err)
        raise Exception("expected TimeoutExpired")
    except subprocess.TimeoutExpired as exc:
        print("got expected TimeoutExpired:", exc)
        _status()

    print("Going on...")
    gc.collect()
    _status()


def _test_popen_timeout():
    print("Starting subproc...")
    p = subprocess.Popen(
        [sys.executable, __file__, "--loop"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _status()
    return p.communicate(timeout=1)


def _status():
    print("subprocs:")
    cur_proc = psutil.Process()
    have = False
    for subproc in cur_proc.children(recursive=True):
        print(" ", subproc)
        have = True
    if not have:
        print("  none")

    print("open files:")
    have = False
    for f in cur_proc.open_files():
        print(" ", f)
        have = True
    if not have:
        print("  none")

    print("connections:")
    have = False
    for c in cur_proc.connections():
        print(" ", c)
        have = True
    if not have:
        print("  none")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--loop", action="store_true")

    if argparser.parse_args().loop:
        while True:
            time.sleep(1)

    test()


if __name__ == "__main__":
    main()

