"""
subprocess communicate dies with SIGPIPE. Are the file handles closed?

https://github.com/rwth-i6/sisyphus/issues/190
"""

import gc
import os
import subprocess
import sys
import argparse
import time
import psutil


def test_subproc_popen():
    print("Starting...")
    _status()

    try:
        out, err = _test_subproc_popen_timeout()
        print(out)
        print(err)
        raise Exception("expected TimeoutExpired")
    except subprocess.TimeoutExpired as exc:
        print("got expected TimeoutExpired:", exc)
        _status()

    print("Going on...")
    gc.collect()
    _status()


def _test_subproc_popen_timeout():
    print("Starting subproc...")
    p = subprocess.Popen(
        [sys.executable, __file__, "--loop"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _status()
    return p.communicate(timeout=1)


def test_subproc_run():
    print("Starting...")
    _status()

    try:
        print("Starting subproc...")
        res = subprocess.run(
            [sys.executable, __file__, "--loop"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1,
        )
        print(res)
        _status()
        raise Exception("expected TimeoutExpired")
    except subprocess.TimeoutExpired as exc:
        print("got expected TimeoutExpired:", exc)
        _status()

    print("Going on...")
    gc.collect()
    _status()


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
    for fd, dest in _get_open_files().items():
        print(f"  {fd} -> {dest}")
        have = True
    if not have:  # should always have 0,1,2 though...
        print("  none")


def _get_open_files() -> dict[int, str]:
    res = {}
    for f in os.scandir("/proc/self/fd"):
        res[int(f.name)] = os.readlink(f.path)
    return res


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--loop", action="store_true")
    argparser.add_argument("cmd", nargs="?")
    args = argparser.parse_args()

    if args.loop:
        while True:
            time.sleep(1)

    if args.cmd:
        print(f"run {args.cmd}()")
        globals()[args.cmd]()
    else:
        print("No command given. Test funcs:")
        for name in globals():
            if name.startswith("test_"):
                print(f"  {name}")


if __name__ == "__main__":
    main()
