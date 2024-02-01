#!/usr/bin/env python3
"""
Test hang
"""

import sys
import psutil
import threading
import subprocess


def main():
    """main"""
    threading.Thread(target=_proc_stat_worker, daemon=True).start()

    while True:
        thread = threading.Thread(target=_run_sub_proc, daemon=True)
        thread.start()
        thread.join(1)
        assert not thread.is_alive()


def _proc_stat_worker():
    while True:
        proc = psutil.Process()
        proc.children()


def _run_sub_proc():
    subprocess.run(["sleep", "0.01"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        sys.exit(1)
