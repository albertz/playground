#!/usr/bin/env python3
"""
Test hang
https://github.com/rwth-i6/sisyphus/issues/176#issuecomment-1920997179
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
        # Via psutil ppid_map(), this will iterate through all procs of the system,
        # and check /proc/<pid>/stat for each proc.
        # We had cases where this hangs, and it might be because of our own subprocess.
        proc.children()


def _run_sub_proc():
    subprocess.run(["sleep", "0.01"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        sys.exit(1)
