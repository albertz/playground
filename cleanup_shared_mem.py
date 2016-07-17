#!/usr/bin/env python3

import subprocess


def main():
    ls = subprocess.check_output(["ipcs"]).splitlines()
    count = 0
    for l in ls:
        t = l.split()
        if not t: continue
        if t[0] == b"m":
            shmid = int(t[1])
            print("shmid: %i" % shmid)
            try:
                subprocess.check_call(["ipcrm", "-m", "%i" % shmid])
                count += 1
            except subprocess.CalledProcessError:
                print("Cannot remove shmid %i." % shmid)
    print("Removed #: %i" % count)


if __name__ == "__main__":
    main()
