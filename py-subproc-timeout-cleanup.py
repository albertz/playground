"""
https://github.com/rwth-i6/sisyphus/pull/191
"""

import subprocess


def main():
    while True:
        try:
            subprocess.run(["cat", "/dev/zero"], timeout=0.01)
        except subprocess.TimeoutExpired:
            print("TimeoutExpired")


if __name__ == "__main__":
    main()
