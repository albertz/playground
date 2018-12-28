#!/usr/bin/env python3

# https://github.com/PerBothner/DomTerm/
# https://github.com/PerBothner/DomTerm/issues/54
# https://domterm.org/Wire-byte-protocol.html

import sys
import contextlib


@contextlib.contextmanager
def block():
    sys.stdout.write("\033]110\007")
    yield
    sys.stdout.write("\033]111\007")

@contextlib.contextmanager
def hide_button_span(mode):
    """
    :param int mode: 1 or 2
    """
    sys.stdout.write("\033[83;%iu" % mode)
    yield
    sys.stdout.write("\033[83;0u")

def indentation():
    sys.stdout.write("\033]114;\"\u2502\"\007")

def hide_button():
    sys.stdout.write("\033[16u▶▼\033[17u")


def main():
    print("text folding:")
    with block():
        indentation()
        hide_button()
        print("<always visible>", end="")
        with hide_button_span(2):
            print("<hidden", end="\033]118\007")
            print(" hid-next-line", end="\033]118\007")
            print(" hid-final-line>")

    # something unrelated
    print("something unrelated, in a new line")


if __name__ == "__main__":
    main()

