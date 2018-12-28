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
def dummy():
    yield

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

def elem_sep():
    sys.stdout.write("\033]116;\"\",\"\u251C\",\" \u251C\"\007")

def x():
    sys.stdout.write(" \033]114;\"\u2502 \"\007")


def main():
    print("a")
    #elem_sep()
    with block():
        #print("x", end="")
        indentation()
        hide_button()
        #print("foo")
        print("hoo", end="")
        with hide_button_span(2):
            #elem_sep()
            print("foo", end="\033]118\007")
            print("for", end="\033]118\007")
            print("fox")
            #print("fox", end="\033]118\007")
    #print()
    #print("", end="\033]118\007")
    #indentation()
    print("bar")


if __name__ == "__main__":
    main()

