#!/usr/bin/env python3

# https://github.com/PerBothner/DomTerm/
# https://github.com/PerBothner/DomTerm/issues/54
# https://domterm.org/Wire-byte-protocol.html

import sys


def start_block():
    sys.stdout.write("\033]110\007")

def end_block():
    sys.stdout.write("\033]111\007")

def indentation():
    sys.stdout.write("\033]114;\"\u2502\"\007")

def hide_button():
    sys.stdout.write("\033[16u\u25BC\033[17u")

def elem_sep():
    sys.stdout.write("\033]116;\"\",\"\u251C\",\" \u251C\"\007")

def x():
    sys.stdout.write(" \033]114;\"\u2502 \"\007")


def main():
    print("a")
    #elem_sep()
    start_block()
    #print("x", end="")
    #indentation()
    hide_button()
    #print("foo")
    #print("foo", end="")
    print("foo", end="\033]118\007")
    start_block()
    #elem_sep()
    print("for", end="\033]118\007")
    print("fox")
    #print("fox", end="\033]118\007")
    end_block()
    end_block()
    #print()
    #print("", end="\033]118\007")
    #indentation()
    print("bar")


if __name__ == "__main__":
    main()

