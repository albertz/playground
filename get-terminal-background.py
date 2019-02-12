#!/usr/bin/env python3

import subprocess
import sys

# http://www.talisman.org/~erlkonig/documents/xterm-color-queries/
# https://gist.github.com/blueyed/c8470c2aad3381c33ea3
# Vim: https://github.com/vim/vim/blob/05c00c038bc16e862e17f9e5c8d5a72af6cf7788/src/option.c#L3974
# TODO: also check Emacs... (background-mode)
# https://stackoverflow.com/questions/2507337/is-there-a-way-to-determine-a-terminals-background-color
# https://invisible-island.net/xterm/ctlseqs/ctlseqs.html (dynamic colors / Request Termcap/Terminfo String)
# https://unix.stackexchange.com/questions/245378/common-environment-variable-to-set-dark-or-light-terminal-background
# http://thrysoee.dk/xtermcontrol/
# env COLORFGBG (Rxvt)
# https://github.com/rocky/bash-term-background/blob/master/term-background.sh

print('\033]11;?\007')
#sys.stdin.read()

#out = subprocess.check_output(["python", "-c", "print('\033]11;?\007')"])
#print(repr(out))
