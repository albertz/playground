#!/usr/bin/env python3

import sys
import os

# http://www.talisman.org/~erlkonig/documents/xterm-color-queries/
# https://gist.github.com/blueyed/c8470c2aad3381c33ea3
# Vim: https://github.com/vim/vim/blob/05c00c038bc16e862e17f9e5c8d5a72af6cf7788/src/option.c#L3974
# https://stackoverflow.com/questions/38287889/how-does-vim-guess-background-color-on-xterm
# TODO: also check Emacs... (background-mode)
# https://stackoverflow.com/questions/2507337/is-there-a-way-to-determine-a-terminals-background-color
# https://invisible-island.net/xterm/ctlseqs/ctlseqs.html (dynamic colors / Request Termcap/Terminfo String)
# https://unix.stackexchange.com/questions/245378/common-environment-variable-to-set-dark-or-light-terminal-background
# http://thrysoee.dk/xtermcontrol/
# env COLORFGBG (Rxvt)
# https://github.com/rocky/bash-term-background/blob/master/term-background.sh
# https://bugzilla.gnome.org/show_bug.cgi?id=733423
# xtermcontrol --get-bg

# https://github.com/neovim/neovim/issues/2764
# /*
#  * Return "dark" or "light" depending on the kind of terminal.
#  * This is just guessing!  Recognized are:
#  * "linux"         Linux console
#  * "screen.linux"   Linux console with screen
#  * "cygwin"        Cygwin shell
#  * "putty"         Putty program
#  * We also check the COLORFGBG environment variable, which is set by
#  * rxvt and derivatives. This variable contains either two or three
#  * values separated by semicolons; we want the last value in either
#  * case. If this value is 0-6 or 8, our background is dark.
#  */
# static char_u *term_bg_default(void)
# {
#   char_u      *p;
#
#   if (STRCMP(T_NAME, "linux") == 0
#       || STRCMP(T_NAME, "screen.linux") == 0
#       || STRCMP(T_NAME, "cygwin") == 0
#       || STRCMP(T_NAME, "putty") == 0
#       || ((p = (char_u *)os_getenv("COLORFGBG")) != NULL
#           && (p = vim_strrchr(p, ';')) != NULL
#           && ((p[1] >= '0' && p[1] <= '6') || p[1] == '8')
#           && p[2] == NUL))
#     return (char_u *)"dark";
#   return (char_u *)"light";
# }
#
# About COLORFGBG env:
# Out of quite a few terminals I've just tried on linux, only urxvt and konsole set it (the ones that don't: xterm,
# st, terminology, pterm).
# Konsole and Urxvt use different syntax and semantics, i.e. for me konsole sets it to
# "0;15" (even though I use the "Black on Light Yellow" color scheme - so why not "default" instead of "15"?),
# whereas my urxvt sets it to "0;default;15" (it's actually black on white - but why three fields?).  So in neither
# of these two does the value match your specification.


def is_light_terminal_background():
  """
  :return: whether we have a light Terminal background color, or None if unknown
  :rtype: bool|None
  """
  if os.environ.get("COLORFGBG", None):
    parts = os.environ["COLORFGBG"].split(";")
    try:
      last_number = int(parts[-1])
      if 0 <= last_number <= 6 or last_number == 8:
        return False  # dark
      else:
        return True
    except ValueError:  # not an integer?
      pass
  return None  # unknown (and bool(None) == False, i.e. expect dark by default)


print("Light terminal background:", is_light_terminal_background())

#print('\033]11;?\007')


