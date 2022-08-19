#!/usr/bin/env python3

"""
Simple TUI editor

Python TUI?

What I want:
- simple multi-line editor
- not whole screen but only partial
- show interactive feedback. e.g. mark edits, show number of edits, show diff in separate plane or so

https://docs.python.org/3/library/curses.html
- too complex but at the same time too limited?

https://github.com/bczsalba/pytermgui (1.2k stars)
- limited, no real text editor

https://urwid.org/examples/index.html (2.5k stars)
- edit example: https://github.com/urwid/urwid/blob/master/examples/edit.py

https://github.com/prompt-toolkit/python-prompt-toolkit (7.9k stars)
- too complex...? similar as curses...

https://github.com/pfalcon/picotui (0.7k stars)
- good enough? editor: https://github.com/pfalcon/picotui/blob/master/picotui/editor.py
- another editor: https://github.com/pfalcon/picotui/blob/master/seditor.py

https://github.com/Textualize/textual (13k stars)
- async framework, I don't want that...

(Or coding some line edit by hand, should not be too difficult...?)
"""

# https://github.com/pfalcon/picotui/blob/master/seditor.py
#
# Very simple VT100 terminal text editor widget
# Copyright (c) 2015 Paul Sokolovsky, (c) 2022 Albert Zeyer
# Distributed under MIT License

# https://en.wikipedia.org/wiki/ANSI_escape_code#Terminal_input_sequences
# https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h2-The-Alternate-Screen-Buffer

import sys
import tty
import termios
import os
import signal
import re

KEY_UP = 1
KEY_DOWN = 2
KEY_LEFT = 3
KEY_RIGHT = 4
KEY_HOME = 5
KEY_END = 6
KEY_PGUP = 7
KEY_PGDN = 8
KEY_QUIT = 9
KEY_ENTER = 10
KEY_BACKSPACE = 11
KEY_DELETE = 12

KEYMAP = {
  b"\x1b[A": KEY_UP,
  b"\x1b[B": KEY_DOWN,
  b"\x1b[D": KEY_LEFT,
  b"\x1b[C": KEY_RIGHT,
  b"\x1bOH": KEY_HOME,
  b"\x1bOF": KEY_END,
  b"\x1b[1~": KEY_HOME,
  b"\x1b[4~": KEY_END,
  b"\x1b[5~": KEY_PGUP,
  b"\x1b[6~": KEY_PGDN,
  b"\x03": KEY_QUIT,
  b"\r": KEY_ENTER,
  b"\x7f": KEY_BACKSPACE,
  b"\x1b[3~": KEY_DELETE,
}


class Editor:

  def __init__(self):
    self.screen_top = 0
    self.top_line = 0
    self.cur_line = 0
    self.row = 0
    self.col = 0
    self.height = 10  # 25
    self.org_termios = None
    self.org_sig_win_ch = None
    self.content_prefix_escape = b"\x1b[30;106m"
    self.content = []
    self.status_content = [""]

  @staticmethod
  def wr(s: bytes):
    assert isinstance(s, bytes)
    os.write(1, s)

  @staticmethod
  def cls():
    Editor.wr(b"\x1b[2J")

  def goto(self, row, col):
    Editor.wr(b"\x1b[%d;%dH" % (row + 1 + self.screen_top, col + 1))

  def get_cursor_pos(self):
    self.wr(b"\x1b[6n")
    s = b""
    while True:
      s += os.read(0, 1)
      if s[-1:] == b"R":
        break
      if s[-1:] == b"\x03":
        raise KeyboardInterrupt
    res = re.match(rb".*\[(?P<y>\d*);(?P<x>\d*)R", s)
    row, col = res.groups()
    return int(row) - 1, int(col) - 1

  @staticmethod
  def clear_to_eol():
    Editor.wr(b"\x1b[0K")

  @staticmethod
  def cursor(onoff):
    if onoff:
      Editor.wr(b"\x1b[?25h")
    else:
      Editor.wr(b"\x1b[?25l")

  def set_cursor(self):
    self.goto(self.row, self.col)

  def adjust_cursor_eol(self):
    l = len(self.content[self.cur_line])
    if self.col > l:
      self.col = l

  def set_lines(self, lines: list[str]):
    self.content = lines

  @property
  def total_lines(self):
    return len(self.content)

  def set_status_content(self, lines: list[str]):
    assert self.status_content
    lines = lines or [""]
    # assume we have already drawn the screen before
    if len(lines) < len(self.status_content):
      self.goto(self.max_visible_height, 0)
      self.wr(b"\x1b[%iM" % (len(self.status_content) - len(lines)))
    elif len(lines) > len(self.status_content):
      self.goto(self.max_visible_height, 0)
      self.wr(b"\n" * (len(lines) - 1))
      self.update_editor_row_offset(self.max_visible_height + len(lines) - 1)
    self.status_content = lines
    self.update_screen_status()

  def update_screen(self):
    self.cursor(False)
    self.goto(0, 0)
    self.wr(self.content_prefix_escape)
    if self.screen_top == 0:
      self.cls()
    i = self.top_line
    for c in range(self.height):
      self.show_line(self.content[i])
      if self.screen_top > 0:
        self.clear_to_eol()
      self.wr(b"\r\n")
      i += 1
      if i == self.total_lines:
        break
    self.update_screen_status(goto=False)
    self.set_cursor()
    self.cursor(True)

  @property
  def max_visible_height(self):
    return min(self.height, self.total_lines)

  def update_screen_status(self, *, goto=True):
    if goto:
      self.cursor(False)
      self.goto(self.max_visible_height, 0)
    self.wr(b"\x1b[30;102m")
    assert self.status_content
    for c, line in enumerate(self.status_content):
      if c > 0:
        self.wr(b"\n")
      self.show_line(line)
      if self.screen_top > 0:
        self.clear_to_eol()
      self.wr(b"\r")
    self.wr(b"\x1b[0m")
    if goto:
      self.set_cursor()
      self.cursor(True)

  def update_editor_row_offset(self, cur_row=None):
    if cur_row is None:
      cur_row = self.row
    row, col = self.get_cursor_pos()
    expected_row = self.screen_top + cur_row
    self.screen_top += row - expected_row

  def update_line(self):
    self.cursor(False)
    self.wr(b"\r")
    self.wr(self.content_prefix_escape)
    self.show_line(self.content[self.cur_line])
    self.clear_to_eol()
    self.set_cursor()
    self.cursor(True)

  def show_line(self, line: str):
    self.wr(line.encode("utf8"))

  def next_line(self):
    if self.row + 1 == self.height:
      self.top_line += 1
      return True
    else:
      self.row += 1
      return False

  def prev_line(self):
    if self.row == 0:
      if self.top_line > 0:
        self.top_line -= 1
        return True
      return False
    else:
      self.row -= 1
      return False

  def handle_cursor_keys(self, key):
    if key == KEY_DOWN:
      if self.cur_line + 1 != self.total_lines:
        self.cur_line += 1
        self.adjust_cursor_eol()
        if self.next_line():
          self.update_screen()
        else:
          self.set_cursor()
    elif key == KEY_UP:
      if self.cur_line > 0:
        self.cur_line -= 1
        self.adjust_cursor_eol()
        if self.prev_line():
          self.update_screen()
        else:
          self.set_cursor()
    elif key == KEY_LEFT:
      if self.col > 0:
        self.col -= 1
        self.set_cursor()
    elif key == KEY_RIGHT:
      self.col += 1
      self.adjust_cursor_eol()
      self.set_cursor()
    elif key == KEY_HOME:
      self.col = 0
      self.set_cursor()
    elif key == KEY_END:
      self.col = len(self.content[self.cur_line])
      self.set_cursor()
    elif key == KEY_PGUP:
      self.cur_line -= self.height
      self.top_line -= self.height
      if self.top_line < 0:
        self.top_line = 0
        self.cur_line = 0
        self.row = 0
      elif self.cur_line < 0:
        self.cur_line = 0
        self.row = 0
      self.adjust_cursor_eol()
      self.update_screen()
    elif key == KEY_PGDN:
      self.cur_line += self.height
      self.top_line += self.height
      if self.cur_line >= self.total_lines:
        self.top_line = self.total_lines - self.height
        self.cur_line = self.total_lines - 1
        if self.top_line >= 0:
          self.row = self.height - 1
        else:
          self.top_line = 0
          self.row = self.cur_line
      self.adjust_cursor_eol()
      self.update_screen()
    else:
      return False
    return True

  def loop(self):
    self.init_tty()
    try:
      self.update_screen()
      while True:
        buf = os.read(0, 32)
        sz = len(buf)
        i = 0
        while i < sz:
          if buf[0] == 0x1b:
            key = buf
            i = len(buf)
          else:
            key = buf[i:i + 1]
            i += 1
          if key in KEYMAP:
            key = KEYMAP[key]
          if key == KEY_QUIT:
            return key
          if self.handle_cursor_keys(key):
            self.on_cursor_change()
            continue
          self.handle_key(key)
          self.on_cursor_change()
    finally:
      self.deinit_tty()

  def handle_key(self, key):
    l = self.content[self.cur_line]
    if key == KEY_ENTER:
      if len(self.content) < self.height:
        self.cursor(False)
        self.goto(self.max_visible_height + len(self.status_content) - 1, 0)
        self.wr(b"\r\n")  # make space for new line at end
        self.update_editor_row_offset(self.max_visible_height + len(self.status_content))
      self.content[self.cur_line] = l[:self.col]
      self.cur_line += 1
      self.content.insert(self.cur_line, l[self.col:])
      self.col = 0
      self.next_line()
      self.update_screen()
    elif key == KEY_BACKSPACE:
      if self.col:
        self.col -= 1
        l = l[:self.col] + l[self.col + 1:]
        self.content[self.cur_line] = l
        self.update_line()
      elif self.cur_line:
        self.cur_line -= 1
        self.col = len(self.content[self.cur_line])
        self.content[self.cur_line] += self.content[self.cur_line + 1]
        self.content.pop(self.cur_line + 1)
        if self.top_line > 0 and self.top_line + self.height > len(self.content):
          self.top_line -= 1
        elif self.row > 0:
          self.row -= 1
        if len(self.content) < self.height:
          self.wr(b"\x1b[1M")  # delete one line
        self.update_screen()
    elif key == KEY_DELETE:
      l = l[:self.col] + l[self.col + 1:]
      self.content[self.cur_line] = l
      self.update_line()
    else:
      l = l[:self.col] + str(key, "utf-8") + l[self.col:]
      self.content[self.cur_line] = l
      self.col += 1
      self.update_line()
    self.on_edit()

  def init_tty(self):
    self.org_termios = termios.tcgetattr(0)
    tty.setraw(0)
    self.wr(b"\x1b[?7l")  # No Auto-Wrap Mode (DECAWM)

    self._make_enough_space()

    def _on_resize(_signum, _frame):
      self.update_editor_row_offset()
      # If the colum size changed, and this wraps around existing text,
      # this is not handled correctly yet...
      # Updating the screen might be a good idea anyway.
      self.update_screen()

    self.org_sig_win_ch = signal.getsignal(signal.SIGWINCH)
    signal.signal(signal.SIGWINCH, _on_resize)

  def _make_enough_space(self):
    # assuming nothing has been printed yet
    num_lines = self.max_visible_height + len(self.status_content) - 1
    for i in range(num_lines):
      print()
    self.update_editor_row_offset(cur_row=num_lines)

  def deinit_tty(self, clear_editor=True):
    self.wr(b"\x1b[0m")
    if clear_editor:
      self.goto(0, 0)
      self.wr(b"\x1b[%iM" % (self.max_visible_height + len(self.status_content)))
    else:
      # Don't leave cursor in the middle of screen
      self.goto(self.max_visible_height + len(self.status_content) - 1, 0)
      self.wr(b"\r\n")
    termios.tcsetattr(0, termios.TCSANOW, self.org_termios)
    signal.signal(signal.SIGWINCH, self.org_sig_win_ch)

  def on_cursor_change(self):
    # Overwrite this function if you want.
    pass

  def on_edit(self):
    # Overwrite this function if you want.
    pass


def main():
  with open(sys.argv[1]) as f:
    content = f.read().splitlines()

  print("Hello editor!")

  e = Editor()
  e.set_lines(content)
  e.height = 20

  e.on_cursor_change = (
    lambda: e.set_status_content([
      "line: %d/%d" % (e.cur_line + 1, e.total_lines),
      "col: %d" % e.col,
    ]))

  e.loop()

  print("Good bye!")


if __name__ == "__main__":
  main()
