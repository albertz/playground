#!/usr/bin/env python3

"""
Extract comments from PDF files.


https://stackoverflow.com/questions/68266356/extracting-comments-annotations-from-pdf-sequentially-python

"""

from __future__ import annotations
from typing import Tuple
import argparse
import PyPDF2  # pip install PyPDF2
import fitz  # brew install mupdf; pip install pymupdf
import subprocess
from dataclasses import dataclass, field
import os
import better_exchook


CaretSym = "‸"
_Debug = False


def main():
  global _Debug
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("pdf_file", help="PDF file to extract comments from")
  arg_parser.add_argument("--synctex", help="(PDF) file for synctex")
  arg_parser.add_argument("--page", type=int)
  arg_parser.add_argument("--debug", action="store_true")
  args = arg_parser.parse_args()
  if args.debug:
    _Debug = True

  env = Env(args)

  def _handle_page(page_num):
    Page(env=env, page_num=page_num)

  def _handle_all_pages():
    for i in range(env.pypdf2_doc.numPages):
      _handle_page(i)

  if args.page is not None:
    _handle_page(args.page - 1)
  else:
    _handle_all_pages()


class Env:
  def __init__(self, args):
    self.args = args
    self.pypdf2_doc = PyPDF2.PdfReader(args.pdf_file)
    self.fitz_doc = fitz.open(args.pdf_file)


class Page:
  def __init__(self, *, env: Env, page_num: int):
    print("*** page:", page_num + 1)  # in code 0-indexed, while all apps (esp PDF viewers) use 1-index base
    pypdf2_page = env.pypdf2_doc.pages[page_num]
    assert isinstance(pypdf2_page, PyPDF2.PageObject)
    if "/Annots" not in pypdf2_page:
      return
    fitz_page = env.fitz_doc[page_num]
    assert isinstance(fitz_page, fitz.Page)

    page_txt = fitz_page.get_text()
    assert isinstance(page_txt, str)
    page_txt_num_lines = page_txt.count("\n")
    page_txt = page_txt.replace("\n", " ")
    page_txt = _text_replace(page_txt)
    print(page_txt)
    self._page_txt = page_txt

    self._edits_tex_line_start_end = None
    self._edits_page_to_tex = None
    tex_file = None
    tex_center_line = None
    if env.args.synctex:
      page_synctex = f"{page_num + 1}:{fitz_page.rect[2] / 2:.2f}:{fitz_page.rect[3] / 2:.2f}"
      synctex_cmd = ["synctex", "edit", "-o", f"{page_synctex}:{env.args.synctex}"]
      for line in subprocess.check_output(synctex_cmd).splitlines():
        if line.startswith(b"Input:"):
          tex_file = line[len("Input:"):].decode("utf8")
        elif line.startswith(b"Line:"):
          tex_center_line = int(line[len("Line:"):].decode("utf8"))
      print("Tex file:", tex_file, ", center line:", tex_center_line)
      print(f"--- {tex_file}\tv1")
      print(f"+++ {tex_file}.new\tv2")
      lines = open(tex_file, encoding="utf8").readlines()
      self._tex_lines = lines
      w = page_txt_num_lines * 2
      line_start = max(tex_center_line - w, 0)
      line_end = min(tex_center_line + w + 1, len(lines))
      tex_selected_txt = "".join(lines[line_start:line_end])
      tex_selected_txt, tex_to_simplify_tex_edits = tex_simplify(tex_selected_txt)
      page_to_simplify_tex_edits = levenshtein_alignment(page_txt, tex_selected_txt)
      self._edits_tex_line_start_end = (line_start, line_end)
      simplify_to_tex_edits = tex_to_simplify_tex_edits.inverse()
      if _Debug:
        print("Simplified tex edits:")
        simplify_to_tex_edits.dump()
      edits = page_to_simplify_tex_edits.compose(simplify_to_tex_edits)
      self._edits_page_to_tex = edits
      if _Debug:
        print("Page to tex edits:")
        edits.dump()
    self._tex_file = tex_file

    visited_irt = set()
    for annot in pypdf2_page["/Annots"]:
      obj = dict(annot.get_object())
      if obj["/Subtype"] == "/Link":
        continue
      if obj['/Subtype'] == '/Popup':  # popup object itself does not have the information
        continue
      # /Subtype == /Caret could be a duplicate but not always.
      # Use simple generic rule: Skip if we have seen it via /IRT before.
      if annot.idnum in visited_irt:
        continue
      if obj.get('/IRT'):
        visited_irt.add(obj['/IRT'].idnum)
        irt_obj = dict(obj['/IRT'].get_object())
        self._cleanup_obj(fitz_page, irt_obj, is_irt=True)
        # Some further cleanup
        for name in ['/Subj', '/Subtype', '/T', '/IT']:
          irt_obj.pop(name, None)
        obj['/IRT'] = irt_obj
      self._cleanup_obj(fitz_page, obj)
      print(annot, obj)

      if obj.get("<edit>"):
        self.handle_page_edit(*obj.get("<edit>"))

  def _cleanup_obj(self, fitz_page, obj, *, is_irt: bool = False):
    # Drop some keys that are not useful for us.
    for name in [
      '/AP', "/F", "/N", "/NM", '/CreationDate', '/M', "/Open", '/BM',
      "/P",  # page ref
      "/RC",  # XML/HTML, but /Contents has the same info
      "/Popup", "/C", "/Type", '/RD',
    ]:
      obj.pop(name, None)

    # PyPDF counts y from left-bottom point of the page,
    # while fitz counts y from left-top.

    # /Rect is left/bottom/right/top
    rect = [float(v) for v in obj['/Rect']]
    center = float(rect[0] + rect[2]) / 2, float(rect[1] + rect[3]) / 2
    # obj["<synctex>"] = f"{page_num + 1}:{center[0]:.2f}:{fitz_page.rect[3] - center[1]:.2f}"
    # subprocess.getoutput()

    # '/QuadPoints': [103.78458, 532.89081, 139.10219, 532.89081, 103.78458, 521.98169, 139.10219, 521.98169]
    # xy pairs: left-upper, right-upper, left-bottom, right-bottom
    def _translate_quad_points(pts, o=0):
      # return fitz left,top,right,bottom
      pts = [float(v) for v in pts]
      return pts[o + 0] - 1, fitz_page.rect[3] - pts[o + 1] - 1, pts[o + 2] + 1, fitz_page.rect[3] - pts[o + 5] + 1

    def _get_quad_points_txt():
      pts = obj["/QuadPoints"]
      assert len(pts) % 8 == 0, f"{pts=}, {len(pts)=}"
      n = len(pts) // 8
      lines = []
      for i in range(n):
        clip = _translate_quad_points(pts, i * 8)
        lines.append(fitz_page.get_text(clip=clip).rstrip("\n"))
      return " ".join(lines)

    # '/Rect': [134.13168, 520.65894, 144.07268, 528.75903]
    # /Rect is left/bottom/right/top
    # noinspection PyShadowingNames
    def _translate_extend_rect(rect, ctx_w: float = 0):
      # return fitz left,top,right,bottom
      return (
        rect[0] - ctx_w, fitz_page.rect[3] - rect[3] - 7,
        rect[2] + ctx_w, fitz_page.rect[3] - rect[1])

    def _get_rect_text_ctx(*, o=None, ctx_w: float):
      if "/QuadPoints" in obj:
        if o is None:
          assert len(obj["/QuadPoints"]) % 8 == 0
          if len(obj["/QuadPoints"]) > 8:
            lines = []
            for i in range(len(obj["/QuadPoints"]) // 8):
              lines.append(_get_rect_text_ctx(o=i * 8, ctx_w=ctx_w))
            return " ".join(lines)
          o = 0
        rect_ = list(_translate_quad_points(obj["/QuadPoints"], o=o))
        rect_[0] -= ctx_w
        rect_[2] += ctx_w
      else:
        assert o is None
        rect_ = _translate_extend_rect(rect, ctx_w=ctx_w)
      full_txt = fitz_page.get_text(clip=rect_)
      if not full_txt.strip():
        # empty? unexpected...
        assert "/QuadPoints" not in obj and obj.get("/Subj") not in {"StrikeOut", "Replace Text"}
        return ""
      full_txt = full_txt.replace("\n", " ").rstrip()
      if obj.get('/Subtype') == '/Caret':
        p = center[0]
        for w in [0, 1, 2, 3, 4, 5, 2]:
          left_txt = (
            fitz_page.get_text(clip=(rect_[0], rect_[1], p + w, rect_[3])).rstrip("\n").replace("\n", " "))
          right_txt = fitz_page.get_text(clip=(p - w, rect_[1], rect_[2], rect_[3])).replace("\n", " ").rstrip()
          if full_txt == left_txt + right_txt:
            return left_txt + CaretSym + right_txt
          if left_txt[-1:] == right_txt[:1] and full_txt == left_txt + right_txt[1:]:
            return left_txt + CaretSym + right_txt[1:]
        assert full_txt == left_txt + right_txt, f"{full_txt!r} != {left_txt!r} + {right_txt!r}; obj {obj!r}"
        return left_txt + CaretSym + right_txt
      return full_txt

    if "/QuadPoints" in obj:
      obj["<text>"] = _text_replace(_get_quad_points_txt())
    default_ctx_w = 60
    obj["<text-ctx>"] = _text_replace(_get_rect_text_ctx(ctx_w=default_ctx_w))

    edit_pos_range = None
    edit = None
    if not is_irt and (obj.get("<text>") or CaretSym in obj["<text-ctx>"]):
      if obj.get("<text>"):
        txt = obj.get("<text>")
        c = self._page_txt.count(txt)
        assert c >= 1  # if not, maybe newline or other whitespace thing?
      else:
        txt = CaretSym
      c2 = self._page_txt.count(obj["<text-ctx>"].replace(CaretSym, ""))
      assert c2 >= 1  # if not, maybe newline or other whitespace thing?
      assert c2 == 1  # just not implemented otherwise
      page_ctx_pos = self._page_txt.find(obj["<text-ctx>"].replace(CaretSym, ""))
      assert page_ctx_pos >= 0
      txt_ctx = obj["<text-ctx>"]
      ctx_w = default_ctx_w
      while True:
        c3 = obj["<text-ctx>"].count(txt_ctx)
        assert c3 == 1  # not implemented otherwise
        c4 = txt_ctx.count(txt)
        assert c4 >= 1  # sth wrong?
        if c4 == 1:
          pos_ = obj["<text-ctx>"].find(txt_ctx) + txt_ctx.find(txt)
          pos = page_ctx_pos + pos_
          if obj.get("/Subj") in {"StrikeOut", "Replace Text"}:
            # Just nicer visual representation of text-ctx with strikeout effect.
            strike_out_txt = '\u0336'.join(obj["<text>"]) + '\u0336'
            obj["<text-ctx>"] = (
                obj["<text-ctx>"][:pos_] + strike_out_txt + obj["<text-ctx>"][pos_ + len(obj["<text>"]):])
          break
        ctx_w -= 5
        assert ctx_w > 0
        txt_ctx = _text_replace(_get_rect_text_ctx(ctx_w=ctx_w))
      if obj.get("/Subj") == "StrikeOut":  # just delete
        edit_pos_range = (pos, pos + len(obj["<text>"]))
        edit = Edit(delete=obj["<text>"])
      elif obj.get("/Subj") == "Replace Text":  # replace
        assert obj.get("/IRT")
        assert CaretSym in obj["/IRT"]["<text-ctx>"]
        edit_pos_range = (pos, pos + len(obj["<text>"]))
        edit = Edit(delete=obj["<text>"], insert=obj["/IRT"]["/Contents"])
      elif obj.get('/Subj') == 'Insert Text':  # insert (no delete/replace)
        edit_pos_range = (pos, pos)
        edit = Edit(insert=obj["/Contents"])
    if edit:
      obj["<edit>"] = edit_pos_range + (edit,)
      # Some further cleanup, as those info is in the edit.
      if not _Debug:
        for name in ["/Contents", "/Subj", "/Subtype", "<text>", '/IRT', '/IT', '/RT']:
          obj.pop(name, None)

    if not _Debug:
      obj.pop("/QuadPoints", None)
      obj.pop("/Rect", None)

    return obj

  def handle_page_edit(self, page_pos_start: int, page_pos_end: int, page_edit: Edit):
    """
    find pos in latex code (straightforward).
    translate edit.
    """
    latex_start_line, latex_start_line_pos, latex_start_edit_idx, latex_start_exact = (
      self.translate_page_pos_to_latex_line_pos(page_pos_start))
    latex_end_line, latex_end_line_pos, latex_end_edit_idx, latex_end_exact = (
      self.translate_page_pos_to_latex_line_pos(page_pos_end))
    assert latex_start_exact and latex_end_exact, (
      f"{latex_start_line} {page_edit} {self._edits_page_to_tex.edits[latex_start_edit_idx]}")
    # https://en.wikipedia.org/wiki/Diff#Unified_format
    lines = []
    proposed_num_ctx_lines = 3
    num_lines_source = 0
    num_lines_target = 0
    for i in range(latex_start_line - proposed_num_ctx_lines, latex_start_line):
      lines.append(" " + self._tex_lines[i])
      num_lines_source += 1
      num_lines_target += 1
    for i in range(latex_start_line, latex_end_line + 1):
      lines.append("-" + self._tex_lines[i])
      num_lines_source += 1
    line = self._tex_lines[latex_start_line][:latex_start_line_pos]
    insert = page_edit.insert
    if "[" in insert:
      p1 = insert.find("[")
      p2 = insert.rfind("]")
      assert 0 <= p1 < p2
      insert = insert[:p1] + insert[p2 + 1:]
    insert = insert.replace("#", " ")
    assert "\n" not in insert  # not implemented
    line += insert
    line += self._tex_lines[latex_end_line][latex_end_line_pos:]
    lines.append("+" + line)
    num_lines_target += 1
    for i in range(latex_end_line + 1, latex_end_line + proposed_num_ctx_lines + 1):
      lines.append(" " + self._tex_lines[i])
      num_lines_source += 1
      num_lines_target += 1
    new_latex_start_line = latex_start_line  # TODO ... or maybe ignore
    print(f"@@ -{latex_start_line},{num_lines_source} +{new_latex_start_line},{num_lines_target} @@")
    for line in lines:
      print(line, end="")
    # TODO sth is wrong, but not always...

  def translate_page_pos_to_latex_line_pos(self, page_pos: int) -> Tuple[int, int, int, bool]:
    """
    Translate page pos to latex line pos.

    :return: latex line, pos in that line, edit index, exact position?
    """
    cur_page_pos = 0
    cur_latex_line, _ = self._edits_tex_line_start_end
    cur_latex_line_pos = 0
    edit_index = 0
    for edit in self._edits_page_to_tex.edits:
      if cur_page_pos == page_pos:
        break
      break_now = False
      if cur_page_pos + len(edit.delete) > page_pos:
        if edit.insert == edit.delete:
          edit = Edit(
            delete=edit.delete[:page_pos - cur_page_pos],
            insert=edit.insert[:page_pos - cur_page_pos])
          assert cur_page_pos + len(edit.delete) == page_pos
          break_now = True
        else:
          break
      cur_page_pos += len(edit.delete)
      cur_latex_line += edit.insert.count("\n")
      last_newline = edit.insert.rfind("\n")
      if last_newline >= 0:
        cur_latex_line_pos = len(edit.insert) - last_newline - 1
      else:
        cur_latex_line_pos += len(edit.delete)
      if break_now:
        assert cur_page_pos == page_pos
        break
      edit_index += 1
    return cur_latex_line, cur_latex_line_pos, edit_index, cur_page_pos == page_pos


def _text_replace(page_txt: str) -> str:
  page_txt = page_txt.replace("¨a", "ä")
  page_txt = page_txt.replace("¨o", "ö")
  page_txt = page_txt.replace("¨u", "ü")
  page_txt = page_txt.replace("´e", "é")
  page_txt = page_txt.replace("´a", "á")
  page_txt = page_txt.replace("´s", "ś")
  page_txt = page_txt.replace("ﬁ", "fi")
  page_txt = page_txt.replace("ﬀ", "ff")
  return page_txt


@dataclass(frozen=True)
class Edit:
  """
  Edit.
  """
  insert: str = ""  # insert or replace if len(delete) = len(insert)
  delete: str = ""

  @classmethod
  def make_empty(cls) -> Edit:
    """
    :return: empty
    """
    return Edit()

  def char_len(self):
    """
    :return: number of edits on char level
    """
    if self.insert == self.delete:
      return 0
    return max(len(self.delete), len(self.insert))

  def merge_class(self):
    """
    :return: whatever such that if self.merge_class() == other.merge_class(), we should merge them
    """
    return bool(self.insert), bool(self.delete), self.insert == self.delete

  def __add__(self, other: Edit) -> Edit:
    return Edit(insert=self.insert + other.insert, delete=self.delete + other.delete)


@dataclass
class EditList:
  """
  list of edits. also keeps track of total num of char edits
  """
  edits: list[Edit] = field(default_factory=list)
  char_len: int = 0

  @classmethod
  def make_empty(cls) -> EditList:
    """
    :return: empty
    """
    return EditList(edits=[], char_len=0)

  def add_left(self, edit: Edit):
    """
    Modify inplace, edit + self.
    """
    edit0 = self.edits[0] if self.edits else None
    if edit0 and edit0.merge_class() == edit.merge_class():
      edit0 = edit + edit0
      self.edits[0] = edit0
    else:
      self.edits.insert(0, edit)
    if edit.insert != edit.delete:
      self.char_len += edit.char_len()

  def add_right(self, edit: Edit):
    """
    Modify inplace, self + edit.
    """
    last_edit = self.edits[-1] if self.edits else None
    if last_edit and last_edit.merge_class() == edit.merge_class():
      last_edit = last_edit + edit
      self.edits[-1] = last_edit
    else:
      self.edits.append(edit)
    if edit.insert != edit.delete:
      self.char_len += edit.char_len()

  def inverse(self) -> EditList:
    """
    inserts become deletes, deletes become inserts.
    """
    out = EditList.make_empty()
    for edit in self.edits:
      out.add_right(Edit(insert=edit.delete, delete=edit.insert))
    return out

  def compose(self, other: EditList) -> EditList:
    """
    Composes self and other, like both edits are applied.
    Note that this is not simply adding the list of edits.
    """
    out = EditList.make_empty()
    self_edits = self.edits.copy()
    other_edits = other.edits.copy()

    def _next_longest_matching_other_edit(m: str) -> Edit:
      if not other_edits:
        return Edit.make_empty()
      next_other_edit = other_edits.pop(0)
      if m.startswith(next_other_edit.delete):
        return next_other_edit
      prefix = os.path.commonprefix([m, next_other_edit.delete])
      assert prefix != next_other_edit.delete  # otherwise m.startswith(...)
      assert prefix  # they should match
      # need to split it
      part1 = Edit(insert="", delete=prefix)
      part2 = Edit(insert=next_other_edit.insert, delete=next_other_edit.delete[len(prefix):])
      other_edits.insert(0, part2)
      return part1

    while self_edits:
      self_edit = self_edits.pop(0)
      if not self_edit.insert:
        out.add_right(self_edit)
        continue
      other_edit = _next_longest_matching_other_edit(self_edit.insert)
      if self_edit.insert == other_edit.delete:
        out.add_right(Edit(insert=other_edit.insert, delete=self_edit.delete))
      else:
        # TODO sth is broken and/or suboptimal here... we should try to leave unchanged code as much as possible
        assert self_edit.insert.startswith(other_edit.delete)
        # Split it.
        out.add_right(Edit(insert=other_edit.insert, delete=""))
        part2_ = Edit(insert=self_edit.insert[len(other_edit.delete):], delete=self_edit.delete)
        self_edits.insert(0, part2_)

    # Any remaining? They should not delete anything anymore but only add something.
    for other_edit in other_edits:
      assert not other_edit.delete
      out.add_right(other_edit)
    return out

  def dump(self):
    """
    To stdout
    """
    for edit in self.edits:
      if edit.insert == edit.delete:
        print(f"={edit.insert!r}")
      elif edit.insert and edit.delete:
        print(f"-{edit.delete!r} +{edit.insert!r}")
      else:
        if edit.delete:
          print(f"-{edit.delete!r}")
        if edit.insert:
          print(f"+{edit.insert!r}")


def levenshtein_alignment(source: str, target: str) -> EditList:
  """
  get minimal list of edits to transform source to target
  """
  # loop over target pos
  # Build matrix target * source.
  # Start with initial for each source pos
  # list of list of tuple (num total edits, backref, del, add)
  # backref: 0: above, 1: left-above, 2: left
  m = [(0, -1, "", "")]  # initial
  for j in range(1, len(source) + 1):
    m.append((m[-1][0] + 1, 0, source[j - 1], ""))  # del
  ms = [m]
  for i in range(1, len(target) + 1):
    m = [(ms[-1][0][0] + 1, 2, "", target[i - 1])]  # j=0, add
    for j in range(1, len(source) + 1):
      # 3 cases:
      opts = (
        (m[-1][0] + 1, 0, source[j - 1], ""),  # del
        (ms[-1][j - 1][0] + int(source[j - 1] != target[i - 1]), 1, source[j - 1], target[i - 1]),  # replace or keep
        (ms[-1][j][0] + 1, 2, "", target[i - 1])  # add
      )
      m.append(min(opts))
    ms.append(m)
  # build edit list, going through backrefs
  i, j = len(target), len(source)
  total_num_edits = ms[i][j][0]
  ls = EditList.make_empty()
  while i > 0 or j > 0:
    _, backref, del_, add = ms[i][j]
    ls.add_left(Edit(insert=add, delete=del_))
    if backref == 0:
      j -= 1
    elif backref == 1:
      i -= 1
      j -= 1
    elif backref == 2:
      i -= 1
    else:
      assert backref in {0, 1, 2}, f"invalid backref {backref}"
  assert ls.char_len == total_num_edits
  return ls


def tex_simplify(tex: str) -> (str, EditList):
  """
  :param tex: code
  :return: (simplified tex, edit from input tex to simplified tex)
  """
  edits = EditList()
  out = ""
  pos = 0
  tex_len = len(tex)
  while pos < tex_len:
    if tex[pos] == "%":  # comment
      pos_ = pos + 1
      while tex[pos_] != "\n":
        pos_ += 1
      # delete the command
      edits.add_right(Edit(insert="", delete=tex[pos:pos_ + 1]))
      pos = pos_ + 1
    elif tex[pos] == "\\":
      pos_ = pos + 1
      while tex[pos_].isalpha():
        pos_ += 1
      cmd = tex[pos + 1:pos_]
      if cmd in {"Gls", "gls", "Glspl", "glspl"}:
        # Replace by abbreviation.
        assert tex[pos_] == "{"
        pos__ = pos_ + 1
        while tex[pos__] != "}":
          assert tex[pos__].isalpha() or tex[pos__] in "-", f"invalid cmd {tex[pos:pos__ + 1]!r}"
          pos__ += 1
        arg = tex[pos_ + 1:pos__]
        replace = {
          "hybrid-hmm": "hybrid NN-HMM"
        }.get(arg, arg.upper())  # good fallback heuristic
        if cmd.endswith("pl"):
          replace += "s"
        out += replace
        edits.add_right(Edit(insert=replace, delete=tex[pos:pos__ + 1]))
        pos = pos__ + 1
      # any commands with a simple argument which we can completely remove
      elif cmd in {"glsreset", "label"}:
        assert tex[pos_] == "{"
        pos__ = pos_ + 1
        while tex[pos__] != "}":
          assert tex[pos__] != "{"
          pos__ += 1
        edits.add_right(Edit(insert="", delete=tex[pos:pos__ + 1]))
        pos = pos__ + 1
      else:
        # delete the command
        edits.add_right(Edit(insert="", delete=tex[pos:pos_]))
        pos = pos_
    else:
      out += tex[pos]
      edits.add_right(Edit(insert=tex[pos], delete=tex[pos]))
      pos += 1
  return out, edits


if __name__ == "__main__":
  better_exchook.install()
  main()


