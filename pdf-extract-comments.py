#!/usr/bin/env python3

"""
Extract comments from PDF files.


https://stackoverflow.com/questions/68266356/extracting-comments-annotations-from-pdf-sequentially-python

"""

import argparse
import PyPDF2  # pip install PyPDF2
import fitz  # brew install mupdf; pip install pymupdf
import subprocess


CaretSym = "‸"
_Debug = False


def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("pdf_file", help="PDF file to extract comments from")
  arg_parser.add_argument("--page", type=int)
  args = arg_parser.parse_args()

  pypdf2_doc = PyPDF2.PdfReader(args.pdf_file)
  fitz_doc = fitz.open(args.pdf_file)

  values_by_key = {}

  def _cleanup_obj(page_num, obj):
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
    fitz_page = fitz_doc[page_num]
    assert isinstance(fitz_page, fitz.Page)

    # /Rect is left/bottom/right/top
    rect = [float(v) for v in obj['/Rect']]
    center = float(rect[0] + rect[2]) / 2, float(rect[1] + rect[3]) / 2
    obj["<synctex>"] = f"{page_num + 1}:{center[0]:.2f}:{fitz_page.rect[3] - center[1]:.2f}"
    # subprocess.getoutput()

    # '/QuadPoints': [103.78458, 532.89081, 139.10219, 532.89081, 103.78458, 521.98169, 139.10219, 521.98169]
    # xy pairs: left-upper, right-upper, left-bottom, right-bottom
    def _translate_quad_points(pts, o=0):
      # return fitz left,top,right,bottom
      pts = [float(v) for v in pts]
      return pts[o + 0] - 1, fitz_page.rect[3] - pts[o + 1], pts[o + 2] + 1, fitz_page.rect[3] - pts[o + 5]

    def _get_quad_points_txt():
      pts = obj["/QuadPoints"]
      assert len(pts) % 8 == 0, f"{pts=}, {len(pts)=}"
      n = len(pts) // 8
      lines = []
      for i in range(n):
        clip = _translate_quad_points(pts, i * 8)
        lines.append(fitz_page.get_text(clip=clip).rstrip("\n"))
      return " ".join(lines)

    ctx_w = 60

    # '/Rect': [134.13168, 520.65894, 144.07268, 528.75903]
    # /Rect is left/bottom/right/top
    # noinspection PyShadowingNames
    def _translate_extend_rect(rect):
      # return fitz left,top,right,bottom
      return (
        rect[0] - ctx_w, fitz_page.rect[3] - rect[3] - 7,
        rect[2] + ctx_w, fitz_page.rect[3] - rect[1])

    def _get_rect_text_ctx(o=None):
      if "/QuadPoints" in obj:
        if o is None:
          assert len(obj["/QuadPoints"]) % 8 == 0
          if len(obj["/QuadPoints"]) > 8:
            lines = []
            for i in range(len(obj["/QuadPoints"]) // 8):
              lines.append(_get_rect_text_ctx(o=i * 8))
            return " ".join(lines)
          o = 0
        rect_ = list(_translate_quad_points(obj["/QuadPoints"], o=o))
        rect_[0] -= ctx_w
        rect_[2] += ctx_w
      else:
        assert o is None
        rect_ = _translate_extend_rect(rect)
      full_txt = fitz_page.get_text(clip=rect_)
      if not full_txt.strip():
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
      obj["<text>"] = _get_quad_points_txt()
    obj["<text-ctx>"] = _get_rect_text_ctx()

    if not _Debug:
      obj.pop("/QuadPoints", None)
      obj.pop("/Rect", None)

    return obj

  def _handle_page(page_num):
    print("*** page:", page_num + 1)  # in code 0-indexed, while all apps (esp PDF viewers) use 1-index base
    pypdf2_page = pypdf2_doc.pages[page_num]
    assert isinstance(pypdf2_page, PyPDF2.PageObject)
    if "/Annots" not in pypdf2_page:
      return
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
      _cleanup_obj(page_num, obj)
      if obj.get('/IRT'):
        visited_irt.add(obj['/IRT'].idnum)
        irt_obj = dict(obj['/IRT'].get_object())
        _cleanup_obj(page_num, irt_obj)
        # Some further cleanup
        for name in ['/Subj', '/Subtype', '/T', '/IT']:
          irt_obj.pop(name, None)
        obj['/IRT'] = irt_obj
      for key in ['/Subj', '/Subtype', '/T']:
        if key not in obj:
          continue
        if key not in values_by_key:
          values_by_key[key] = set()
        values_by_key[key].add(obj[key])
      print(annot, obj)

  def _handle_all_pages():
    for i in range(pypdf2_doc.numPages):
      _handle_page(i)

  if args.page is not None:
    _handle_page(args.page - 1)
  else:
    _handle_all_pages()

  print(values_by_key)


if __name__ == "__main__":
  main()


