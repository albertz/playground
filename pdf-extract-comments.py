#!/usr/bin/env python3

"""
Extract comments from PDF files.


https://stackoverflow.com/questions/68266356/extracting-comments-annotations-from-pdf-sequentially-python

"""

import argparse
import PyPDF2  # pip install PyPDF2
import fitz  # brew install mupdf; pip install pymupdf
import subprocess


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

    # /Rect is left/bottom/right/top
    rect = [float(v) for v in obj['/Rect']]
    center = float(rect[0] + rect[2]) / 2, float(rect[1] + rect[3]) / 2
    obj["<synctex>"] = f"{page_num + 1}:{center[0]:.2f}:{center[1]:.2f}"
    # subprocess.getoutput()

    fitz_page = fitz_doc[page_num]
    assert isinstance(fitz_page, fitz.Page)

    # PyPDF counts y from left-bottom point of the page,
    # while fitz counts y from left-top.

    # '/QuadPoints': [103.78458, 532.89081, 139.10219, 532.89081, 103.78458, 521.98169, 139.10219, 521.98169]
    # xy pairs: left-upper, right-upper, left-bottom, right-bottom
    def _translate_quad_points(pts):
      # return fitz left,top,right,bottom
      pts = [float(v) for v in pts]
      return pts[0] - 1, fitz_page.rect[3] - pts[1], pts[2] + 1, fitz_page.rect[3] - pts[5]

    # '/Rect': [134.13168, 520.65894, 144.07268, 528.75903]
    # /Rect is left/bottom/right/top
    # noinspection PyShadowingNames
    def _translate_extend_rect(rect):
      # return fitz left,top,right,bottom
      return (
        rect[0] - 20, fitz_page.rect[3] - rect[3] - 10,
        rect[2] + 20, fitz_page.rect[3] - rect[1] + 10)

    if "/QuadPoints" in obj:
      clip = _translate_quad_points(obj["/QuadPoints"])
    else:
      clip = _translate_extend_rect(rect)

    obj["<text-ctx>"] = fitz_page.get_text(clip=clip)

    return obj

  def _handle_page(page_num):
    print("*** page:", page_num)
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

  if args.page is not None:
    _handle_page(args.page)
  else:
    for i in range(pypdf2_doc.numPages):
      _handle_page(i)

  print(values_by_key)


if __name__ == "__main__":
  main()


