#!/usr/bin/env python3

"""
Extract comments from PDF files.


https://stackoverflow.com/questions/68266356/extracting-comments-annotations-from-pdf-sequentially-python

"""

import argparse
import PyPDF2  # pip install PyPDF2
import fitz  # brew install mupdf; pip install pymupdf
import subprocess


def _cleanup_obj(obj):
  # Drop some keys that are not useful for us.
  for name in [
    '/AP', "/F", "/N", "/NM", '/CreationDate', '/M', "/Open", '/BM',
    "/P",  # page ref
    "/RC",  # XML/HTML, but /Contents has the same info
    "/Popup", "/C", "/Type", '/RD',
  ]:
    obj.pop(name, None)
  return obj


def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("pdf_file", help="PDF file to extract comments from")
  arg_parser.add_argument("--page", type=int)
  args = arg_parser.parse_args()

  reader = PyPDF2.PdfReader(args.pdf_file)

  values_by_key = {}

  def _handle_page(page_num):
    page = reader.pages[page_num]
    assert isinstance(page, PyPDF2.PageObject)
    print("*** page:", page_num)
    if "/Annots" not in page:
      return
    visited_irt = set()
    for annot in page["/Annots"]:
      obj = dict(annot.get_object())
      if obj["/Subtype"] == "/Link":
        continue
      if obj['/Subtype'] == '/Popup':  # popup object itself does not have the information
        continue
      # /Subtype == /Caret could be a duplicate but not always.
      # Use simple generic rule: Skip if we have seen it via /IRT before.
      if annot.idnum in visited_irt:
        continue
      _cleanup_obj(obj)
      if obj.get('/IRT'):
        visited_irt.add(obj['/IRT'].idnum)
        irt_obj = dict(obj['/IRT'].get_object())
        _cleanup_obj(irt_obj)
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
      # /Rect is left/bottom/right/top
      rect = obj['/Rect']
      center = float(rect[0] + rect[2]) / 2, float(rect[1] + rect[3]) / 2
      # obj["/RectCenter"] = center
      # obj["/PageNum"] = page_num
      obj["<synctex>"] = f"{page_num + 1}:{center[0]:.2f}:{center[1]:.2f}"
      # subprocess.getoutput()
      print(annot, obj)

  if args.page is not None:
    _handle_page(args.page)
  else:
    for i in range(reader.numPages):
      _handle_page(i)

  print(values_by_key)


if __name__ == "__main__":
  main()


