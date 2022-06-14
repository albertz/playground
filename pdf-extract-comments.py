#!/usr/bin/env python3

"""
Extract comments from PDF files.


https://stackoverflow.com/questions/68266356/extracting-comments-annotations-from-pdf-sequentially-python

"""

import argparse
import PyPDF2


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
  args = arg_parser.parse_args()

  reader = PyPDF2.PdfReader(args.pdf_file)

  values_by_key = {}

  for page_num, page in enumerate(reader.pages):
    assert isinstance(page, PyPDF2.PageObject)
    print("*** page:", page_num)
    if "/Annots" in page:
      for annot in page["/Annots"]:
        obj = dict(annot.get_object())
        if obj["/Subtype"] == "/Link":
          continue
        if obj['/Subtype'] == '/Popup':  # popup object itself does not have the information
          continue
        if obj["/Subtype"] == '/Caret':  # already via IRT
          continue
        _cleanup_obj(obj)
        if obj.get('/IRT'):
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
        # print(annot, obj)

  print(values_by_key)


if __name__ == "__main__":
  main()


