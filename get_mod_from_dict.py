#!/usr/bin/env python3

"""
https://stackoverflow.com/questions/56171796/get-module-instance-given-its-vars-dict
"""

import sys
import types
import gc
import pprint
import better_exchook


def get_dict_from_mod(mod):
  """
  :param types.ModuleType mod:
  :rtype: dict[str]
  """
  assert isinstance(mod, types.ModuleType)
  return vars(mod)


_cached_mod_dict_map = {}


def get_mod_from_dict_1(d):
  """
  :param dict[str] d:
  :rtype: types.ModuleType|None
  """
  # Note: `list(sys.modules.items())` because the lazy module loaders
  # (e.g. from RETURNN) could change `sys.modules` while iterating through it.
  # Note: The cache is purely for nicer output, to avoid many import warnings
  # e.g. by the RETURNN lazy module loader.
  if not _cached_mod_dict_map:
    _cached_mod_dict_map.update(
      {id(mod.__dict__): mod
       for (modname, mod) in list(sys.modules.items())
       if mod and modname != "__main__"})
  return _cached_mod_dict_map.get(id(d), None)


def get_mod_from_dict_2(d):
  """
  :param dict[str] d:
  :rtype: types.ModuleType|None
  """
  if '__name__' not in d:
    return None
  module_name = d['__name__']
  if module_name not in sys.modules:
    return None
  mod = sys.modules[module_name]
  assert vars(mod) is d
  return mod


def get_mod_from_dict_3(d):
  """
  :param dict[str] d:
  :rtype: types.ModuleType|None
  """
  objects = gc.get_referrers(d)
  for obj in objects:
    if isinstance(obj, types.ModuleType) and vars(obj) is d:
      return obj
  return None


def test():
  import os
  import numpy
  import tensorflow.contrib
  import returnn.TFUtil
  funcs = [f for name, f in globals().items() if name.startswith("get_mod_from_dict_")]
  mods = [sys, os, gc, pprint, better_exchook, numpy, tensorflow, tensorflow.contrib, returnn, returnn.TFUtil]
  for f in funcs:
    print("Testing func:", f)
    for mod in mods:
      print("Testing mod:", mod)
      d = get_dict_from_mod(mod)
      mod2 = f(d)
      assert mod2 and vars(mod2) is d
      # Note: For lazy module loaders, such as in RETURNN,
      # we do not necessarily have `mod is mod2`.
      if mod is not mod2:
        print("Note: Mod is different:", mod2)


if __name__ == '__main__':
  better_exchook.install()
  test()
