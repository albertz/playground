

_have_inplace_increment = None
_native_inplace_increment = None


def inplace_increment(x, idx, y):
  global _have_inplace_increment, inplace_increment, _native_inplace_increment
  if _have_inplace_increment is None:
    native_inpl_incr = None
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    import theano
    if theano.config.cxx:
      # noinspection PyPackageRequirements,PyUnresolvedReferences
      import theano.gof.cutils  # needed to import cutils_ext
      try:
        from cutils_ext.cutils_ext import inplace_increment as native_inpl_incr
      except ImportError:
        pass
    if native_inpl_incr:
      _have_inplace_increment = True
      _native_inplace_increment = native_inpl_incr
      inplace_increment = native_inpl_incr  # replace myself
      return inplace_increment(x, idx, y)  # warning: 'inplace_increment' is not callable
    _have_inplace_increment = False
  if _have_inplace_increment is True:
    return _native_inplace_increment(x, idx, y)  # warning: '_native_inplace_increment' is not callable
  raise NotImplementedError("need Numpy 1.8 or later")


def global_func():
  # https://youtrack.jetbrains.com/issue/PY-28498
  global global_func  # Global variable 'global_func' is undefined at the module level

  def new_func():
    print("Hello")

  global_func = new_func
  global_func()


def optional_import_func():
  """
  https://youtrack.jetbrains.com/issue/PY-43917
  """
  func = None
  try:
    # warning: 'func' in try block with 'except ImportError' should also be defined in except block
    from some_random_mod import func
  except ImportError:
    pass
  if func:
    func()  # warning: 'func' is not callable
