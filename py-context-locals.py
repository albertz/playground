

# https://github.com/rwth-i6/returnn_common/issues/16#issuecomment-929030603

from contextlib import contextmanager


@contextmanager
def loop1(vs):
  print("enter", vs)
  yield
  print("exit", vs)


def test1():
  a, b = 1, 2
  with loop1(locals()):
    b = b + 1


@contextmanager
def loop2(vs):
  print("enter", vs())
  yield
  print("exit", vs())


def test2():
  import better_exchook
  frame = better_exchook.get_current_frame()
  a, b = 1, 2
  with loop2(lambda: frame.f_locals):
    b = b + 1
  locals()  # mark b as used


class _Loop3:
  def __init__(self):
    self.new_vs = None

  # noinspection PyMethodMayBeStatic
  def cond(self, test):
    print("cond", test)

  def exit(self, vs_):
    assert self.new_vs is None and vs_ is not None
    self.new_vs = vs_


@contextmanager
def loop3(vs):
  print("enter", vs)

  _loop = _Loop3()
  yield _loop
  assert _loop.new_vs is not None
  print("exit", _loop.new_vs)


def test3():
  a, b = 1, 2
  with loop3(locals()) as loop:
    loop.cond(b < 10)
    b = b + 1
    loop.exit(locals())


def _test4_demo_func():
  a, b = 1, 2
  while b < 10:
    b = b + 1


def test4():
  import better_exchook
  better_exchook.install()

  func = _test4_demo_func
  from tensorflow.python.autograph.pyct import transpiler
  from tensorflow.python.autograph.pyct.transformer import Base as PythonCodeTransformer
  from tensorflow.python.autograph.pyct import templates
  import inspect

  class _MyTransformer(PythonCodeTransformer):
    def visit_While(self, node):
      # see TF ControlFlowTransformer.visit_While
      print("visit_While", node)
      template = """
        with loop3(locals()) as loop_name:
          loop_name.cond(cond_node)
          body_node
          loop_name.exit(locals())
      """
      return templates.replace(
        template,
        body_node=node.body,
        cond_node=node.test,
        loop_name=self.ctx.namer.new_symbol('loop', reserved_locals=()))

  class _MyFuncTranspiler(transpiler.FunctionTranspiler):
    def transform_ast(self, node, user_context):
      _transformer = _MyTransformer(user_context)
      return _transformer.visit(node)

  _transpiler = _MyFuncTranspiler()
  new_func, _, _ = _transpiler.transform_function(
    func, caching_subkey=None, user_context=None, extra_locals={})
  print(inspect.getsource(new_func))
  new_func()


def main():
  test4()


if __name__ == '__main__':
  main()
