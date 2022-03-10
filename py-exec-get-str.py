#!/usr/bin/env python3

"""

https://stackoverflow.com/questions/34126292/getting-the-c-python-exec-argument-string-or-accessing-the-evaluation-stack
https://stackoverflow.com/questions/44346433/in-c-python-accessing-the-bytecode-evaluation-stack/44443331
https://github.com/a-rahimi/python-checkpointing2


https://github.com/python/cpython/blob/main/Python/bltinmodule.c
  Defs: builtin_exec_impl (exec), builtin_eval_impl (eval).
  This calls PyEval_EvalCode or PyRun_StringFlags.

https://github.com/python/cpython/blob/main/Python/ceval.c
  Def: PyEval_EvalCode

https://github.com/python/cpython/blob/main/Python/pythonrun.c
  Def: PyRun_StringFlags.
  Calls _PyParser_ASTFromString and then run_mod -> run_eval_code_obj -> PyEval_EvalCode.
  _PyParser_ASTFromString gets the string but does not store it anywhere (I think).

"""

import inspect


def inside():
  print("inside")
  stack = inspect.stack()
  print(stack)


code_str = """
def hello_world():
  print("Hello World!")
  inside()
  
hello_world()
"""


def main():
  exec(code_str, globals())


if __name__ == '__main__':
  main()
