
"""
https://youtrack.jetbrains.com/issue/IDEA-275494
"""


from __future__ import annotations


class A:

  def __init__(self, parent: A):
    self.attrib = False
    assert parent.attrib  # note, no warning here
    self.parent = parent
    assert self.parent.attrib  # Warning: Unresolved attribute reference 'attrib' for class 'A'
