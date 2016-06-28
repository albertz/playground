#!/usr/bin/env python

# Confusing when you don't know about name mangling in Python classes.
# https://docs.python.org/2/tutorial/classes.html#private-variables-and-class-local-references

class A:
    __X = 42  # actually will be the attrib "_A__X"

class B:
    Y = A.__X  # "__X" lookup will be translated to "_B__X" here
