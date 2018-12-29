#!/usr/bin/env python3

# https://github.com/PerBothner/DomTerm/
# https://github.com/PerBothner/DomTerm/issues/54
# https://domterm.org/Wire-byte-protocol.html

import sys
import contextlib
import os


_is_domterm = None

def is_domterm():
    global _is_domterm
    if _is_domterm is not None:
        return _is_domterm
    if not os.environ.get("DOMTERM"):
        _is_domterm = False
        return False
    _is_domterm = True
    return True


@contextlib.contextmanager
def logical_block(file=sys.stdout):
    file.write("\033]110\007")
    yield
    file.write("\033]111\007")

@contextlib.contextmanager
def hide_button_span(mode, file=sys.stdout):
    """
    :param int mode: 1 or 2
    """
    file.write("\033[83;%iu" % mode)
    yield
    file.write("\033[83;0u")

def indentation(file=sys.stdout):
    file.write("\033]114;\"│\"\007")

def hide_button(file=sys.stdout):
    file.write("\033[16u▶▼\033[17u")

@contextlib.contextmanager
def temp_replace_attrib(obj, attr, new_value):
    old_value = getattr(obj, attr)
    setattr(obj, attr, new_value)
    yield old_value
    setattr(obj, attr, old_value)


@contextlib.contextmanager
def fold_text_stream(prefix, postfix="", hidden_stream=None, **kwargs):
    """
    :param str prefix: always visible
    :param str postfix: always visible, right after.
    :param io.Base hidden_stream: sys.stdout by default.
        If this is sys.stdout, it will replace that stream,
        and collect the data during the context (in the `with` block).
    :param io.IOBase file: sys.stdout by default.
    """
    import io
    if hidden_stream is None:
        hidden_stream = sys.stdout
    assert isinstance(hidden_stream, io.IOBase)
    assert hidden_stream is sys.stdout, "currently not supported otherwise"
    hidden_buf = io.StringIO()
    with temp_replace_attrib(sys, "stdout", hidden_buf):
        yield
    fold_text(prefix=prefix, postfix=postfix, hidden=hidden_buf.getvalue(), **kwargs)


def fold_text(prefix, hidden, postfix="", file=None):
    """
    :param str prefix: always visible
    :param str hidden: hidden
        If this is sys.stdout, it will replace that stream,
        and collect the data during the context (in the `with` block).
    :param str postfix: always visible, right after. "" by default.
    :param io.IOBase file: sys.stdout by default.
    """
    if file is None:
        file = sys.stdout
    # Extra logic: Multi-line hidden. Add initial "\n" if not there.
    if "\n" in hidden:
        if hidden[:1] != "\n":
            hidden = "\n" + hidden
    # Extra logic: A final "\n" of hidden, make it always visible such that it looks nicer.
    if hidden[-1:] == "\n":
        hidden = hidden[:-1]
        postfix += "\n"
    if is_domterm():
        with logical_block(file=file):
            indentation(file=file)
            hide_button(file=file)
            file.write(prefix)
            with hide_button_span(2, file=file):
                file.write(hidden.replace("\n", "\033]118\007"))
    else:
        file.write(prefix)
        file.write(hidden.replace("\n", "\n "))
    file.write(postfix)
    file.flush()


def main():
    if is_domterm():
        print("In DomTerm.")
        print("text folding:")
        with logical_block():
            indentation()
            hide_button()
            print("<always visible>", end="")
            with hide_button_span(2):
                print("<hidden", end="\033]118\007")
                print(" hid-next-line", end="\033]118\007")
                print(" hid-final-line>")
    else:
        print("Not in DomTerm.")

    print("Something unrelated, in a new line.")
    print("Now with nice Python construct:")

    with fold_text_stream("Further information:"):
        print("Foo")
        print("Bar")
        print("Blubber")

    print("Out of fold.")
    print("Try single-line:")
    with fold_text_stream("Extra here:", "\n"):
        print("This is the extra.", end="")

    print("Now try some nested fold:")

    with fold_text_stream("And more:"):
        print("Hello there.")
        print("Now we try nesting.")
        with fold_text_stream("Nested:"):
            print("Yes, that's nice!")
            print("Even more?")
            with fold_text_stream("Extra:"):
                print("Very cool.")
                print("Extra line.")
            print("After the deepest fold.")
        print("After second fold.")
        print("Some extra.")

    print("Enough.")
    print("Bye.")


if __name__ == "__main__":
    try:
        import better_exchook
        better_exchook.install()
    except ImportError:
        pass
    main()

