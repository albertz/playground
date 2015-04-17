
# Copyright (c) 2012, Albert Zeyer, www.az2000.de
# All rights reserved.
# file created 2011-04-15


# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution. 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# This is a simple replacement for the standard Python exception handler (sys.excepthook).
# In addition to what the standard handler does, it also prints all referenced variables
# (no matter if local, global or builtin) of the code line of each stack frame.
# See below for some examples and some example output.

# https://github.com/albertz/py_better_exchook

import sys, os, os.path

try:
	unicode
except NameError: # Python3
	unicode = str   # Python 3 compatibility

try:
	raw_input
except NameError: # Python3
	raw_input = input

def parse_py_statement(line):
	state = 0
	curtoken = ""
	spaces = " \t\n"
	ops = ".,;:+-*/%&!=|(){}[]^<>"
	i = 0
	def _escape_char(c):
		if c == "n": return "\n"
		elif c == "t": return "\t"
		else: return c
	while i < len(line):
		c = line[i]
		i += 1
		if state == 0:
			if c in spaces: pass
			elif c in ops: yield ("op", c)
			elif c == "#": state = 6
			elif c == "\"": state = 1
			elif c == "'": state = 2
			else:
				curtoken = c
				state = 3
		elif state == 1: # string via "
			if c == "\\": state = 4
			elif c == "\"":
				yield ("str", curtoken)
				curtoken = ""
				state = 0
			else: curtoken += c
		elif state == 2: # string via '
			if c == "\\": state = 5
			elif c == "'":
				yield ("str", curtoken)
				curtoken = ""
				state = 0
			else: curtoken += c
		elif state == 3: # identifier
			if c in spaces + ops + "#\"'":
				yield ("id", curtoken)
				curtoken = ""
				state = 0
				i -= 1
			else: curtoken += c
		elif state == 4: # escape in "
			curtoken += _escape_char(c)
			state = 1
		elif state == 5: # escape in '
			curtoken += _escape_char(c)
			state = 2
		elif state == 6: # comment
			curtoken += c
	if state == 3: yield ("id", curtoken)
	elif state == 6: yield ("comment", curtoken)


import keyword
pykeywords = set(keyword.kwlist)

def grep_full_py_identifiers(tokens):
	global pykeywords
	tokens = list(tokens)
	i = 0
	while i < len(tokens):
		tokentype, token = tokens[i]
		i += 1
		if tokentype != "id": continue
		while i+1 < len(tokens) and tokens[i] == ("op", ".") and tokens[i+1][0] == "id":
			token += "." + tokens[i+1][1]
			i += 2
		if token == "": continue
		if token in pykeywords: continue
		if token[0] in ".0123456789": continue
		yield token

def set_linecache(filename, source):
	import linecache
	linecache.cache[filename] = None, None, [line+'\n' for line in source.splitlines()], filename

def simple_debug_shell(globals, locals):
	try: import readline
	except ImportError: pass # ignore
	COMPILE_STRING_FN = "<simple_debug_shell input>"
	while True:
		try:
			s = raw_input("> ")
		except (KeyboardInterrupt, EOFError):
			print("breaked debug shell: " + sys.exc_info()[0].__name__)
			break
		if s.strip() == "": continue
		try:
			c = compile(s, COMPILE_STRING_FN, "single")
		except Exception as e:
			print("%s : %s in %r" % (e.__class__.__name__, str(e), s))
		else:
			set_linecache(COMPILE_STRING_FN, s)
			try:
				ret = eval(c, globals, locals)
			except (KeyboardInterrupt, SystemExit):
				print("debug shell exit: " + sys.exc_info()[0].__name__)
				break
			except Exception:
				print("Error executing %r" % s)
				better_exchook(*sys.exc_info(), autodebugshell=False)
			else:
				try:
					if ret is not None: print(ret)
				except Exception:
					print("Error printing return value of %r" % s)
					better_exchook(*sys.exc_info(), autodebugshell=False)
		
def debug_shell(user_ns, user_global_ns, execWrapper=None):
	ipshell = None
	try:
		import IPython
		import IPython.terminal.embed
		class DummyMod(object): pass
		module = DummyMod()
		module.__dict__ = user_global_ns
		module.__name__ = "DummyMod"
		ipshell = IPython.terminal.embed.InteractiveShellEmbed(
			user_ns=user_ns, user_module=module)
	except Exception: pass
	else:
		if execWrapper:
			old = ipshell.run_code
			ipshell.run_code = lambda code: execWrapper(lambda: old(code))
	if ipshell:
		ipshell()
	else:
		simple_debug_shell(user_global_ns, user_ns)						

def output(s): print(s)

def output_limit():
	return 300

def pp_extra_info(obj, depthlimit = 3):
	s = []
	if hasattr(obj, "__len__"):
		try:
			if type(obj) in (str,unicode,list,tuple,dict) and len(obj) <= 5:
				pass # don't print len in this case
			else:
				s += ["len = " + str(obj.__len__())]
		except Exception: pass
	if depthlimit > 0 and hasattr(obj, "__getitem__"):
		try:
			if type(obj) in (str,unicode):
				pass # doesn't make sense to get subitems here
			else:
				subobj = obj.__getitem__(0)
				extra_info = pp_extra_info(subobj, depthlimit - 1)
				if extra_info != "":
					s += ["_[0]: {" + extra_info + "}"]
		except Exception: pass
	return ", ".join(s)
	
def pretty_print(obj):
	s = repr(obj)
	limit = output_limit()
	if len(s) > limit:
		s = s[:limit - 3] + "..."
	extra_info = pp_extra_info(obj)
	if extra_info != "": s += ", " + extra_info
	return s

def fallback_findfile(filename):
	mods = [ m for m in sys.modules.values() if m and hasattr(m, "__file__") and filename in m.__file__ ]
	if len(mods) == 0: return None
	altfn = mods[0].__file__
	if altfn[-4:-1] == ".py": altfn = altfn[:-1] # *.pyc or whatever
	return altfn

def print_traceback(tb=None, allLocals=None, allGlobals=None):
	if tb is None:
		try:
			tb = sys._getframe()
			assert tb
		except Exception:
			output("print_traceback: tb is None and sys._getframe() failed")
			return
	import inspect
	isframe = inspect.isframe
	if isframe(tb): output('Traceback (most recent call first)')
	else: output('Traceback (most recent call last):') # expect traceback-object (or compatible)
	try:
		import linecache
		limit = None
		if hasattr(sys, 'tracebacklimit'):
			limit = sys.tracebacklimit
		n = 0
		_tb = tb
		def _resolveIdentifier(namespace, id):
			obj = namespace[id[0]]
			for part in id[1:]:
				obj = getattr(obj, part)
			return obj
		def _trySet(old, prefix, func):
			if old is not None: return old
			try: return prefix + func()
			except KeyError: return old
			except Exception as e:
				return prefix + "!" + e.__class__.__name__ + ": " + str(e)
		while _tb is not None and (limit is None or n < limit):
			if isframe(_tb): f = _tb
			else: f = _tb.tb_frame
			if allLocals is not None: allLocals.update(f.f_locals)
			if allGlobals is not None: allGlobals.update(f.f_globals)
			if hasattr(_tb, "tb_lineno"): lineno = _tb.tb_lineno
			else: lineno = f.f_lineno
			co = f.f_code
			filename = co.co_filename
			name = co.co_name
			output('  File "%s", line %d, in %s' % (filename,lineno,name))
			if not os.path.isfile(filename):
				altfn = fallback_findfile(filename)
				if altfn:
					output("    -- couldn't find file, trying this instead: " + altfn)
					filename = altfn
			linecache.checkcache(filename)
			line = linecache.getline(filename, lineno, f.f_globals)
			if line:
				line = line.strip()
				output('    line: ' + line)
				output('    locals:')
				alreadyPrintedLocals = set()
				for tokenstr in grep_full_py_identifiers(parse_py_statement(line)):
					splittedtoken = tuple(tokenstr.split("."))
					for token in [splittedtoken[0:i] for i in range(1, len(splittedtoken) + 1)]:
						if token in alreadyPrintedLocals: continue
						tokenvalue = None
						tokenvalue = _trySet(tokenvalue, "<local> ", lambda: pretty_print(_resolveIdentifier(f.f_locals, token)))
						tokenvalue = _trySet(tokenvalue, "<global> ", lambda: pretty_print(_resolveIdentifier(f.f_globals, token)))
						tokenvalue = _trySet(tokenvalue, "<builtin> ", lambda: pretty_print(_resolveIdentifier(f.f_builtins, token)))
						tokenvalue = tokenvalue or "<not found>"
						output('      ' + ".".join(token) + " = " + tokenvalue)
						alreadyPrintedLocals.add(token)
				if len(alreadyPrintedLocals) == 0: output("       no locals")
			else:
				output('    -- code not available --')
			if isframe(_tb): _tb = _tb.f_back
			else: _tb = _tb.tb_next
			n += 1

	except Exception as e:
		output("ERROR: cannot get more detailed exception info because:")
		import traceback
		for l in traceback.format_exc().split("\n"): output("   " + l)
		output("simple traceback:")
		if isframe(tb): traceback.print_stack(tb)
		else: traceback.print_tb(tb)
	

def better_exchook(etype, value, tb, debugshell=False, autodebugshell=True):
	output("EXCEPTION")
	allLocals,allGlobals = {},{}
	if tb is not None:
		print_traceback(tb, allLocals=allLocals, allGlobals=allGlobals)
	else:
		output("better_exchook: traceback unknown")
	
	import types
	def _some_str(value):
		try: return str(value)
		except Exception: return '<unprintable %s object>' % type(value).__name__
	def _format_final_exc_line(etype, value):
		valuestr = _some_str(value)
		if value is None or not valuestr:
			line = "%s" % etype
		else:
			line = "%s: %s" % (etype, valuestr)
		return line
	if (isinstance(etype, BaseException) or
		(hasattr(types, "InstanceType") and isinstance(etype, types.InstanceType)) or
		etype is None or type(etype) is str):
		output(_format_final_exc_line(etype, value))
	else:
		output(_format_final_exc_line(etype.__name__, value))

	if autodebugshell:
		try: debugshell = int(os.environ["DEBUG"]) != 0
		except Exception: pass
	if debugshell:
		output("---------- DEBUG SHELL -----------")
		debug_shell(user_ns=allLocals, user_global_ns=allGlobals)
		
def install():
	sys.excepthook = better_exchook
	
if __name__ == "__main__":
	# some examples
	# this code produces this output: https://gist.github.com/922622
	
	try:
		x = {1:2, "a":"b"}
		def f():
			y = "foo"
			x, 42, sys.stdin.__class__, sys.exc_info, y, z
		f()
	except Exception:
		better_exchook(*sys.exc_info())

	try:
		f = lambda x: None
		f(x, y)
	except Exception:
		better_exchook(*sys.exc_info())

	# use this to overwrite the global exception handler
	sys.excepthook = better_exchook
	# and fail
	finalfail(sys)
