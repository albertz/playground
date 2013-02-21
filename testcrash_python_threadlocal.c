// compile:
// gcc -c testcrash_python_threadlocal.c -I /System/Library/Frameworks/Python.framework/Headers/
// libtool -dynamic -o testcrash_python_threadlocal.so testcrash_python_threadlocal.o -framework Python -lc

#include <Python.h>
#include <unistd.h>

static
void test_dealloc(PyObject* obj) {
	printf("test_dealloc\n");
	
	Py_BEGIN_ALLOW_THREADS
	usleep(1000 * 1000 * 3);
	Py_END_ALLOW_THREADS

	Py_TYPE(obj)->tp_free(obj);
}

struct TestObj {
	PyObject_HEAD
};

PyTypeObject Test_Type = {
	PyVarObject_HEAD_INIT(&PyType_Type, 0)
	"TestType",
	sizeof(struct TestObj),	// basicsize
	0,	// itemsize
	test_dealloc,		/*tp_dealloc*/
	0,                  /*tp_print*/
	0,		/*tp_getattr*/
	0,		/*tp_setattr*/
	0,                  /*tp_compare*/
	0,					/*tp_repr*/
	0,                  /*tp_as_number*/
	0,                  /*tp_as_sequence*/
	0,                  /*tp_as_mapping*/
	0,					/*tp_hash */
	0, // tp_call
	0, // tp_str
	0, // tp_getattro
	0, // tp_setattro
	0, // tp_as_buffer
	Py_TPFLAGS_DEFAULT, // flags
	"Test type", // doc
	0, // tp_traverse
	0, // tp_clear
	0, // tp_richcompare
	0, // weaklistoffset
	0, // iter
	0, // iternext
	0, // methods
	0, //PlayerMembers, // members
	0, // getset
	0, // base
	0, // dict
	0, // descr_get
	0, // descr_set
	0, // dictoffset
	0, // tp_init
	0, // alloc
	PyType_GenericNew, // new
};


PyMODINIT_FUNC
inittestcrash_python_threadlocal(void)
{
	if (PyType_Ready(&Test_Type) < 0)
		Py_FatalError("Can't initialize test type");
	PyObject* m = Py_InitModule3("testcrash_python_threadlocal", NULL, NULL);
	if(!m) {
		Py_FatalError("Can't initialize module");
		return;
	}
	
	Py_INCREF(&Test_Type);
	PyModule_AddObject(m, "Test", (PyObject*) &Test_Type);
}
