/*
see cpp-link-py-tf-compile.py
*/

#include <Python.h>

int main() {
    Py_Initialize();
    PyImport_ImportModule("numpy");

    return 0;
}
