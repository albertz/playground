/*
see cpp-link-py-tf-compile.py
*/

#include <Python.h>
#include <tensorflow/core/public/session.h>

int main() {
    Py_Initialize();
    PyImport_ImportModule("hashlib");
    PyImport_ImportModule("numpy");
    PyImport_ImportModule("tensorflow");

    tensorflow::GraphDef graph;
    tensorflow::SessionOptions options;
    auto session(tensorflow::NewSession(options));
    tensorflow::Status s = session->Create(graph);

    return 0;
}
