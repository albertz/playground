/*
see cpp-link-py-tf-compile.py
*/

#include <sys/resource.h>
#include <Python.h>
#include <tensorflow/core/public/session.h>

int main() {
    struct rlimit rl;
    getrlimit(RLIMIT_STACK, &rl);
    rl.rlim_cur = 64 * 1024 * 1024;   // 64MiB stack
    setrlimit(RLIMIT_STACK, &rl);

    Py_Initialize();
    PyImport_ImportModule("numpy");
    PyImport_ImportModule("tensorflow");

    tensorflow::GraphDef graph;
    tensorflow::SessionOptions options;
    auto session(tensorflow::NewSession(options));
    tensorflow::Status s = session->Create(graph);

    return 0;
}
