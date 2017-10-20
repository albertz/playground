/*

https://stackoverflow.com/questions/46845496/ld-preload-and-linkage

compile this via:
gcc atfork_demo_main.c -o atfork_demo_main.exec -ldl

See also atfork_demo.c and atfork_patch.c.

gcc -shared -fPIC atfork_demo.c -o atfork_demo1.so
gcc -shared -fPIC atfork_demo.c -o atfork_demo2.so -lpthread
gcc atfork_demo_main.c -o atfork_demo_main.exec -ldl

Then set LD_PRELOAD=./atfork_patch.so, and run:

./atfork_demo_main.exec ./atfork_demo1.so
./atfork_demo_main.exec ./atfork_demo2.so

I get:

Ignoring pthread_atfork call!
Hello from atfork prepare.

*/

#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, const char** argv) {
    if(argc <= 1) {
        printf("usage: ... lib.so\n");
        return 1;
    }
    void* plib = dlopen("libpthread.so.0", RTLD_NOW|RTLD_GLOBAL);
    if(!plib) {
        printf("cannot load pthread, error %s\n", dlerror());
        return 1;
    }
    void* lib = dlopen(argv[1], RTLD_LAZY);
    if(!lib) {
        printf("cannot load %s, error %s\n", argv[1], dlerror());
        return 1;
    }
    void (*reg)();
    reg = dlsym(lib, "register_hello_from_fork_prepare");
    if(!reg) {
        printf("did not found func, error %s\n", dlerror());
        return 1;
    }
    reg();
    fork();
}

