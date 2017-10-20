/*

See atfork_demo_main.c.
Compile this:
gcc -shared -fPIC atfork_patch_extended.c -o atfork_patch_extended.so

*/

#define _GNU_SOURCE
#include <sched.h>
#include <signal.h>
#include <sys/syscall.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <unistd.h>
#include <string.h>


void* (*real_dlsym)(void *handle, const char *symbol);
void* (*real_dlopen)(const char *, int flag);
extern void *_dl_sym (void *handle, const char *name, void *who);


int pthread_atfork(void (*prepare)(void), void (*parent)(void), void (*child)(void)) {
    printf("Ignoring pthread_atfork call extended!\n");
    fflush(stdout);
    return 0;
}


int _pthread_atfork(void (*prepare)(void), void (*parent)(void), void (*child)(void)) {
    printf("Ignoring _pthread_atfork call extended!\n");
    fflush(stdout);
    return 0;
}

int __pthread_atfork(void (*prepare)(void), void (*parent)(void), void (*child)(void)) {
    printf("Ignoring __pthread_atfork call extended!\n");
    fflush(stdout);
    return 0;
}


void *dlsym(void *handle, const char *symbol) { 
    printf("dlsym: %s\n", symbol);
    void* result = real_dlsym(handle, symbol);
    //if ((handle != RTLD_NEXT) || (handle != RTLD_DEFAULT))
    if(strcmp(symbol, "pthread_atfork") == 0) { 
        printf("Return our pthread_atfork\n");
        return pthread_atfork;
    }
    return result; 
}

void *dlopen(const char *filename, int flag) {
    printf("dlopen: %s\n", filename ? filename : "(NULL)");
    int new_flag = flag & (~RTLD_DEEPBIND);
    return (*real_dlopen)(filename, new_flag);
}


// Another way to ignore atfork handlers: Override fork.
pid_t fork(void) {
  syscall(SYS_clone, SIGCHLD, 0);
}


__attribute__((constructor))
static void init(void) {
    printf("atfork_patch_extended init\n");
    fflush(stdout);
    real_dlsym = _dl_sym(RTLD_NEXT, "dlsym", dlsym);
    real_dlopen = dlsym(RTLD_NEXT, "dlopen");
}
