/*

See atfork_demo_main.c.
And:
https://stackoverflow.com/questions/46845496/ld-preload-and-linkage/

Compile this:
gcc -shared -fPIC atfork_patch_extended.c -o atfork_patch_extended.so

*/

#define _GNU_SOURCE
#include <sched.h>
#include <signal.h>
#include <sys/syscall.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>



int pthread_atfork(void (*prepare)(void), void (*parent)(void), void (*child)(void)) {
    printf("Ignoring pthread_atfork call extended!\n");
    fflush(stdout);
    return 0;
}

int __register_atfork(void (*prepare)(void), void (*parent)(void), void (*child)(void)) {
    printf("Ignoring __register_atfork call extended!\n");
    fflush(stdout);
    return 0;
}


// Another way to ignore atfork handlers: Override fork.
pid_t fork(void) {
    printf("Patched fork().\n");
    fflush(stdout);
    return syscall(SYS_clone, SIGCHLD, 0);
}


__attribute__((constructor))
static void init(void) {
    printf("atfork_patch_extended init\n");
    fflush(stdout);
}
