/*

See atfork_demo_main.c.
Compile this:
gcc -shared -fPIC atfork_patch.c -o atfork_patch.so

*/

#include <stdio.h>

int pthread_atfork(void (*prepare)(void), void (*parent)(void), void (*child)(void)) {
    printf("Ignoring pthread_atfork call!\n");
    fflush(stdout);
    return 0;
}
