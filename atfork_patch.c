/*

gcc -shared -O2 -fPIC atfork_patch.c -o atfork_patch.so

*/

#include <stdio.h>

int pthread_atfork(void (*prepare)(void), void (*parent)(void), void (*child)(void)) {
    printf("Ignoring pthread_atfork call!\n");
    fflush(stdout);
    return 0;
}
