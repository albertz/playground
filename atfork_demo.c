/*

See atfork_demo_main.c.
Compile this:
gcc -shared -fPIC atfork_demo.c -o atfork_demo1.so
gcc -shared -fPIC atfork_demo.c -o atfork_demo2.so -lpthread

*/

#include <stdio.h>
#include <pthread.h>

void hello_from_fork_prepare() {
    printf("Hello from atfork prepare.\n");
    fflush(stdout);
}

void register_hello_from_fork_prepare() {
    printf("Register.\n");
    fflush(stdout);
    pthread_atfork(&hello_from_fork_prepare, 0, 0);
}
