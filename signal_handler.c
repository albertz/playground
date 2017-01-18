// compile:
// cc signal_handler.c -shared -o signal_handler.so

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>



sig_t old_signal_handler = 0;


void signal_handler(int sig) {
  void *array[100];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, sizeof(array)/sizeof(array[0]));

  // print out all the frames to stderr
  fprintf(stderr, "Signal handler: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);

  // call previous handler
  signal(sig, old_signal_handler);
  raise(sig);
}

void install_signal_handler() {
  old_signal_handler = signal(SIGSEGV, signal_handler);
}

