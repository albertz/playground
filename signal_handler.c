// Code under the MIT license, copyright Albert Zeyer.
// Reuses NSIG definition from Ruby code.

// compile:
// cc signal_handler.c -fPIC -shared -o signal_handler.so

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>


// https://github.com/ruby/ruby/blob/bbfd735b887/vm_core.h#L118
#if defined(NSIG_MAX)           /* POSIX issue 8 */
# undef NSIG
# define NSIG NSIG_MAX
#elif defined(_SIG_MAXSIG)      /* FreeBSD */
# undef NSIG
# define NSIG _SIG_MAXSIG
#elif defined(_SIGMAX)          /* QNX */
# define NSIG (_SIGMAX + 1)
#elif defined(NSIG)             /* 99% of everything else */
# /* take it */
#else                           /* Last resort */
# define NSIG (sizeof(sigset_t) * CHAR_BIT + 1)
#endif


sig_t old_signal_handler[NSIG];


void signal_handler(int sig) {
  void *array[16 * 1024];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, sizeof(array)/sizeof(array[0]));

  // print out all the frames to stderr
  fprintf(stderr, "Signal handler: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);

  // call previous handler
  signal(sig, old_signal_handler[sig]);
  raise(sig);
}

void install_signal_handler() {
  old_signal_handler[SIGSEGV] = signal(SIGSEGV, signal_handler);
  old_signal_handler[SIGBUS] = signal(SIGBUS, signal_handler);
  old_signal_handler[SIGILL] = signal(SIGILL, signal_handler);
  old_signal_handler[SIGABRT] = signal(SIGABRT, signal_handler);
  old_signal_handler[SIGFPE] = signal(SIGFPE, signal_handler);
}

