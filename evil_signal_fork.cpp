#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
#include <iostream>

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <ctype.h>
#include <unistd.h>
#include <setjmp.h>
#include <execinfo.h>

void DumpCallstackPrintf() {
	void *callstack[128];
	int framesC = backtrace(callstack, sizeof(callstack));
	printf("backtrace() returned %d addresses\n", framesC);
	char** strs = backtrace_symbols(callstack, framesC);
	for(int i = 0; i < framesC; ++i) {
		if(strs[i])
			printf("%s\n", strs[i]);
		else
			break;
	}
	free(strs);
}

sigjmp_buf longJumpBuffer;

void SimpleSignalHandler(int Sig) {
	printf("SignalHandler: %i\n", Sig);
	signal(Sig, SIG_IGN); // discard all remaining signals
		
	DumpCallstackPrintf();
		
	if(!fork()) {
		abort();
	}

	signal(Sig, SimpleSignalHandler); // reset handler
	printf("resuming ...\n");
	fflush(stdout);
	siglongjmp(longJumpBuffer, 1); // jump back to main loop, maybe we'll be able to continue somehow
}

int main() {
	printf("installing signal handler ...\n");
	signal(SIGSEGV, &SimpleSignalHandler);
	signal(SIGTRAP, &SimpleSignalHandler);
	signal(SIGABRT, &SimpleSignalHandler);
	signal(SIGHUP, &SimpleSignalHandler);
	signal(SIGBUS, &SimpleSignalHandler);
	signal(SIGILL, &SimpleSignalHandler);
	signal(SIGFPE, &SimpleSignalHandler);		
	signal(SIGSYS, &SimpleSignalHandler);
	
	printf("callstack:\n");
	DumpCallstackPrintf(); // dummy call to force loading dynamic lib at this point (with sane heap) for backtrace and friends

	printf("entering main loop ...\n");

	bool quit = false;
	while(!quit) {
		sigsetjmp(longJumpBuffer, 1);

		using namespace std;
		cout << "command: " << flush;
		std::string cmd;
		cin >> cmd;
		if(cmd == "quit") quit = true;
		else if(cmd == "crash") {
			cout << "crashing ..." << endl;
			(*(int*)0x13) = 42;
			assert(false);
		} else if(cmd == "evil") {
			cout << "CRASH" << endl;
			sigsetjmp(longJumpBuffer, 1);
			assert(false);			
		} else
			cout << "unknown cmd" << endl;
	}

	return 0;
}
