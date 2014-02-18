// compile:
// c++ -framework CoreFoundation -framework Foundation test_NSAutoreleaseNoPool.mm

#include <dlfcn.h>
#include <stdio.h>
#import <Foundation/Foundation.h>

typedef void (*NSAutoreleaseNoPoolFunc) (void* obj);

void foo() {
	NSAutoreleaseNoPoolFunc __NSAutoreleaseNoPool = (NSAutoreleaseNoPoolFunc) dlsym(RTLD_DEFAULT, "__NSAutoreleaseNoPool");
	printf("func: %p\n", __NSAutoreleaseNoPool);	
}

id raiseNoPool() {
	NSString* foo = [[NSString alloc] init];
	return [foo autorelease];
}

void bar() {
	id x = raiseNoPool();
}

int main() {
	foo();
	bar();
	return 0;
}

