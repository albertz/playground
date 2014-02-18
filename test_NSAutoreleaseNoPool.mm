// compile:
// c++ -framework CoreFoundation -framework Foundation test_NSAutoreleaseNoPool.mm

#include <dlfcn.h>
#include <stdio.h>
#import <Foundation/Foundation.h>
#include <mach-o/dyld.h>

typedef void (*NSAutoreleaseNoPoolFunc) (void* obj);

void foo() {
	NSAutoreleaseNoPoolFunc __NSAutoreleaseNoPool = (NSAutoreleaseNoPoolFunc) dlsym(RTLD_DEFAULT, "__NSAutoreleaseNoPool");
	//void* p2 = NSAddressOfSymbol("__NSAutoreleaseNoPool");
	const struct mach_header* img = NSAddImage("/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation", NSADDIMAGE_OPTION_NONE);
	NSSymbol sym = NSLookupSymbolInImage(img, "___NSAutoreleaseNoPool", NSLOOKUPSYMBOLINIMAGE_OPTION_BIND_NOW);
	NSSymbol sym2 = NSLookupAndBindSymbol("___NSAutoreleaseNoPool");
	void* p3 = 0;
	_dyld_lookup_and_bind("___NSAutoreleaseNoPool", &p3, NULL);
	printf("func: %p (%p) %p %p %p\n", __NSAutoreleaseNoPool, img, sym, sym2, p3);
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

