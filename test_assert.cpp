#include <iostream>
#include <cassert>

using namespace std;

void do_the_warning (void) __attribute__((warning("assertion always false")));

#define ASSERT(e)	{ \
if(__builtin_constant_p(e)) { \
if(!(e)) { \
do_the_warning(); \
cerr << __LINE__ << ": compiletime check for " << #e << " failed" << endl; } } \
else { if(!(e)) { cerr << __LINE__ << ": runtime check for " << #e << " failed" << endl; } }; }


struct Vector {
	/* ... */
	inline int get(int i) { ASSERT(i >= 0); return 0; }
};


int main(int argc, char** argv) {
	// some very basic samples
	ASSERT(true);
	ASSERT(false); // that gives a compiler warning
	ASSERT(sizeof(int) == 4);
	ASSERT(sizeof(char) == 2); // that gives also a compiler warning
	ASSERT(argc == 1); // that will do the assert() runtime check
	
	// some more complex samples
	Vector v;
	int a = v.get(1); // that's ok
	int b = v.get(-1); // that should give a compiler warning
	int c = v.get(a + 1); // that's ok again
	int d = v.get(a - 1); // that should also give a compiler warning
	int e = v.get(argc); // that will do the assert() runtime check
	
	return a+b+c+d+e;
}
