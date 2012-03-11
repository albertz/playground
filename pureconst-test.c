#include <stdio.h>

int foo_pure() __attribute__ ((pure));
int bar_pure() __attribute__ ((pure));
int foo_const() __attribute__ ((const));
int bar_const() __attribute__ ((const));
int bar2_const(int*) __attribute__ ((const));
int bar2_pure(int*) __attribute__ ((pure));
int bar3_const(int*) __attribute__ ((const));


int barc = 0;

#define COUNTER(func, params, C, extra) int callfunc__ ## func () { \
	int c = 0; \
	int i = 0; \
	for(; i < C; ++i) { \
		c += func params; \
		extra; \
	} \
	return c; \
}

COUNTER(foo_pure, (), 100, {});
COUNTER(bar_pure, (), 100, barc++);
COUNTER(foo_const, (), 100, {});
COUNTER(bar_const, (), 100, barc++);
COUNTER(bar2_const, (&barc), 100, barc++);
COUNTER(bar2_pure, (&barc), 100, barc++);
COUNTER(bar3_const, (&barc), 100, {});

#define CALLFUNC(func) { \
	int r = callfunc__ ## func (); \
	printf(#func " = %i\n", r); \
	barc = 0; \
}

int main() {
	CALLFUNC(foo_pure);
	CALLFUNC(bar_pure);
	CALLFUNC(foo_const);
	CALLFUNC(bar_const);
	CALLFUNC(bar2_const);
	CALLFUNC(bar2_pure);
	CALLFUNC(bar3_const);
}

