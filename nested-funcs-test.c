// gcc -O0 -fnested-functions nested-funcs-test.c -o nested-funcs-test

#include <stdio.h>
#include <string.h>

//#define NESTED_FUNC_ATTR __attribute__ ((__regparm__ (1)))
#define NESTED_FUNC_ATTR

void do_call0(int (*f)());
void do_call2(void (*f)(int,int), int a1, int a2);
int do_callX(int (*f)(int (*g)()), int (*g)());

int foo() {
	int x = 0;
	auto int NESTED_FUNC_ATTR a();
	int NESTED_FUNC_ATTR a() { printf("a, x=%i\n", x); x++; return x; }
	a(); x++; a();
	auto void NESTED_FUNC_ATTR b(int a1, int a2);
	void NESTED_FUNC_ATTR b(int a1, int a2) { printf("b, x=%i\n", x); x++; }
	b(0,0);
	do_call0(a);
	do_call2(b, 0,0);

	auto int NESTED_FUNC_ATTR c(int (*g)());
	int NESTED_FUNC_ATTR c(int (*g)()) { printf("c\n"); return g(); }

	do_callX(c, a);
}

void do_call0(int (*f)()) {
	int r = f();
	printf("call0 ret: %i\n", r);
}

void do_call2(void (*f)(int,int), int a1, int a2) {
	f(a1, a2);
}

int do_callX(int (*f)(int (*g)()), int (*g)()) {
	return f(g);
}

int main() {
	printf("hello\n");
	foo();
}