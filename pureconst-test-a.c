#include <stdio.h>

extern int barc;

int foo_pure() {
	static int c = 0;
	printf("foo_pure %i\n", c);
	c++;
	return 1;
}

int bar_pure() {
	static int c = 0;
	printf("bar_pure %i\n", c);
	c++;
	return barc;
}

int foo_const() {
	static int c = 0;
	printf("foo_const %i\n", c);
	c++;
	return 1;	
}

int bar_const() {
	static int c = 0;
	printf("bar_const %i\n", c);
	c++;
	return barc;
}

int bar2_const(int* i) {
	static int c = 0;
	printf("bar2_const %i\n", c);
	c++;
	return *i;
}

int bar2_pure(int* i) {
	static int c = 0;
	printf("bar2_pure %i\n", c);
	c++;
	return *i;
}

int bar3_const(int* i) {
	static int c = 0;
	printf("bar3_const %i\n", c);
	c++;
	(*i)++;
	return *i;
}
