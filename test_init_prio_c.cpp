#include <stdio.h>
#include "test_init_prio.hpp"

C::C() {
	printf("init: %f %f\n", A<float>::a, B::b);
	c = B::b;
}

C c;

__attribute__((constructor))
static void c_init() {
	printf("c_init: %f %f\n", A<float>::a, B::b);	
}

