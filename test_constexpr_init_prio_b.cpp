#include <stdio.h>
#include "test_constexpr_init_prio.hpp"

__attribute__((noinline))
static void dump(const int* a, const int* b) {
	printf("init: %i %i %p\n", *a, *b, a);
}

__attribute__((constructor))
static void init() {
	dump(&S::a, &S::b);
}

constexpr int S::b;
