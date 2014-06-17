#include <stdio.h>
#include "test_init_prio.hpp"

__attribute__((constructor))
static void init() {
	printf("init: %f %f\n", A<float>::a, B::b);
}

int main() {}
