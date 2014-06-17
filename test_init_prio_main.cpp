#include <stdio.h>
#include "test_init_prio.hpp"

__attribute__((constructor))
static void init() {
	printf("init: %i %i\n", a, b);
}

int main() {}
