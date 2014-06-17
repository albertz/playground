#include <stdio.h>
#include "test_init_prio.hpp"

C::C() : c(3) {
	printf("init: %f %f\n", A<float>::a, B::b);
}

C c;


