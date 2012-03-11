//
//  constr_test.cpp
//
//  Created by Albert Zeyer on 08.12.11.
//  Copyright (c) 2011 Albert Zeyer. All rights reserved.
//

// demonstrating non-determinism of constructor execution order

// http://stackoverflow.com/questions/8433484/c-static-initialization-vs-attribute-constructor/8433525#8433525
// http://llvm.org/bugs/show_bug.cgi?id=11521

#include <string>
#include <random>
#include <iostream>

std::mt19937 rnd;

static void printRndState() {
	struct RND {
		uint32_t __x_[rnd.state_size];
		uint32_t __i_;
	};
	for(int i = 0; i < rnd.state_size && i < 6; ++i)
		printf("x[%i] = %u\n", i, ((RND*)&rnd)->__x_[i]);
}

__attribute__((constructor))
static void _rand_engine__init() {
	uint32_t n = (uint32_t)time(NULL);
	n = (n << 16) + (n >> 16);
	printf("seeded with %u\n", n);
	rnd.seed(n);
	printf("rnd state:\n");
	printRndState();
}

int main() {
	printf("main; rnd state:\n");
	printRndState();
	for(int i = 0; i < 6; ++i)
		printf("rnd: %i\n", rnd());
}

