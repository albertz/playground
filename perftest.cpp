// compile: c++ -pg perftest.cpp
// doesnt work on MacOSX (doesnt produce a gmon.out)

#include <unistd.h>

void test1() {
	for(unsigned long i = 0; i < 100 * 1000 * 1000; ++i) {}
	usleep(1000);
}

void test2() {
	usleep(10 * 1000);
}

int main() {
	test1();
	test2();
}

