#include <typeinfo>
#include <cstdio>

class Foo {
	virtual ~Foo() {}
};

int main() {
	printf("%p\n", &typeid(Foo));
}

