#include <stdio.h>

struct Foo { Foo() { printf("foo\n"); } };
static Foo foo;

__attribute__((constructor)) static void _bar() { printf("bar\n"); }

int main() {}
