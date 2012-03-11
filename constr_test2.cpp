// demonstrating non-determinism of constructor execution order

// http://stackoverflow.com/questions/8433484/c-static-initialization-vs-attribute-constructor/8433525#8433525
// http://llvm.org/bugs/show_bug.cgi?id=11521

#include <stdio.h>

struct Foo { Foo() { printf("foo\n"); } };
static Foo foo;

__attribute__((constructor)) static void _bar() { printf("bar\n"); }

int main() {}
