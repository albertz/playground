#include <iostream>
#include <string>
using namespace std;

struct A {
	A(string s) { cout << s << endl; }
};

struct A2 {
	A2() { cout << (long)this << endl; }
};

struct B {
	void test1(string prefix) {
		A a(prefix + ":1");
		A2 a2;
	}
	void test2(string prefix) {
		static A a(prefix + ":2");
		static A2 a2;
	}
	static void test3(string prefix) {
		A a(prefix + ":3");
		A2 a2;
	}
	static void test4(string prefix) {
		static A a(prefix + ":4");
		static A2 a2;
	}

	void test(string prefix) {
		test1(prefix);
		test2(prefix);
		test3(prefix);
		test4(prefix);
	}
};

void test(string prefix) {
	B b1;
	b1.test(prefix + "-1");
	static B b2;
	b2.test(prefix + "-2");
}

struct C {
	C() { test("C"); }
};
static C c;

// ----

struct Foo {
	Foo();
};

struct S {
	bool inited;
	S() : inited(false) {
		cout << "S()" << endl;
		//Foo foo; // this causes a runtime error
		inited = true;
		cout << "S() end" << endl;
	}
	static S* get() {
		static S s;
		return &s;
	}
};

Foo::Foo() {
	cout << "Foo(" << (long)this << "):" << endl;
	cout << "Foo(): S.inited: " << S::get()->inited << endl;
}

static Foo foo;

// ----

int main() {
	test("main");
}
