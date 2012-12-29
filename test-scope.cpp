
#include <iostream>
#include <string>
using namespace std;

struct Scope {
	string name;
	Scope(const string& n) : name(n) { cout << "Scope" << name << endl; }
	~Scope() { cout << "~Scope" << name << endl; }
};

struct DoubleScope {
	Scope scope1, scope2;
	DoubleScope() : scope1("_d1"), scope2("_d2") { cout << "DoubleScope" << endl; }
	~DoubleScope() { cout << "~DoubleScope" << endl; }
};

static int bar() {
	Scope scope1("_bar1");
	return 0;
}

static int foo() {
	Scope scope1("_foo1");
	{
		DoubleScope dscope;
		Scope scope2("_foo2");
	}
	Scope scope2("_foo2");
	return bar();
}

int main() {
	Scope scope1("1");
	Scope scope2("2");

	{
		Scope scope3("3");
		foo();
	}

	Scope scope4("4");
	return foo();
}
