#include <iostream>
using namespace std;

struct A {
	A() { cout << "A()" << endl; }
	A(const A&) { cout << "A(const A&)" << endl; }
	A& operator=(const A&) { cout << "A::op=()" << endl; return *this; }
};

struct B : A {
	B() { cout << "B()" << endl; }
	B(const B&) { cout << "B(const B&)" << endl; }
	B& operator=(const B&) { cout << "B::op=()" << endl; return *this; }
};

int main() {
	cout << "* init.." << endl;
	B a, b;
	
	cout << "* a = b" << endl;
	a = b;
	
	cout << "* copy" << endl;
	B c(a);
	
	cout << "* end" << endl;
}
