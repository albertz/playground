#include <iostream>

class Base {
public:
	virtual void func() {}
};

class Derived : public Base {
public:
	virtual void func() {}
};

bool isSameType(const Base& a, const Base& b) {
	using namespace std;
	cout << "&a=" << &a << ",&b=" << &b;
	cout << ",*&a=" << *(void**)&a << ",*&b=" << *(void**)&b;
	cout << " ";
	return *(void**)&a == *(void**)&b;
}


int main() {
	Base a1, a2;
	Derived b1, b2;

	using namespace std;
	cout << "a1 == a2: " << isSameType(a1, a2) << endl;
	cout << "a1 == b1: " << isSameType(a1, b1) << endl;
	cout << "a2 == b2: " << isSameType(a2, b2) << endl;
	cout << "b1 == b2: " << isSameType(b1, b2) << endl;

	cout << "sizeof(Base) = " << sizeof(Base) << endl;
}


