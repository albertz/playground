
// http://stackoverflow.com/questions/38140630/protected-member-access-works-only-when-not-getting-its-address

#include <iostream>
using namespace std;

class A {
protected:
	int x;
public:
	A() : x(42) {}
};

class B : public A {
};

class C : public B {
protected:
	typedef B Precursor;

public:
	void foo() {
		//cout << x << endl;  // error: ‘x’ was not declared in this scope
		cout << Precursor::x << endl;
		cout << this->x << endl;
	}

	int get() {
		return Precursor::x;
	}

	int* getPtr() {
		// error: ‘int A::x’ is protected
		// error: within this context
		// error: cannot convert ‘int A::*’ to ‘int*’ in return
		return &Precursor::x;
		//return &this->x;  // this works
	}
};


int main() {
	C obj;
	obj.foo();
	cout << obj.get() << endl;
	cout << obj.getPtr() << endl;
}

