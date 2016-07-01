
#include <iostream>
using namespace std;

class A {
protected:
	typedef int FT;
	FT x;
public:
	A() : x(42) {}
};

class B : public A {
};

class C : public B {
protected:
	typedef int FT;
	typedef B Precursor;

public:
	void foo() {
		//cout << x << endl;  // error: ‘x’ was not declared in this scope
		cout << Precursor::x << endl;
		cout << this->x << endl;
	}

	FT get() {
		return Precursor::x;
	}

	FT* getPtr() {
		// error: ‘A<int>::FT A<int>::x’ is protected
		// error: within this context
		// error: cannot convert ‘A<int>::FT A<int>::* {aka int A<int>::*}’ to ‘C<int>::FT* {aka int*}’ in return
		return &Precursor::x;
		//return &this->x;
	}
};


int main() {
	C obj;
	obj.foo();
	cout << obj.get() << endl;
	cout << obj.getPtr() << endl;
}

