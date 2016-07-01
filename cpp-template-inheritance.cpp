
#include <iostream>
using namespace std;

template<typename F>
class A {
protected:
	typedef F FT;
	FT x;
public:
	A() : x(42) {}
};

template<typename F>
class B : public A<F> {
};

template<typename F>
class C : public B<F> {
protected:
	typedef F FT;
	typedef B<F> Precursor;

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
		//return &Precursor::x;
		return &this->x;
	}
};


int main() {
	C<int> obj;
	obj.foo();
	cout << obj.get() << endl;
	cout << obj.getPtr() << endl;
}

