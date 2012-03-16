#include <iostream>
using namespace std;

int c = 0;

struct CommonBase {
	int x;
	CommonBase() : x(++c) {}
};

struct A : virtual CommonBase {};
struct B : virtual CommonBase {};

struct C : A, B {};

int main() {
	C c;
	cout << c.x << endl;
	cout << ((A&)c).x << endl;
	cout << ((B&)c).x << endl;
}

