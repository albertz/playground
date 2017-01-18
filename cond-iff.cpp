#include <iostream>

using namespace std;

int foo() {
	cout << "foo" << endl;
	return 1;
}

int bar() {
	cout << "bar" << endl;
	return 2;
}

int main() {
	cout << (0 ? foo() : bar()) << endl;
	cout << (1 ? foo() : bar()) << endl;
	return 0;
}
