#include <iostream>
using namespace std;

template<float f> void foo() { cout << f << endl; }

int main() {
	foo<42.23f>();
}
