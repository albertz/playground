#include <iostream>
using namespace std;

int main() {
#if not_defined_macro
	cout << "foo" << endl;
#endif
	cout << "hello" << endl;
	return 0;
}

