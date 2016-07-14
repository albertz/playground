#include <iostream>
#include <type_traits>
using namespace std;

int x;
int& foo() { return x; }
const int& bar() { return x; }

int main() {
	cout << std::is_same<int, int>::value << endl;
	cout << std::is_same<int, int&>::value << endl;
	cout << std::is_same<int&, decltype(foo())>::value << endl;
	cout << std::is_same<const int&, decltype(bar())>::value << endl;
}
