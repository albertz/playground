#include <iostream>
#include <type_traits>
using namespace std;

int x;
int& foo() { return x; }

int main() {
	cout << std::is_same<int, int>::value << endl;
	cout << std::is_same<int, int&>::value << endl;
	cout << std::is_same<int&, decltype(foo())>::value << endl;
}
