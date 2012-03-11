#include <iostream>
using namespace std;

template<typename t> int A(int j) { return t::C + j; }
template<int i> struct B { static const int C = i; };


int main() {
//	int res = A<B< 4>>(2) >> (1);
	int res1 = A<B< 4> >(2) >> (1);
	int res2 = A<B< 4>>(2) > > (1);
//	int res3 = A<B< 4> >(2) > > (1);
//	cout << res << endl;
	cout << res1 << endl;
	cout << res2 << endl;
//	cout << res3 << endl;
}

