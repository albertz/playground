#include <iostream>
using namespace std;

int a() { return 2; }
int b() { return 3; }


int main() {
	cout << a() << " " << b() << endl;
	
#define a b
#define b a
	cout << a() << " " << b() << endl;

#undef a
#undef b
#define b a
#define a b
	cout << a() << " " << b() << endl;

#undef a
#undef b
#define b a
	cout << a() << " " << b() << endl;

#undef b
#define b() a()
#define a() b()
	cout << a() << " " << b() << endl;

#undef a
#undef b
#define a() a()
	cout << a() << endl;

#undef a
#define a(x) (a() + x)
	cout << a(a(0)) << endl;

	return 0;
}

