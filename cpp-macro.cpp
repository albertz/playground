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

	return 0;
}

