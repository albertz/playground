#include <iostream>
using namespace std;

template<typename T>
struct Type {};

template<> struct Type<float> {
	static constexpr float max = 1;
};

void dump(float x) __attribute__((noinline));
void dump(float x) {
	cout << x << endl;
}

void f1(float x) { dump(x); }
void f2(const float& x) { dump(x); }

int main() {
	int a = Type<float>::max;
	dump(a);
	f1(Type<float>::max);
//	f2(Type<float>::max); // requires symbol
	
}

