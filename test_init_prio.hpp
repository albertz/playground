template<typename T> struct A {
	static const T a;
};

struct B {
	static const float b;
};

struct C {
	C();
	float c;
};

extern C c;
