struct A {
	template<typename T> void foo() {}
};

template<typename X, typename Y>
void foo() { X().template foo<Y>(); }

int main() {
	foo<A,int>();
}
