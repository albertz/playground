#include <boost/type_traits/is_base_of.hpp>
using namespace boost;

struct A {};

template<typename T> struct GetType;


template<> struct GetType<bool> { typedef bool type; static const int value = 1; };
template<> struct GetType<int> { typedef int type; static const int value = 2; };

template<bool> struct GetType_BaseA;
template<> struct GetType_BaseA<true> {
	struct Type { typedef A type; static const int value = 3; };
};

template<typename T> struct GetType : GetType_BaseA<is_base_of<A,T>::value>::Type {};



//template<> struct GetType<A> { typedef A type; static const int value = 3; };

struct B : A {};

#include <iostream>
int main() {
	using namespace std;
	cout << GetType<bool>::value << endl;
	cout << GetType<int>::value << endl;
	cout << GetType<A>::value << endl;
	cout << GetType<B>::value << endl;
//	cout << GetType<char>::value << endl;
}
