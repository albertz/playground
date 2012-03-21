#include <stdint.h>
#include <string>
#include <iostream>
#include <boost/typeof/typeof.hpp>
using namespace std;

struct Color{};
struct CVec{};
struct CustomVar {
	struct Ref{};
	struct WeakRef{};
	Ref getRefCopy() { return Ref(); }
	WeakRef weakRef;
};
struct NullCustomVar : CustomVar {};

template<typename T>
T& operator<<(T& s, const CustomVar::Ref&) { s << "CustomVar::Ref"; return s; }

template<typename T>
T& operator<<(T& s, const CustomVar::WeakRef&) { s << "CustomVar::WeakRef"; return s; }


enum ScriptVarType_t
{
	SVT_BOOL = 0,
	SVT_INT32 = 1,
	SVT_UINT64 = 6,
	SVT_FLOAT = 2,
	SVT_STRING = 3,
	SVT_COLOR = 4,
	SVT_VEC2 = 5,
	SVT_CUSTOM = 20,
	SVT_CustomWeakRefToStatic = 21,
	SVT_CALLBACK,	// Cannot be referenced from XML files directly, only as string
	SVT_DYNAMIC
};

static const ScriptVarType_t SVT_INVALID = ScriptVarType_t(-1);


template<typename T> struct GetType;

template<typename T> struct _GetTypeSimple {
	typedef T type;
	static type defaultValue() { return T(); }
	static const type& constRef(const type& v) { return v; }
};

template<> struct GetType<bool> : _GetTypeSimple<bool> { static const ScriptVarType_t value = SVT_BOOL; };
template<> struct GetType<int32_t> : _GetTypeSimple<int32_t> { static const ScriptVarType_t value = SVT_INT32; };
template<> struct GetType<uint64_t> : _GetTypeSimple<uint64_t> { static const ScriptVarType_t value = SVT_UINT64; };
template<> struct GetType<float> : _GetTypeSimple<float> { static const ScriptVarType_t value = SVT_FLOAT; };
template<> struct GetType<std::string> : _GetTypeSimple<std::string> { static const ScriptVarType_t value = SVT_STRING; };
template<> struct GetType<Color> : _GetTypeSimple<Color> { static const ScriptVarType_t value = SVT_COLOR; };
template<> struct GetType<CVec> : _GetTypeSimple<CVec> { static const ScriptVarType_t value = SVT_VEC2; };
template<> struct GetType<CustomVar::Ref> {
	typedef CustomVar::Ref type;
	static const ScriptVarType_t value = SVT_CUSTOM;
	static type defaultValue() { return NullCustomVar().getRefCopy(); }
	static const type& constRef(const type& v) { return v; }
};


template<typename T>
struct CustomVarWeakRefType {
	typedef CustomVar::WeakRef type;
	static const ScriptVarType_t value = SVT_CustomWeakRefToStatic;
	static CustomVar::Ref defaultValue() { return T().getRefCopy(); }
	static CustomVar::WeakRef constRef(const T& v) { return v.weakRef; }
};

struct StringType : GetType<std::string> {};

template<typename T>
struct SelectType {
	static CustomVarWeakRefType<T>* selectType(const CustomVar&) { return NULL; }

	static StringType* selectType(const char*) { return NULL; }
	static StringType* selectType(char[]) { return NULL; }

	typedef BOOST_TYPEOF(*selectType(*(T*)NULL)) type;
};

template<typename T> struct GetType : SelectType<T>::type {};


template<typename T>
void dump(const T& v) {
	cout << GetType<T>::constRef(v)	<< endl;
}

struct DummyVar : CustomVar {};

int main() {
	dump("hello");
	dump(true);
	dump(42);
	dump(3.14f);
	dump(CustomVar());
	dump(CustomVar::Ref());
	dump(NullCustomVar());
	dump(DummyVar());
}
