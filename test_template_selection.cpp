#include <boost/type_traits.hpp>

enum {
	SVT_INT,
	SVT_FLOAT,
	SVT_BASEOBJ,
	SVT_CUSTOMVAR
};

struct BaseObject {};
struct CustomVar {};

template<typename T> struct GetType;
template<> struct GetType<int> { static const int value = SVT_INT; };
template<> struct GetType<float> { static const int value = SVT_FLOAT; };
template<> struct GetType<BaseObject> { static const int value = SVT_BASEOBJ; };

template<bool> struct GetType_BaseCustomVar;
template<> struct GetType_BaseCustomVar<true> {
	struct Type { static const int value = SVT_CUSTOMVAR; };
};
template<typename T> struct GetType : GetType_BaseCustomVar<boost::is_base_of<CustomVar,T>::value>::Type {};

struct ScriptVar_t;
template<typename T> T CastScriptVarConst(const ScriptVar_t& s);

struct ScriptVar_t {
	operator int() const { return 0; }
	operator float() const { return 0.0f; }
	operator BaseObject() const { return BaseObject(); }
	template<typename T> T* as() const { return NULL; }

	template <typename T> T castConst() const { return CastScriptVarConst<T>(*this); }
};

template<typename T>
T _CastScriptVarConst(const ScriptVar_t& s, T*, typename boost::enable_if_c<(GetType<T>::value < SVT_BASEOBJ), T>::type*) {
	return (T) s;
}

template<typename T>
T _CastScriptVarConst(const ScriptVar_t& s, T*, typename boost::enable_if_c<boost::is_base_of<CustomVar,T>::value, T>::type*) {
	return *s.as<T>();
}

template<typename T> T CastScriptVarConst(const ScriptVar_t& s) { return _CastScriptVarConst(s, (T*)NULL, (T*)NULL); }

/*
template<typename T>
typename boost::enable_if_c<(GetType<T>::value < SVT_BASEOBJ), T>::type
CastScriptVarConst(const ScriptVar_t& s) {
    return (T) s;
}

template<typename T>
typename boost::enable_if_c<!(GetType<T>::value < SVT_BASEOBJ)
&& boost::is_base_of<CustomVar,T>::value,T>::type
CastScriptVarConst(const ScriptVar_t& s) {
    return *s.as<T>();
}
*/

/*
template<typename T> T CastScriptVarConst(const ScriptVar_t& s);

template<bool> struct CastScriptVar1;
template<typename T> struct CastScriptVar1_IsSimpleType {
    static const bool value = GetType<T>::value < SVT_BASEOBJ;
};
template<> struct CastScriptVar1<true> {
    template<typename T> static T castConst(const ScriptVar_t& s, const T&) { return (T) s; }
};

template<bool> struct CastScriptVar2;
template<typename T> struct CastScriptVar2_IsCustomVar {
    static const bool value = boost::is_base_of<CustomVar,T>::value;
};
template<> struct CastScriptVar2<true> {
    template<typename T> static T castConst(const ScriptVar_t& s, const T&) { return *s.as<T>(); }
};

template<> struct CastScriptVar1<false> {
    template<typename T> static T castConst(const ScriptVar_t& s, const T&) {
        return CastScriptVar2<CastScriptVar2_IsCustomVar<T>::value>::castConst(s, T());
    }
};
template<typename T> T CastScriptVarConst(const ScriptVar_t& s) {
    return CastScriptVar1<CastScriptVar1_IsSimpleType<T>::value>::castConst(s, T());
}
*/

int main() {
	ScriptVar_t v;
	v.castConst<int>();
	v.castConst<CustomVar>();
}
