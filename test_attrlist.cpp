
#include <iostream>
#include <iomanip>

#include <string>
#include <map>
#include <exception>
#include <sstream>
#include <set>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)



void TrimSpaces(std::string& szLine) {
	size_t n = 0;
	std::string::iterator p;
	for(p = szLine.begin(); p != szLine.end(); p++, n++)
		if(!isspace((unsigned char)*p) || isgraph((unsigned char)*p)) break;
	if(n>0) szLine.erase(0,n);
	
	n = 0;
	std::string::reverse_iterator p2;
	for(p2 = szLine.rbegin(); p2 != szLine.rend(); p2++, n++)
		if(!isspace((unsigned char)*p2) || isgraph((unsigned char)*p2)) break;
	if(n>0) szLine.erase(szLine.size()-n);
}

std::string stringtolower(const std::string& txt) {
	std::string res;
	for(std::string::const_iterator i = txt.begin(); i != txt.end(); i++)
		res += tolower((unsigned char)*i);
	return res;
}


template<typename T>
T from_string(const std::string& s, std::ios_base& (*f)(std::ios_base&), bool& failed) {
	std::istringstream iss(s); T t;
	failed = (iss >> f >> t).fail();
	return t;
}

template<typename T>
T from_string(const std::string& s, std::ios_base& (*f)(std::ios_base&)) {
	std::istringstream iss(s); T t;
	iss >> f >> t;
	return t;
}


template<typename T>
T from_string(const std::string& s, bool& failed) {
	std::istringstream iss(s); T t;
	failed = (iss >> t).fail();
	return t;
}

template<typename T>
T from_string(const std::string& s) {
	std::istringstream iss(s); T t;
	iss >> t;
	return t;
}

template<>
inline bool from_string<bool>(const std::string& s) {
	std::string s1(stringtolower(s));
	TrimSpaces(s1);
	if( s1 == "true" ) return true;
	else if( s1 == "false" ) return false;
	else return ( from_string<int>(s1.c_str()) != 0 );
}


template<typename T>
std::string to_string(T val) {
	std::ostringstream oss;
	oss << val;
	return oss.str();
}

template<> inline std::string to_string<bool>(bool val) {
	if(val) return "true"; else return "false";
}







#define ATTRIBUTES_BEGIN \
AttributeBeginMark attributeBeginMark; \
Attribute* operator[](const std::string& id) { attributeBeginMark.init(); return attributeBeginMark.get(id); } \
AttribMap& attributes() { attributeBeginMark.init(); return attributeBeginMark.get(); }


struct Attribute {
	~Attribute() {}
	std::string id;
	bool isLast() { return id.size() == 0; }
	virtual Attribute* next() { return NULL; }
	virtual std::string asString() const { return ""; }
	virtual void fromString(const std::string& str) {}
	virtual bool isDefault() const { return true; }
};

typedef std::map<std::string,Attribute*> AttribMap;

class AttributeBeginMark {
private:
	bool inited;
	AttribMap attributes;
	Attribute* next() { return (Attribute*)(this + 1); }
	void reg(Attribute* attr) { attributes[attr->id] = attr; }
	void registerAll() {
		Attribute* attr = next();
		while(!attr->isLast()) {
			reg(attr);
			attr = attr->next();
		}
		inited = true;
	}
public:
	AttributeBeginMark() : inited(false) {}
	void init() { if(!inited) registerAll(); }
	Attribute* get(const std::string& id) const { AttribMap::const_iterator f = attributes.find(id); if(f != attributes.end()) return f->second; return NULL; }
	AttribMap& get() { return attributes; }
};

template< typename _T >
struct SpecialPrintableAttr : Attribute {
	_T data;
	virtual std::string asString() const { return to_string<_T>(data); }
	virtual void fromString(const std::string& str) { data = from_string<_T>(str); }
};


template<typename _T>
struct SpecialAttribute : Attribute {
	_T data;
	virtual std::string asString() const { return ""; }
	virtual void fromString(const std::string& str) { data = _T(); }
};

template<> struct SpecialAttribute<bool> : SpecialPrintableAttr<bool> {};
template<> struct SpecialAttribute<int> : SpecialPrintableAttr<int> {};
template<> struct SpecialAttribute<long> : SpecialPrintableAttr<long> {};
template<> struct SpecialAttribute<float> : SpecialPrintableAttr<float> {};
template<> struct SpecialAttribute<double> : SpecialPrintableAttr<double> {};
template<> struct SpecialAttribute<std::string> : SpecialPrintableAttr<std::string> {};



#define ATTRIBUTES_END \
Attribute attributeEndMark;

#define ATTRIBUTE_STRUCT(_T, ID, def) \
struct Attribute_##ID : SpecialAttribute< _T > { \
Attribute_##ID() { id = TOSTRING(ID); data = def; } \
_T& operator()() { return data; } \
virtual Attribute* next() { return (Attribute*)(this + 1); } \
static _T Default() { return def; } \
virtual bool isDefault() const { return data == def; } \
}

#define ATTRIBUTE(_T, ID, def) \
ATTRIBUTE_STRUCT(_T, ID, def) ID;

struct TestAttrList {
	ATTRIBUTES_BEGIN;
	ATTRIBUTE(bool, DetOnOccurrence, false);
	ATTRIBUTE(bool, StoreSubTree, true);
	ATTRIBUTE(bool, StoreWholeString, false);
	ATTRIBUTE(bool, StackDetOnMatch, false);
	ATTRIBUTE(bool, SkipSubErrors, false);
	ATTRIBUTE(int, DetPriority, 0);
	ATTRIBUTES_END;
};


namespace Blub {
	template<long L> struct TypeInfo {
		static const bool Defined = false;
	};

	template<> struct TypeInfo<0> {
		static const bool Defined = true;
		typedef int Type;
	};
	
	template<long L> struct Counter {
		static const long Count = 0;
	};

	template<> struct Counter<0> {
		static const long Count = Counter<1>::Count;
	};
	
	template<> struct Counter<1> {
		static const long Count = 1 + Counter<-2>::Count;		
	};
	template<> struct Counter<-1> {
		static const long Count = 1 + Counter<2>::Count;		
	};
	
	//template<> struct Counter< Counter<0>::Count + 1 > {
	//	static const long Count = 1 + Counter< -Counter<0>::Count - 1 >::Count;
	//};
	//template<> struct Counter< Counter<0>::Count > {
	
	template<bool dummy, long L, typename T> struct Joined {
		static const bool Defined = false;
	};
	
	template<long L, typename T> struct Joined<true, L, T> {
		static const bool Defined = true;
		typedef T Type;
		Joined<true, L - sizeof(T), typename TypeInfo<L - sizeof(T)>::Type> last;
		T val;
	};
	template<typename T> struct Joined<true,0,T> {
		static const bool Defined = true;
		T val;
	};



	
}




struct Test2 {
	
};


int main() {
	using namespace std;
	TestAttrList test;
	
	for(AttribMap::iterator i = test.attributes().begin(); i != test.attributes().end(); ++i) {
		cout << i->first << " (" << i->second << ") = " << i->second->asString() << endl;
	}
}
