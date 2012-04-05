#include <iostream>
#include <string>
using namespace std;

struct B {};
B operator&&(bool a, const B& b) { return B(); }
B f(const std::string& s) { cout << s << endl; }

struct C {
	C(bool) {}
	operator bool() const { return true; }
};
C operator&&(bool a, const C& b) { if(a) return b; return a; }
C g(const std::string& s) { cout << s << endl; }


int main() {
	true && f("f1");
	false && f("f2");
	
	if(true && g("g1")) {}
	if(false && g("g2")) {}
}
