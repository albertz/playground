#include <iostream>
#include <string>

using namespace std;

int main() {
	int input;
	while((input = cin.get()) > 0) {
		char c = (char)input;
		string o;
		switch(c) {
			case 'g': o = "gh"; break;
			case 'i': o = "ee"; break;
			case 'a': o = "ah"; break;
			case 'u': o = "uu"; break;
			case 'e': o = "eh"; break;
			default: o = c;
		}
		cout << o;
		cout.flush();
	}
}
