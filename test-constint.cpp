#include <iostream>
#include <unistd.h>
#include <sys/mman.h>
#include <cerrno>

using namespace std;

int main() {
	(int&)(const int&)5 = 0;
	const int& i = 5;
	(int&)i = 6;
	cout << i << " ; " << 5 << endl;

	long pagesize = sysconf(_SC_PAGE_SIZE);
	const char* abc = "abc";

	char* p = (char*)abc;
	p -= (unsigned long)p % pagesize;
	if(mprotect(p, pagesize, PROT_READ|PROT_WRITE)) {
		cout << "mprotect err: " << errno << endl;
		return 0;
	}

	((char*)abc)[0] = 'x';
	cout << "abc" << endl;
}
