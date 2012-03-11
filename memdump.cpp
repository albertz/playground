#include <cstdio>
#include <unistd.h>
#include <iostream>

int main() {
	size_t c = 0;
	size_t chunksize = 64 * 1024;
	while(true) {
		c++;
		char* ptr = (char*) malloc(chunksize);
		fwrite(ptr, chunksize, 1, stdout);
	}
}

