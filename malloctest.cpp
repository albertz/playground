#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define SIZE 2048
#define LIMIT 10000

int main() {
	unsigned int i = LIMIT + 1;
	bool havenonnull = false;
	while(--i) {
		char* d = (char*) malloc(SIZE);
		for(char* p = d; p < d + SIZE; ++p) {
			if(havenonnull || *p)
				write(1, p, 1);
			havenonnull = *p != 0;
		}
	}
}
