#include <stdio.h>


int main() {
	static char str[254];
	int i;
	for(i = 0; i < 254; i++) {
		printf("%i: %i\n", i, str[i]);
	}

	return 0;
}
