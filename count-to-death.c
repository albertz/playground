#include <stdio.h>
#include <signal.h>

long long int a = 0;

void handle() {
	printf("%lld\n", a);
}

int main() {
	signal(SIGUSR1, handle);
	while(1) {
		a++;
	}
}

