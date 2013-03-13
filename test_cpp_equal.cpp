#include <assert.h>

struct T {
	int a, b;
};

int main() {
	T a, b;
	// This does not work: There is no automatic implicit `op==`.
	assert(a == b);
}
