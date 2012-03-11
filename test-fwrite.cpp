#include <cstdio>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace std;

int main() {
	int fd = open("test-fwrite.data", O_RDWR);
	if(fd < 0) {
		cerr <<  "failed to open file " << endl;
		return 1;
	}
	
	cout << sizeof(long) << " " << sizeof(size_t) << endl;

	FILE* file = fdopen(fd, "w+b");

	//ftell64(file);

	fwrite("foo", 3, 1, file);
	fclose(file);

	return 0;
}

