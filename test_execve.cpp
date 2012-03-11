#include <iostream>
#include <unistd.h>

//     int
//     execve(const char *path, char *const argv[], char *const envp[]);

using namespace std;

int main() {
	char arg1[] = {NULL};
	char* paths[] = { "/usr/bin/sudo" /*, "/sbin/sudo", "/bin/sudo", NULL*/ };
	
	int ret = 0;
	for(char** path = paths; *path != NULL; ++path) {
		cout << "executing " << *path << endl;
		char* argv[] = {NULL /*, NULL, NULL*/};
		char tmp = 'a';
		//argv[0] = *path;
		argv[0] = &tmp + 1000;
		ret = execve(*path, argv, NULL);
	}
	cout << "returning with " << ret << endl;

	return -1;
}

