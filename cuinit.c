/* cc cuinit.c -I/usr/local/cuda-11.7/include -L/usr/local/cuda-11.7/lib64 -lcuda -ocuinit.bin
*/

#include <cuda.h>

int main() {
	return cuInit(0);
}

