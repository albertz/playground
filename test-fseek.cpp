
// c++ test-fseek.cpp

#include <stdio.h>

#define putc putc_unlocked

int main() {

    FILE* fp = fopen("test-fseek.dummy", "wb");
    flockfile(fp);

    putc('a', fp);
    putc('b', fp);
    putc('c', fp);
    putc('d', fp);

    long old = ftell(fp);
    fseek(fp, 1, SEEK_SET);
    putc('X', fp);
    fseek(fp, old, SEEK_SET);
    putc('Y', fp);

    fwrite("\n\x00", 2, 1, fp);
    fclose(fp);
}
