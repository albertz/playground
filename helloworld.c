// https://blog.sunfishcode.online/bugs-in-hello-world/
// https://news.ycombinator.com/item?id=30611367

#include <stdio.h>

int main() {
  if(printf("hello world\n") < 0)
    return 1;
  if(fflush(stdout) < 0)
    return 1;
  return 0;
}

