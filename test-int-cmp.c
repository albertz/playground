// discussion: http://stackoverflow.com/questions/11129368/strange-c-integer-inequality-comparison-result

#include <limits.h>
#include <stdio.h>
int main() {
    long ival = 0;
    printf("ival: %li, min: %i, max: %i, too big: %i, too small: %i\n",
           ival, INT_MIN, INT_MAX, ival > INT_MAX, ival < INT_MIN);

// --- more ---
	
/*  int ival2 = 0;
    printf("ival2: %i, min: %i, max: %i, too big: %i, too small: %i\n",
           ival2, INT_MIN, INT_MAX, ival2 > INT_MAX, ival2 < INT_MIN);	

    long long ival3 = 0;
    printf("ival3: %lli, min: %i, max: %i, too big: %i, too small: %i\n",
           ival3, INT_MIN, INT_MAX, ival3 > INT_MAX, ival3 < INT_MIN);	*/
}

// /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang -arch armv7 -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS5.1.sdk test-int-cmp.c
// -> output on iOS:
//  ival: 0, min: -2147483648, max: 2147483647, too big: 0, too small: 1
