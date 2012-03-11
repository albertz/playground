#include <iostream>
using namespace std;
int main() {
	cout << (-7 % 2) << endl;
	float px = 0.2f;
	int x = -3;
	int pxx = (int)px + x;
	unsigned int w = 504;
	cout << pxx << endl;
	pxx %= w;
	cout << pxx << endl;
	if(pxx < 0) pxx += w;
	cout << pxx << endl;
//----
	int y = -3;
	y %= (unsigned int)(504);
	cout << y << endl;
	cout << ((unsigned int)-3 % (unsigned int)(100)) << endl;
}
