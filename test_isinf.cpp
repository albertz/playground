#include <iostream>
#include <cmath>
#include <stdint.h>
#include <limits>

using std::cout;
using std::endl;

static bool my_isnan1(double val) {
	union { double f; uint64_t x; } u = { val };
	return (u.x << 1) > 0x7ff0000000000000u;
}

static bool my_isnan2(double val) {
	return val != val;
}

static bool my_isnan3(double val) {
	volatile double a = val;
	return a != a;
}

static void check_nan(double val) {
	cout << "value: " << val << endl;
	cout << std::isnan(val) << endl;
	cout << ::isnan(val) << endl;
	cout << __isnan(val) << endl;
	cout << my_isnan1(val) << endl;
	cout << my_isnan2(val) << endl;
	cout << my_isnan3(val) << endl;
}

int main() {
	double a = std::log(0.0);
	cout << std::isinf(a) << endl;

	double b = std::sqrt(-1.0);
	check_nan(b);

	double c = std::numeric_limits<double>::quiet_NaN();
	check_nan(c);

	return 0;
}

