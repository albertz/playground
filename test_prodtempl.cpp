// fails: g++-4.9 test_prodtempl.cpp -o tmp -std=c++11
// works: clang++ test_prodtempl.cpp -o tmp -std=c++11

namespace X {
	template<typename T> struct Mat{};
	template<typename T> struct MatExpr {};

	template<typename T>
	MatExpr<T> prod(Mat<T> const& A, Mat<T> const& B) { return MatExpr<T>(); }
};

struct Mat2 {};

template<typename T>
X::Mat<T> prod(X::Mat<T> const& A, Mat2 const& B) { return X::Mat<T>(); }


template<typename T1, typename T2>
auto operator*(const T1& a, const T2& b) -> decltype(X::prod(a, b)) {
	return X::prod(a, b);
}

template<typename T1, typename T2>
auto operator*(const T1& a, const T2& b) -> decltype(prod(a, b)) {
	return prod(a, b);
}



int main() {}
