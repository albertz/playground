/* 
ViennaCL

http://viennacl.sourceforge.net/viennacl-manual-current.pdf

compile:
c++ viennacl_test.cpp -std=c++11
*/

#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/linalg/prod.hpp>

template<typename T1, typename T2>
auto operator*(const T1& a, const T2& b) -> decltype(viennacl::linalg::prod(a, b)) {
	return  viennacl::linalg::prod(a, b);
}


typedef float ScalarType;

int main () {
    using namespace viennacl;
    using namespace viennacl::linalg;

    matrix<ScalarType> m (3, 3);
    vector<ScalarType> v (3);

    for (unsigned i = 0; i < std::min (m.size1 (), v.size ()); ++ i) {
        for (unsigned j = 0; j < m.size2 (); ++ j)
            m (i, j) = 3 * i + j;
        v (i) = i;
    }

    std::cout << m * v << std::endl;
    std::cout << prod (m, v) << std::endl;
//    std::cout << prod (v, m) << std::endl;
}

