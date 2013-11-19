

/* 

http://www.boost.org/doc/libs/1_55_0/libs/numeric/ublas/doc/index.htm
https://github.com/uBLAS/ublas/wiki

compile:
c++ ublas_test.cpp
*/

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

int main () {
    using namespace boost::numeric::ublas;
    matrix<double> m (3, 3);
    vector<double> v (3);
    for (unsigned i = 0; i < std::min (m.size1 (), v.size ()); ++ i) {
        for (unsigned j = 0; j < m.size2 (); ++ j)
            m (i, j) = 3 * i + j;
        v (i) = i;
    }

    std::cout << prod (m, v) << std::endl;
    std::cout << prod (v, m) << std::endl;
}

