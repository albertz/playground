/*
from here: http://www.stanford.edu/~akle/using_openblas.html
see also: https://github.com/xianyi/OpenBLAS

compile:
  c++ blas_test.cpp -lopenblas
or:
  c++ blas_test.cpp -framework vecLib -I/System/Library/Frameworks/vecLib.framework/Headers
 */

#include <cblas.h>
#include <iostream>
#include <vector>

int main()
{
    int n = 10;
    int in_x =1;
    int in_y =1;

    std::vector<double> x(n);
    std::vector<double> y(n);

    double alpha = 10;

    std::fill(x.begin(),x.end(),1.0);
    std::fill(y.begin(),y.end(),2.0);

    cblas_daxpy( n, alpha, &x[0], in_x, &y[0], in_y);

    //Print y 
    for(int j=0;j<n;j++)
        std::cout << y[j] << "\t";

    std::cout << std::endl;
}

