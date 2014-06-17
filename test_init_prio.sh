#!/bin/bash

set -e
CXX=${CXX:-c++}
PREFIX=test_init_prio

for f in a b main; do
	a=${PREFIX}_$f.a
	o=${PREFIX}_$f.o
	cpp=${PREFIX}_$f.cpp
	$CXX -Wall $cpp -c -o $o
	ar rucs $a $o
done

exe=${PREFIX}.exe
objs="${PREFIX}_b.a ${PREFIX}_a.a ${PREFIX}_main.a"
$CXX -Wl,-\( $objs -Wl,-\) -o $exe

./${PREFIX}.exe
