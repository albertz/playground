#!/bin/bash

set -e
CXX=${CXX:-c++}
LD=$CXX
if [ "$(uname)" != "Darwin" ]; then
	LD_START_GROUP=-Wl,-\(
	LD_END_GROUP=-Wl,-\)
fi

PREFIX=test_constexpr_init_prio

for f in a b main; do
	a=${PREFIX}_$f.a
	o=${PREFIX}_$f.o
	cpp=${PREFIX}_$f.cpp
	$CXX -O0 -std=c++0x -Wall $cpp -c -o $o
	ar rucs $a $o
done

exe=${PREFIX}.exe
objs="${PREFIX}_b.a ${PREFIX}_main.a ${PREFIX}_a.a"
$CXX -O0 $LD_START_GROUP $objs $LD_END_GROUP -o $exe

./${PREFIX}.exe
