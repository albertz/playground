#!/bin/bash

PREFIX=test_init_prio

for f in a b main; do
	$CXX ${PREFIX}_$f.cpp -c -o ${PREFIX}_$f.o
	ar rucs ${PREFIX}_$f.a ${PREFIX}_$f.o 
done

$CXX ${PREFIX}_b.a ${PREFIX}_a.a ${PREFIX}_main.a -o ${PREFIX}.exe

./${PREFIX}.exe
