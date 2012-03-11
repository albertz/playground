// g++ functor_demo.cpp -c -S

// With this set to `inline`, G++ will inline both examples.
// With this empty, G++ will inline none of them.
#define OPT_INLINE inline

OPT_INLINE int f1(int) { return 42; }

template< int (*f)(int) >
void do_sth1() { f(0); }

void test1() {
    do_sth1<f1>();
}

// ----- vs -----

struct F2 {
	OPT_INLINE int operator()(int) { return 42; }
};

template< typename T >
void do_sth2(T f) { f(0); }

void test2() {
    do_sth2(F2());
}

// often, you must write some wrapper when you have a function

struct F1_wrap {
	OPT_INLINE int operator()(int x) { return f1(x); }
};

void test3() {
    do_sth2(F1_wrap());
}

