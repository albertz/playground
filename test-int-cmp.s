	.section	__TEXT,__text,regular,pure_instructions
	.section	__TEXT,__textcoal_nt,coalesced,pure_instructions
	.section	__TEXT,__const_coal,coalesced
	.section	__TEXT,__picsymbolstub4,symbol_stubs,none,16
	.section	__TEXT,__StaticInit,regular,pure_instructions
	.syntax unified
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.align	2
	.code	16
	.thumb_func	_main
_main:
	push	{r7, lr}
	mov	r7, sp
	sub	sp, #20
	movw	r0, #65535
	movt	r0, #32767
	movs	r1, #0
	movt	r1, #0
	str	r1, [sp, #16]
	str	r1, [sp, #12]
	ldr	r1, [sp, #12]
	ldr	r2, [sp, #12]
	cmp	r2, r0
	movw	r0, #0
	it	gt
	movgt	r0, #1
	and	r0, r0, #1
	ldr	r2, [sp, #12]
	cmn.w	r2, #-2147483648
	movw	r2, #0
	it	lt
	movlt	r2, #1
	and	r2, r2, #1
	mov	r3, sp
	str	r2, [r3, #4]
	str	r0, [r3]
	mov.w	r2, #-2147483648
	mvn	r3, #-2147483648
	movw	r0, :lower16:(L_.str-(LPC0_0+4))
	movt	r0, :upper16:(L_.str-(LPC0_0+4))
LPC0_0:
	add	r0, pc
	blx	_printf
	ldr	r1, [sp, #16]
	str	r0, [sp, #8]
	mov	r0, r1
	add	sp, #20
	pop	{r7, pc}

	.section	__TEXT,__cstring,cstring_literals
L_.str:
	.asciz	 "ival: %li, min: %i, max: %i, too big: %i, too small: %i\n"


.subsections_via_symbols
