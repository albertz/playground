	.section	__TEXT,__text,regular,pure_instructions
	.globl	_callfunc__foo_pure
	.align	4, 0x90
_callfunc__foo_pure:
	pushl	%ebp
	movl	%esp, %ebp
	subl	$8, %esp
	call	_foo_pure
	imull	$100, %eax, %eax
	addl	$8, %esp
	popl	%ebp
	ret

	.globl	_callfunc__bar_pure
	.align	4, 0x90
_callfunc__bar_pure:
	pushl	%ebp
	movl	%esp, %ebp
	pushl	%ebx
	pushl	%edi
	pushl	%esi
	subl	$12, %esp
	call	L1$pb
L1$pb:
	popl	%eax
	movl	%eax, -20(%ebp)
	xorl	%esi, %esi
	movl	$1, %edi
	movl	_barc-L1$pb(%eax), %eax
	movl	%eax, -16(%ebp)
	.align	4, 0x90
LBB1_1:
	movl	-16(%ebp), %eax
	leal	(%eax,%edi), %ebx
	call	_bar_pure
	movl	-20(%ebp), %ecx
	movl	%ebx, _barc-L1$pb(%ecx)
	addl	%eax, %esi
	incl	%edi
	cmpl	$101, %edi
	jne	LBB1_1
	movl	%esi, %eax
	addl	$12, %esp
	popl	%esi
	popl	%edi
	popl	%ebx
	popl	%ebp
	ret

	.globl	_callfunc__foo_const
	.align	4, 0x90
_callfunc__foo_const:
	pushl	%ebp
	movl	%esp, %ebp
	subl	$8, %esp
	call	_foo_const
	imull	$100, %eax, %eax
	addl	$8, %esp
	popl	%ebp
	ret

	.globl	_callfunc__bar_const
	.align	4, 0x90
_callfunc__bar_const:
	pushl	%ebp
	movl	%esp, %ebp
	pushl	%edi
	pushl	%esi
	call	L3$pb
L3$pb:
	popl	%esi
	movl	_barc-L3$pb(%esi), %edi
	addl	$100, %edi
	call	_bar_const
	movl	%edi, _barc-L3$pb(%esi)
	imull	$100, %eax, %eax
	popl	%esi
	popl	%edi
	popl	%ebp
	ret

	.globl	_callfunc__bar2_const
	.align	4, 0x90
_callfunc__bar2_const:
	pushl	%ebp
	movl	%esp, %ebp
	pushl	%edi
	pushl	%esi
	subl	$16, %esp
	call	L4$pb
L4$pb:
	popl	%esi
	movl	_barc-L4$pb(%esi), %edi
	leal	_barc-L4$pb(%esi), %eax
	movl	%eax, (%esp)
	addl	$100, %edi
	call	_bar2_const
	movl	%edi, _barc-L4$pb(%esi)
	imull	$100, %eax, %eax
	addl	$16, %esp
	popl	%esi
	popl	%edi
	popl	%ebp
	ret

	.globl	_callfunc__bar2_pure
	.align	4, 0x90
_callfunc__bar2_pure:
	pushl	%ebp
	movl	%esp, %ebp
	pushl	%ebx
	pushl	%edi
	pushl	%esi
	subl	$12, %esp
	call	L5$pb
L5$pb:
	popl	%esi
	xorl	%edi, %edi
	movl	$1, %ebx
	movl	_barc-L5$pb(%esi), %eax
	movl	%eax, -20(%ebp)
	.align	4, 0x90
LBB5_1:
	leal	_barc-L5$pb(%esi), %ecx
	movl	%ecx, (%esp)
	movl	-20(%ebp), %eax
	leal	(%eax,%ebx), %eax
	movl	%eax, -16(%ebp)
	call	_bar2_pure
	movl	-16(%ebp), %ecx
	movl	%ecx, _barc-L5$pb(%esi)
	addl	%eax, %edi
	incl	%ebx
	cmpl	$101, %ebx
	jne	LBB5_1
	movl	%edi, %eax
	addl	$12, %esp
	popl	%esi
	popl	%edi
	popl	%ebx
	popl	%ebp
	ret

	.globl	_callfunc__bar3_const
	.align	4, 0x90
_callfunc__bar3_const:
	pushl	%ebp
	movl	%esp, %ebp
	subl	$8, %esp
	call	L6$pb
L6$pb:
	popl	%eax
	leal	_barc-L6$pb(%eax), %eax
	movl	%eax, (%esp)
	call	_bar3_const
	imull	$100, %eax, %eax
	addl	$8, %esp
	popl	%ebp
	ret

	.globl	_main
	.align	4, 0x90
_main:
	pushl	%ebp
	movl	%esp, %ebp
	pushl	%ebx
	pushl	%edi
	pushl	%esi
	subl	$12, %esp
	call	L7$pb
L7$pb:
	popl	%esi
	call	_foo_pure
	imull	$100, %eax, %eax
	movl	%eax, 4(%esp)
	leal	L_.str-L7$pb(%esi), %eax
	movl	%eax, (%esp)
	call	_printf
	movl	$0, _barc-L7$pb(%esi)
	xorl	%edi, %edi
	movl	$1, %ebx
	.align	4, 0x90
LBB7_1:
	call	_bar_pure
	addl	%eax, %edi
	movl	%ebx, _barc-L7$pb(%esi)
	incl	%ebx
	cmpl	$101, %ebx
	jne	LBB7_1
	movl	%edi, 4(%esp)
	leal	L_.str1-L7$pb(%esi), %ecx
	movl	%ecx, (%esp)
	call	_printf
	movl	$0, _barc-L7$pb(%esi)
	call	_foo_const
	imull	$100, %eax, %eax
	movl	%eax, 4(%esp)
	leal	L_.str2-L7$pb(%esi), %eax
	movl	%eax, (%esp)
	call	_printf
	call	_bar_const
	movl	$100, _barc-L7$pb(%esi)
	imull	$100, %eax, %eax
	movl	%eax, 4(%esp)
	leal	L_.str3-L7$pb(%esi), %eax
	movl	%eax, (%esp)
	call	_printf
	leal	_barc-L7$pb(%esi), %ecx
	movl	%ecx, (%esp)
	call	_bar2_const
	movl	$100, _barc-L7$pb(%esi)
	imull	$100, %eax, %eax
	movl	%eax, 4(%esp)
	leal	L_.str4-L7$pb(%esi), %eax
	movl	%eax, (%esp)
	call	_printf
	movl	$0, _barc-L7$pb(%esi)
	xorl	%edi, %edi
	movl	$1, %ebx
	.align	4, 0x90
LBB7_3:
	leal	_barc-L7$pb(%esi), %ecx
	movl	%ecx, (%esp)
	call	_bar2_pure
	addl	%eax, %edi
	movl	%ebx, _barc-L7$pb(%esi)
	incl	%ebx
	cmpl	$101, %ebx
	jne	LBB7_3
	movl	%edi, 4(%esp)
	leal	L_.str5-L7$pb(%esi), %ecx
	movl	%ecx, (%esp)
	call	_printf
	movl	$0, _barc-L7$pb(%esi)
	leal	_barc-L7$pb(%esi), %ecx
	movl	%ecx, (%esp)
	call	_bar3_const
	imull	$100, %eax, %eax
	movl	%eax, 4(%esp)
	leal	L_.str6-L7$pb(%esi), %eax
	movl	%eax, (%esp)
	call	_printf
	movl	$0, _barc-L7$pb(%esi)
	xorl	%eax, %eax
	addl	$12, %esp
	popl	%esi
	popl	%edi
	popl	%ebx
	popl	%ebp
	ret

	.globl	_barc
.zerofill __DATA,__common,_barc,4,2
	.section	__TEXT,__cstring,cstring_literals
L_.str:
	.asciz	 "foo_pure = %i\n"

L_.str1:
	.asciz	 "bar_pure = %i\n"

L_.str2:
	.asciz	 "foo_const = %i\n"

L_.str3:
	.asciz	 "bar_const = %i\n"

	.align	4
L_.str4:
	.asciz	 "bar2_const = %i\n"

L_.str5:
	.asciz	 "bar2_pure = %i\n"

	.align	4
L_.str6:
	.asciz	 "bar3_const = %i\n"


.subsections_via_symbols
