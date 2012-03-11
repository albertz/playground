	.cstring
LC0:
	.ascii "%p\12\0"
	.text
	.align 4,0x90
.globl _main
_main:
LFB27:
	pushl	%ebp
LCFI0:
	movl	%esp, %ebp
LCFI1:
	pushl	%ebx
LCFI2:
	subl	$20, %esp
LCFI3:
	call	L3
"L00000000001$pb":
L3:
	popl	%ebx
	movl	L__ZTI3Foo$non_lazy_ptr-"L00000000001$pb"(%ebx), %eax
	movl	%eax, 4(%esp)
	leal	LC0-"L00000000001$pb"(%ebx), %eax
	movl	%eax, (%esp)
	call	_printf
	xorl	%eax, %eax
	addl	$20, %esp
	popl	%ebx
	leave
	ret
LFE27:
.globl __ZTI3Foo
	.weak_definition __ZTI3Foo
	.section __DATA,__const_coal,coalesced
	.align 2
__ZTI3Foo:
	.long	__ZTVN10__cxxabiv117__class_type_infoE+8
	.long	__ZTS3Foo
.globl __ZTS3Foo
	.weak_definition __ZTS3Foo
	.section __TEXT,__const_coal,coalesced
__ZTS3Foo:
	.ascii "3Foo\0"
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame1:
	.set L$set$0,LECIE1-LSCIE1
	.long L$set$0
LSCIE1:
	.long	0x0
	.byte	0x1
	.ascii "zPR\0"
	.byte	0x1
	.byte	0x7c
	.byte	0x8
	.byte	0x6
	.byte	0x9b
	.long	L___gxx_personality_v0$non_lazy_ptr-.
	.byte	0x10
	.byte	0xc
	.byte	0x5
	.byte	0x4
	.byte	0x88
	.byte	0x1
	.align 2
LECIE1:
.globl _main.eh
_main.eh:
LSFDE1:
	.set L$set$1,LEFDE1-LASFDE1
	.long L$set$1
LASFDE1:
	.long	LASFDE1-EH_frame1
	.long	LFB27-.
	.set L$set$2,LFE27-LFB27
	.long L$set$2
	.byte	0x0
	.byte	0x4
	.set L$set$3,LCFI0-LFB27
	.long L$set$3
	.byte	0xe
	.byte	0x8
	.byte	0x84
	.byte	0x2
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0xd
	.byte	0x4
	.byte	0x4
	.set L$set$5,LCFI3-LCFI1
	.long L$set$5
	.byte	0x83
	.byte	0x3
	.align 2
LEFDE1:
	.section __IMPORT,__pointers,non_lazy_symbol_pointers
L___gxx_personality_v0$non_lazy_ptr:
	.indirect_symbol ___gxx_personality_v0
	.long	0
L__ZTI3Foo$non_lazy_ptr:
	.indirect_symbol __ZTI3Foo
	.long	0
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
