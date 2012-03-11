	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z5test1v
	.align	4, 0x90
__Z5test1v:
Leh_func_begin1:
	pushq	%rbp
Ltmp0:
	movq	%rsp, %rbp
Ltmp1:
	callq	__Z7do_sth1IXadL_Z2f1iEEEvv
	popq	%rbp
	ret
Leh_func_end1:

	.section	__TEXT,__textcoal_nt,coalesced,pure_instructions
	.globl	__Z2f1i
.weak_definition __Z2f1i
	.align	4, 0x90
__Z2f1i:
Leh_func_begin2:
	pushq	%rbp
Ltmp2:
	movq	%rsp, %rbp
Ltmp3:
	movl	%edi, -4(%rbp)
	movl	$42, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -8(%rbp)
	movl	-8(%rbp), %eax
	popq	%rbp
	ret
Leh_func_end2:

	.globl	__ZN2F2clEi
.weak_definition __ZN2F2clEi
	.align	1, 0x90
__ZN2F2clEi:
Leh_func_begin3:
	pushq	%rbp
Ltmp4:
	movq	%rsp, %rbp
Ltmp5:
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	$42, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	popq	%rbp
	ret
Leh_func_end3:

	.globl	__ZN7F1_wrapclEi
.weak_definition __ZN7F1_wrapclEi
	.align	1, 0x90
__ZN7F1_wrapclEi:
Leh_func_begin4:
	pushq	%rbp
Ltmp6:
	movq	%rsp, %rbp
Ltmp7:
	subq	$32, %rsp
Ltmp8:
	movq	%rdi, %rax
	movq	%rax, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, %edi
	callq	__Z2f1i
	movl	%eax, %ecx
	movl	%ecx, -20(%rbp)
	movl	-20(%rbp), %ecx
	movl	%ecx, -16(%rbp)
	movl	-16(%rbp), %eax
	addq	$32, %rsp
	popq	%rbp
	ret
Leh_func_end4:

	.globl	__Z7do_sth1IXadL_Z2f1iEEEvv
.weak_definition __Z7do_sth1IXadL_Z2f1iEEEvv
	.align	4, 0x90
__Z7do_sth1IXadL_Z2f1iEEEvv:
Leh_func_begin5:
	pushq	%rbp
Ltmp9:
	movq	%rsp, %rbp
Ltmp10:
	movl	$0, %eax
	movl	%eax, %edi
	callq	__Z2f1i
	popq	%rbp
	ret
Leh_func_end5:

	.globl	__Z7do_sth2I2F2EvT_
.weak_definition __Z7do_sth2I2F2EvT_
	.align	4, 0x90
__Z7do_sth2I2F2EvT_:
Leh_func_begin6:
	pushq	%rbp
Ltmp11:
	movq	%rsp, %rbp
Ltmp12:
	leaq	16(%rbp), %rax
	xorl	%ecx, %ecx
	movq	%rax, %rdi
	movl	%ecx, %esi
	callq	__ZN2F2clEi
	popq	%rbp
	ret
Leh_func_end6:

	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z5test2v
	.align	4, 0x90
__Z5test2v:
Leh_func_begin7:
	pushq	%rbp
Ltmp13:
	movq	%rsp, %rbp
Ltmp14:
	subq	$32, %rsp
Ltmp15:
	movb	$0, -16(%rbp)
	movb	-8(%rbp), %al
	movq	%rsp, %rcx
	movb	%al, (%rcx)
	callq	__Z7do_sth2I2F2EvT_
	addq	$32, %rsp
	popq	%rbp
	ret
Leh_func_end7:

	.section	__TEXT,__textcoal_nt,coalesced,pure_instructions
	.globl	__Z7do_sth2I7F1_wrapEvT_
.weak_definition __Z7do_sth2I7F1_wrapEvT_
	.align	4, 0x90
__Z7do_sth2I7F1_wrapEvT_:
Leh_func_begin8:
	pushq	%rbp
Ltmp16:
	movq	%rsp, %rbp
Ltmp17:
	leaq	16(%rbp), %rax
	xorl	%ecx, %ecx
	movq	%rax, %rdi
	movl	%ecx, %esi
	callq	__ZN7F1_wrapclEi
	popq	%rbp
	ret
Leh_func_end8:

	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z5test3v
	.align	4, 0x90
__Z5test3v:
Leh_func_begin9:
	pushq	%rbp
Ltmp18:
	movq	%rsp, %rbp
Ltmp19:
	subq	$32, %rsp
Ltmp20:
	movb	$0, -16(%rbp)
	movb	-8(%rbp), %al
	movq	%rsp, %rcx
	movb	%al, (%rcx)
	callq	__Z7do_sth2I7F1_wrapEvT_
	addq	$32, %rsp
	popq	%rbp
	ret
Leh_func_end9:

	.section	__TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame0:
Lsection_eh_frame:
Leh_frame_common:
Lset0 = Leh_frame_common_end-Leh_frame_common_begin
	.long	Lset0
Leh_frame_common_begin:
	.long	0
	.byte	1
	.asciz	 "zR"
	.byte	1
	.byte	120
	.byte	16
	.byte	1
	.byte	16
	.byte	12
	.byte	7
	.byte	8
	.byte	144
	.byte	1
	.align	3
Leh_frame_common_end:
	.globl	__Z5test1v.eh
__Z5test1v.eh:
Lset1 = Leh_frame_end1-Leh_frame_begin1
	.long	Lset1
Leh_frame_begin1:
Lset2 = Leh_frame_begin1-Leh_frame_common
	.long	Lset2
Ltmp21:
	.quad	Leh_func_begin1-Ltmp21
Lset3 = Leh_func_end1-Leh_func_begin1
	.quad	Lset3
	.byte	0
	.byte	4
Lset4 = Ltmp0-Leh_func_begin1
	.long	Lset4
	.byte	14
	.byte	16
	.byte	134
	.byte	2
	.byte	4
Lset5 = Ltmp1-Ltmp0
	.long	Lset5
	.byte	13
	.byte	6
	.align	3
Leh_frame_end1:

	.globl	__Z2f1i.eh
.weak_definition __Z2f1i.eh
__Z2f1i.eh:
Lset6 = Leh_frame_end2-Leh_frame_begin2
	.long	Lset6
Leh_frame_begin2:
Lset7 = Leh_frame_begin2-Leh_frame_common
	.long	Lset7
Ltmp22:
	.quad	Leh_func_begin2-Ltmp22
Lset8 = Leh_func_end2-Leh_func_begin2
	.quad	Lset8
	.byte	0
	.byte	4
Lset9 = Ltmp2-Leh_func_begin2
	.long	Lset9
	.byte	14
	.byte	16
	.byte	134
	.byte	2
	.byte	4
Lset10 = Ltmp3-Ltmp2
	.long	Lset10
	.byte	13
	.byte	6
	.align	3
Leh_frame_end2:

	.globl	__ZN2F2clEi.eh
.weak_definition __ZN2F2clEi.eh
__ZN2F2clEi.eh:
Lset11 = Leh_frame_end3-Leh_frame_begin3
	.long	Lset11
Leh_frame_begin3:
Lset12 = Leh_frame_begin3-Leh_frame_common
	.long	Lset12
Ltmp23:
	.quad	Leh_func_begin3-Ltmp23
Lset13 = Leh_func_end3-Leh_func_begin3
	.quad	Lset13
	.byte	0
	.byte	4
Lset14 = Ltmp4-Leh_func_begin3
	.long	Lset14
	.byte	14
	.byte	16
	.byte	134
	.byte	2
	.byte	4
Lset15 = Ltmp5-Ltmp4
	.long	Lset15
	.byte	13
	.byte	6
	.align	3
Leh_frame_end3:

	.globl	__ZN7F1_wrapclEi.eh
.weak_definition __ZN7F1_wrapclEi.eh
__ZN7F1_wrapclEi.eh:
Lset16 = Leh_frame_end4-Leh_frame_begin4
	.long	Lset16
Leh_frame_begin4:
Lset17 = Leh_frame_begin4-Leh_frame_common
	.long	Lset17
Ltmp24:
	.quad	Leh_func_begin4-Ltmp24
Lset18 = Leh_func_end4-Leh_func_begin4
	.quad	Lset18
	.byte	0
	.byte	4
Lset19 = Ltmp6-Leh_func_begin4
	.long	Lset19
	.byte	14
	.byte	16
	.byte	134
	.byte	2
	.byte	4
Lset20 = Ltmp7-Ltmp6
	.long	Lset20
	.byte	13
	.byte	6
	.align	3
Leh_frame_end4:

	.globl	__Z7do_sth1IXadL_Z2f1iEEEvv.eh
.weak_definition __Z7do_sth1IXadL_Z2f1iEEEvv.eh
__Z7do_sth1IXadL_Z2f1iEEEvv.eh:
Lset21 = Leh_frame_end5-Leh_frame_begin5
	.long	Lset21
Leh_frame_begin5:
Lset22 = Leh_frame_begin5-Leh_frame_common
	.long	Lset22
Ltmp25:
	.quad	Leh_func_begin5-Ltmp25
Lset23 = Leh_func_end5-Leh_func_begin5
	.quad	Lset23
	.byte	0
	.byte	4
Lset24 = Ltmp9-Leh_func_begin5
	.long	Lset24
	.byte	14
	.byte	16
	.byte	134
	.byte	2
	.byte	4
Lset25 = Ltmp10-Ltmp9
	.long	Lset25
	.byte	13
	.byte	6
	.align	3
Leh_frame_end5:

	.globl	__Z7do_sth2I2F2EvT_.eh
.weak_definition __Z7do_sth2I2F2EvT_.eh
__Z7do_sth2I2F2EvT_.eh:
Lset26 = Leh_frame_end6-Leh_frame_begin6
	.long	Lset26
Leh_frame_begin6:
Lset27 = Leh_frame_begin6-Leh_frame_common
	.long	Lset27
Ltmp26:
	.quad	Leh_func_begin6-Ltmp26
Lset28 = Leh_func_end6-Leh_func_begin6
	.quad	Lset28
	.byte	0
	.byte	4
Lset29 = Ltmp11-Leh_func_begin6
	.long	Lset29
	.byte	14
	.byte	16
	.byte	134
	.byte	2
	.byte	4
Lset30 = Ltmp12-Ltmp11
	.long	Lset30
	.byte	13
	.byte	6
	.align	3
Leh_frame_end6:

	.globl	__Z5test2v.eh
__Z5test2v.eh:
Lset31 = Leh_frame_end7-Leh_frame_begin7
	.long	Lset31
Leh_frame_begin7:
Lset32 = Leh_frame_begin7-Leh_frame_common
	.long	Lset32
Ltmp27:
	.quad	Leh_func_begin7-Ltmp27
Lset33 = Leh_func_end7-Leh_func_begin7
	.quad	Lset33
	.byte	0
	.byte	4
Lset34 = Ltmp13-Leh_func_begin7
	.long	Lset34
	.byte	14
	.byte	16
	.byte	134
	.byte	2
	.byte	4
Lset35 = Ltmp14-Ltmp13
	.long	Lset35
	.byte	13
	.byte	6
	.align	3
Leh_frame_end7:

	.globl	__Z7do_sth2I7F1_wrapEvT_.eh
.weak_definition __Z7do_sth2I7F1_wrapEvT_.eh
__Z7do_sth2I7F1_wrapEvT_.eh:
Lset36 = Leh_frame_end8-Leh_frame_begin8
	.long	Lset36
Leh_frame_begin8:
Lset37 = Leh_frame_begin8-Leh_frame_common
	.long	Lset37
Ltmp28:
	.quad	Leh_func_begin8-Ltmp28
Lset38 = Leh_func_end8-Leh_func_begin8
	.quad	Lset38
	.byte	0
	.byte	4
Lset39 = Ltmp16-Leh_func_begin8
	.long	Lset39
	.byte	14
	.byte	16
	.byte	134
	.byte	2
	.byte	4
Lset40 = Ltmp17-Ltmp16
	.long	Lset40
	.byte	13
	.byte	6
	.align	3
Leh_frame_end8:

	.globl	__Z5test3v.eh
__Z5test3v.eh:
Lset41 = Leh_frame_end9-Leh_frame_begin9
	.long	Lset41
Leh_frame_begin9:
Lset42 = Leh_frame_begin9-Leh_frame_common
	.long	Lset42
Ltmp29:
	.quad	Leh_func_begin9-Ltmp29
Lset43 = Leh_func_end9-Leh_func_begin9
	.quad	Lset43
	.byte	0
	.byte	4
Lset44 = Ltmp18-Leh_func_begin9
	.long	Lset44
	.byte	14
	.byte	16
	.byte	134
	.byte	2
	.byte	4
Lset45 = Ltmp19-Ltmp18
	.long	Lset45
	.byte	13
	.byte	6
	.align	3
Leh_frame_end9:


.subsections_via_symbols
