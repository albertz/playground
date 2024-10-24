; https://www.cs.bham.ac.uk/~exr/lectures/opsys/10_11/lectures/os-dev.pdf
;
; A simple boot sector program that loops forever.
;
; compile: nasm boot_sect.asm -f bin -o boot_sect.bin
; run: qemu-system-i386 boot_sect.bin 

; Define a label , " loop ", that will allow
; us to jump back to it , forever.
loop :

; Use a simple CPU instruction that jumps
; to a new memory address to continue execution.
; In our case , jump to the address of the current
; instruction.
jmp loop

; When compiled , our program must fit into 512 bytes ,
; with the last two bytes being the magic number ,
; so here , tell our assembly compiler to pad out our
; program with enough zero bytes (db 0) to bring us to the
; 510 th byte.
times 510-($-$$) db 0

; Last two bytes ( one word ) form the magic number ,
; so BIOS knows we are a boot sector.
dw 0xaa55
