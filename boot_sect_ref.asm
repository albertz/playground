; https://www.cs.bham.ac.uk/~exr/lectures/opsys/10_11/lectures/os-dev.pdf
;
; compile: nasm boot_sect_hello.asm -f bin -o boot_sect_hello.bin
; run: qemu-system-i386 boot_sect_hello.bin 

mov ah, 0x0e ; tty mode

[org 0x7c00]
mov al, [data]
int 0x10

jmp $  ; loop

data:
db "X"

; padding to 510th byte
times 510-($-$$) db 0

; boot sector magic number
dw 0xaa55
