// demo_NSAutoreleaseNoPool.mm
//
// This is a demonstration to make NSAutoreleaseNoPool more useful. It replaces
// that function at runtime (via mach_override) and adds a `print_backtrace()` call to it.
//
// The code can easily be adopted for similar debug hook functions.
//
// Sample output:
//
//     $ ./a.out
//     ___NSAutoreleaseNoPool addr: 0x7fff8724bd30
//     2014-02-18 15:20:09.362 a.out[25277:903] *** __NSAutoreleaseNoPool(): Object 0x7fff71154190 of class NSCFString autoreleased with no pool in place - just leaking
//     __NSAutoreleaseNoPool backtrace:
//     backtrace() returned 8 addresses
//     0   a.out                               0x0000000100007928 _Z15print_backtracev + 40
//     1   a.out                               0x0000000100007a10 _Z33__NSAutoreleaseNoPool_replacementPv + 48
//     2   CoreFoundation                      0x00007fff87196e79 _CFAutoreleasePoolAddObject + 649
//     3   CoreFoundation                      0x00007fff87196be6 -[NSObject(NSObject) autorelease] + 22
//     4   a.out                               0x00000001000078dd _Z11raiseNoPoolv + 109
//     5   a.out                               0x0000000100007ac2 main + 18
//     6   a.out                               0x0000000100001524 start + 52
//     7   ???                                 0x0000000000000001 0x0 + 1
//
// compile:
// cc -framework CoreFoundation -framework Foundation -Imach_override mach_override/*.c mach_override/libudis86/*.c demo_NSAutoreleaseNoPool.mm

#include <dlfcn.h>
#include <stdio.h>
#import <Foundation/Foundation.h>
#include <mach-o/dyld.h>
#import <mach-o/nlist.h>
#import <string.h>
#include <assert.h>
#include <mach_override.h>
#include <execinfo.h>

// Adapted from:
// https://github.com/0xced/iOS-Artwork-Extractor/blob/master/Classes/FindSymbol.c
// Adapted from MoreAddrToSym / GetFunctionName()
// http://www.opensource.apple.com/source/openmpi/openmpi-8/openmpi/opal/mca/backtrace/darwin/MoreBacktrace/MoreDebugging/MoreAddrToSym.c
void *FindSymbol(const struct mach_header *img, const char *symbol)
{
	if ((img == NULL) || (symbol == NULL))
		return NULL;

	// only 64bit supported
#if defined (__LP64__)

	if(img->magic != MH_MAGIC_64)
		// we currently only support Intel 64bit
		return NULL;

	struct mach_header_64 *image = (struct mach_header_64*) img;
		
	struct segment_command_64 *seg_linkedit = NULL;
	struct segment_command_64 *seg_text = NULL;
	struct symtab_command *symtab = NULL;
	unsigned int index;
	
	struct load_command *cmd = (struct load_command*)(image + 1);
	
	for (index = 0; index < image->ncmds; index += 1, cmd = (struct load_command*)((char*)cmd + cmd->cmdsize))
	{
		switch(cmd->cmd)
		{
			case LC_SEGMENT_64: {
				struct segment_command_64* segcmd = (struct segment_command_64*)cmd;
				if (!strcmp(segcmd->segname, SEG_TEXT))
					seg_text = segcmd;
				else if (!strcmp(segcmd->segname, SEG_LINKEDIT))
					seg_linkedit = segcmd;
				break;
			}
			
			case LC_SYMTAB:
				symtab = (struct symtab_command*)cmd;
				break;
				
			default:
				break;
		}
	}
	
	if ((seg_text == NULL) || (seg_linkedit == NULL) || (symtab == NULL))
		return NULL;
	
	unsigned long vm_slide = (unsigned long)image - (unsigned long)seg_text->vmaddr;
	unsigned long file_slide = ((unsigned long)seg_linkedit->vmaddr - (unsigned long)seg_text->vmaddr) - seg_linkedit->fileoff;
	struct nlist_64 *symbase = (struct nlist_64*)((unsigned long)image + (symtab->symoff + file_slide));
	char *strings = (char*)((unsigned long)image + (symtab->stroff + file_slide));
	struct nlist_64 *sym;
	
	for (index = 0, sym = symbase; index < symtab->nsyms; index += 1, sym += 1)
	{
		if (sym->n_un.n_strx != 0 && !strcmp(symbol, strings + sym->n_un.n_strx))
		{
			unsigned long address = vm_slide + sym->n_value;
			if (sym->n_desc & N_ARM_THUMB_DEF)
				return (void*)(address | 1);
			else
				return (void*)(address);
		}
	}	
#endif
	
	return NULL;
}

typedef void (*NSAutoreleaseNoPoolFunc) (void* obj);

NSAutoreleaseNoPoolFunc __NSAutoreleaseNoPool_reenter;

void print_backtrace() {
    void *callstack[128];
    int framesC = backtrace(callstack, sizeof(callstack));
    printf("backtrace() returned %d addresses\n", framesC);
    char** strs = backtrace_symbols(callstack, framesC);
    for(int i = 0; i < framesC; ++i) {
        if(strs[i])
            printf("%s\n", strs[i]);
        else
            break;
    }
    free(strs);
}

void __NSAutoreleaseNoPool_replacement(void* obj) {
	__NSAutoreleaseNoPool_reenter(obj);
	printf("__NSAutoreleaseNoPool backtrace:\n");
	print_backtrace();
}

void replaceNSAutoreleaseNoPool() {	
	const struct mach_header* img = NSAddImage("/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation", NSADDIMAGE_OPTION_NONE);
	NSAutoreleaseNoPoolFunc f = (NSAutoreleaseNoPoolFunc) FindSymbol((struct mach_header*)img, "___NSAutoreleaseNoPool");
	
	printf("___NSAutoreleaseNoPool addr: %p\n", f);
	
	// Newer MacOSX versions wont have the symbol at all. (I guess it went away with ARC.)
	if(f) {
		mach_override_ptr(
			(void*)f,
			(void*)__NSAutoreleaseNoPool_replacement,
			(void**)&__NSAutoreleaseNoPool_reenter);
	}
}

id raiseNoPool() {
	NSString* foo = [[NSString alloc] init];
	// This will lead to a NSAutoreleaseNoPool() call.
	return [foo autorelease];
}

int main() {
	replaceNSAutoreleaseNoPool();
	raiseNoPool();
	return 0;
}

