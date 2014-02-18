// compile:
// c++ -framework CoreFoundation -framework Foundation test_NSAutoreleaseNoPool.mm

#include <dlfcn.h>
#include <stdio.h>
#import <Foundation/Foundation.h>
#include <mach-o/dyld.h>
#import <mach-o/nlist.h>
#import <string.h>
#include <assert.h>

typedef void (*NSAutoreleaseNoPoolFunc) (void* obj);

// Adapget from:
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

void foo() {
	NSAutoreleaseNoPoolFunc __NSAutoreleaseNoPool = (NSAutoreleaseNoPoolFunc) dlsym(RTLD_DEFAULT, "__NSAutoreleaseNoPool");
	
	const struct mach_header* img = NSAddImage("/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation", NSADDIMAGE_OPTION_NONE);
	NSAutoreleaseNoPoolFunc f = (NSAutoreleaseNoPoolFunc) FindSymbol((struct mach_header*)img, "___NSAutoreleaseNoPool");
		
	printf("func: %p (%p) %p\n", __NSAutoreleaseNoPool, img, f);
	
	if(f) {
		NSObject* foo = [[NSObject alloc] init];
		f(foo);
	}
}

id raiseNoPool() {
	NSString* foo = [[NSString alloc] init];
	return [foo autorelease];
}

void bar() {
	id x = raiseNoPool();
}

int main() {
	foo();
	bar();
	return 0;
}

