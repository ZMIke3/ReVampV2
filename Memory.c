#include "C:\Programming\Revamp2\Memory.h"
#include <stdlib.h>


Header *global_memory;

Header *split_block(Header *block, size_t size) {

    /*

        [Header|-- 32 bytes --][Header|-- 60 bytes free --]
         ^                      ^
         block                  new_block
        
         Block is what will get used
        
    */

    if (!block) {
        printf("Block is NULL, cannot split!\n");
        return NULL;
    }

    if (block->md.is_free && block->md.size >= HEADER_SIZE + size + MIN_PAYLOAD) {

        Header *new_block = (Header *) ((char *) (block + 1) + size); // Cast to char so we move by bytesize

        new_block->md.id = (uintptr_t)new_block;
        new_block->md.is_free = true;
        new_block->md.size = block->md.size - size - HEADER_SIZE;
        new_block->md.next = NULL;
        new_block->md.prev = NULL;

        Header *next = block->md.next;

        new_block->md.next = block->md.next;
        new_block->md.prev = block;
        
        if (block->md.next) {
            block->md.next->md.prev = new_block;
        }
        block->md.next = new_block;

        block->md.size = size;
        
        return new_block; 

    }


    return NULL;


}

Header *get_free_memory(size_t size) {
	Header *curr = global_memory;

	while(curr) {
		if (curr->md.is_free && curr->md.size >= size) {
            
            if (curr->md.size >= SAFE_SIZE(size)) {
                split_block(curr, size);
            }

            curr->md.is_free = false;
            return curr;
        }
		curr = curr->md.next;
	}

	return NULL;
}

void *alloc_memory(size_t size) {

    size_t total_size = HEADER_SIZE + size;

    Header *block = get_free_memory(size);


    if(block) {
        return (void *) (block + 1);
    }

    block = malloc(total_size);
    block->md.id = (uintptr_t)block;
    block->md.is_free = false;
    block->md.size = size;
    block->md.next = NULL;
    block->md.prev = NULL;

    if (global_memory == NULL) {

        global_memory = block;
        block->md.next = NULL;
        block->md.prev = NULL;
    } else {

        Header *curr = global_memory;

        while (curr->md.next) {
            curr = curr->md.next;
        }

        curr->md.next = block;
        block->md.prev = curr;
        
    }

    return (void *)(block + 1);

}

void free_memory(void *memory) {
    if (!memory) {
        printf("Input is NULL\n");
    }

    Header *block = ((Header *)memory) - 1;

    if (!validate_block(block)) {
        printf("Invalid free: pointer not from allocator!\n");
        return;
    }

    block->md.is_free = true;

    Header *prev = block->md.prev;
    Header *next = block->md.next;

    if (prev && prev->md.is_free) {
        prev->md.size += HEADER_SIZE + block->md.size;
        prev->md.next = block->md.next;
        if (block->md.next) {
            block->md.next->md.prev = prev;
        }
        block = prev; 
    }

    if (next && next->md.is_free) {
        block->md.size += HEADER_SIZE + next->md.size;
        block->md.next = next->md.next;
        if (next->md.next) {
            next->md.next->md.prev = block;
        }
    }
}

bool validate_block(Header *block) {
    if (block->md.id != (uintptr_t)block) {
        printf("ERROR: Block ID doesn't match address!\n");
        printf("Expected: %p, Got: %p\n", (void*)block, (void*)block->md.id);
        return false;
    }
    return true;
}

void print_memory_layout() {
    printf("Memory Layout:\n");
    Header *curr = global_memory;
    int block_num = 0;
    
    while (curr) {
        printf("Block %d: Size=%zu, Free=%s, ID=0x%p\n", 
               block_num++, curr->md.size, 
               curr->md.is_free ? "Yes" : "No", 
              curr->md.id);
        curr = curr->md.next;
    }
    printf("\n");
}