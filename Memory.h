#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

typedef char ALIGN[16];

typedef union Header Header;

typedef union Header {
   struct {
        uintptr_t id;
        bool is_free;
        size_t size;
        Header *next;
        Header *prev;
    } md;
    ALIGN align;
}Header;

// The compiler performs natural alignment with the elements of the struct. The struct size is then rounded up to the multiple of the greatest alignment requirements of its elements
// The union will take the size of the element with the biggest size requirement but will take the alignment of the element with the biggest alignment requirement

#define MIN_PAYLOAD 16
#define HEADER_SIZE (sizeof(Header))
#define BLOCK_SIZE(block) ((block)->md.size)
#define SAFE_SIZE(size) (HEADER_SIZE + size + MIN_PAYLOAD)

Header *get_free_memory(size_t size);

void *alloc_memory(size_t size);

// void *copy_to_block();

void free_memory(void *memory);

void print_memory_layout();

bool validate_block(Header *block);