// #include "C:\Programming\Revamp2\Memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>  // add this at the top

// #include <stdint.h>

// // void copy_to_block(void *memory, int *array, int size) {

// //     Header *mem = (Header *) memory;

// //     if (mem->md.size < size)
// //     {
// //         return;
// //     }



// //     void *block = (void *)mem + 1;

// //     memcpy(block, array, size);
// // }


// // void print_(void *memory) {
    
// //     Header *mem = (Header *) memory;

// //     void *block = (void *)mem + 1;

// //     int *b = (int *)block;

// //     for (int i = 0; i < 5; i++){
// //         printf("%d", b[i]);
// //     }

// // }


// #include <string.h> // for memcpy
// #include <stdio.h>

// void copy_to_block(void *payload_ptr, void *array, size_t size) {
//     if (!payload_ptr || !array || size <= 0) return;

//     Header *hdr = ((Header*)payload_ptr) - 1;         // header is just before payload
//     size_t payload_bytes = hdr->md.size;              // stored in bytes

//     size_t copy_bytes = size;
//     if (payload_bytes < copy_bytes) {
//         fprintf(stderr, "copy_to_block: payload too small (%zu bytes) for %zu bytes\n", payload_bytes, copy_bytes);
//         return;
//     }

//     void *payload = (void*)(hdr + 1);                 // start of payload
//     memcpy(payload, array, copy_bytes);
// }

// void print_(void *payload_ptr, int count) {
//     if (!payload_ptr || count <= 0) return;

//     Header *hdr = ((Header*)payload_ptr) - 1;
//     size_t payload_bytes = hdr->md.size;
//     size_t need = (size_t)count * sizeof(int);

//     if (payload_bytes < need) {
//         fprintf(stderr, "print_: asking for %zu bytes but block has %zu bytes\n", need, payload_bytes);
//         count = (int)(payload_bytes / sizeof(int)); // print only what fits
//     }

//     int *b = (int*)(hdr + 1);
//     for (int i = 0; i < count; ++i) {
//         printf("%d", b[i]);
//         if (i + 1 < count) printf(" ");
//     }
//     printf("\n");
// }


// int main() {

//    // void *ptr1 = alloc_memory(32);
//     // void *ptr2 = alloc_memory(64);
//     // void *ptr3 = alloc_memory(16);
    
//     // printf("After allocations:\n");
//        // void *ptr1 = alloc_memory(32);
//     // print_memory_layout();
    
//     // free_memory(ptr2);
//     // printf("After freeing ptr2:\n");
//     // print_memory_layout();
    
//     // void *ptr4 = alloc_memory(32); // Should reuse freed space
//     // printf("After allocating ptr4:\n");
//     // print_memory_layout();


//     // void *ptr5 = alloc_memory(32);
//     // printf("After allocating ptr5:\n");
//     // print_memory_layout();

    
//     // free_memory(ptr1);
//     // free_memory(ptr3);
//     // free_memory(ptr4);
//     // free_memory(ptr5);

//     // printf("After Freeing all:\n");
//     // print_memory_layout();


//     // void *ptr6 = alloc_memory(32);
//     // printf("After allocating ptr6:\n");
//     // print_memory_layout();

//     // printf("After allocating ptr7 with malloc:\n");
//     // void *ptr7 = malloc(32);

//     // printf("Try to free ptr7\n");
//     // free_memory(ptr7);

//     int *arr = malloc(sizeof(int) * 5);

//     for (int i = 0; i < 5; i++) {
//         arr[i] = i;
//     }

//     for (int i = 0; i < 5; i++) {
//         printf("%d ", arr[i]);
//     }

//     printf("\n");

//     void *ptr1 = alloc_memory(5*8);
//     print_memory_layout();

//     printf("\n");



//     copy_to_block(ptr1, arr, 5);

//     print_(ptr1, 5);


    
//     return 0;
// }

int *matmaul(int *a1, int *a2, int a1_r, int a1_c, int a2_r, int a2_c) {

    if (a1_c != a2_r) {

        printf("Invalid dimensions for matmul\n");
        return NULL;
    }

    int noe = a1_r * a2_c;

    int *rs = calloc(sizeof(int), noe);

    for (int i = 0; i < a1_r; i++) {

        for (int j = 0; j < a2_c; j++) {
            
            for (int k = 0; k < a1_c; k++) {

                rs[i * a2_c + j] += a1[i * a1_c + k] * a2[k * a2_c + j]; 
            }

        }

    }


    return rs;


} 


void print_mat(int *rs, int r, int c) {

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%d ", rs[i * c + j]);
        }
        printf("\n");
    }
}


void matmul_code() {

    int a1_r = 512;
    int a1_c = 512;

    int a2_r = 512;
    int a2_c = 512;

    int *a1 = malloc(a1_r * a1_c * sizeof(int));
    int *a2 = malloc(a2_r * a2_c * sizeof(int));

    // Fill matrices with some values (e.g., i % 100)
    for (int i = 0; i < a1_r * a1_c; i++) {
        a1[i] = i % 100;
    }
    for (int i = 0; i < a2_r * a2_c; i++) {
        a2[i] = (i * 2) % 100;
    }

    // Time measurement
    clock_t start = clock();

    int *result = matmaul(a1, a2, a1_r, a1_c, a2_r, a2_c);

    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Matrix multiplication completed in %.3f seconds\n", time_spent);

    // Optional: verify a few results (commented out to avoid massive output)
    // for (int i = 0; i < 10; i++) {
    //     printf("%d ", result[i]);
    // }

    free(a1);
    free(a2);
    free(result);

}


int *calc_stride(int ndim, int *shape, size_t dtype_size) {
    int *stride = malloc(ndim * sizeof(int));
 //   assert(stride != NULL);
    stride[ndim - 1] = dtype_size;

    for (int i = ndim - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }

    return stride;
}


void print_from_offset(int *ar, int offset, int r, int c, int r_s, int c_s) {

    int *kr = ar + offset;

    for (int i = 0; i < r; i++) {

        for (int j = 0; j < c; j++) {
            printf("%d ", kr[i * c + j]);
        }

        printf("\n");
    }
}

int extract_slice(int ndim, int *shape, int *stride, int *ar, int batch_index, int r, int c) {

        int bi = batch_index;
        int offset = 0;

        for (int i = ndim - 3; i >= 0; i--) {
            int dim_size = shape[i];

            // printf("Shape: %d at i: %d\n", shape[i], i);

            // printf("Batch Index:%d\n", batch_index);

            int idx_i = bi % dim_size;   // index along this dimension

            // printf("idx_i: %d\n", idx_i);

            // printf("bi / ds: %d\n", batch_index / dim_size);

            bi /= dim_size;

        //    printf("Stride: %d at i: %d\n", stride[i] / sizeof(int), i);

            offset += idx_i * stride[i] / sizeof(int);

            // printf("Offset: %d at Iteration: %d\n", offset, i);

            // printf("\n");

        }

      //  print_from_offset(ar, offset, r, c, stride[ndim - 2], stride[ndim - 1]);

        return offset;



}

void matmul(int *rs, int *a1, int *a2, int a1_r, int a1_c, int a2_r, int a2_c) {

    if (a1_c != a2_r) {
        printf("Invalid dimensions for matmul\n");
    }

    for (int i = 0; i < a1_r; i++) {

        for (int j = 0; j < a2_c; j++) {
            
            for (int k = 0; k < a1_c; k++) {

                rs[i * a2_c + j] += a1[i * a1_c + k] * a2[k * a2_c + j]; 
            }

        }

    }

} 

int *batched_matmul(int ndim, int *shape, int *stride, int size, int *ar1, int *ar2, int B, int r, int c) {

    int offset1 = 0;
    int offset2 = 0;

    int *rs = calloc(sizeof(int), size);

   for (int batch_index = 0; batch_index < B; batch_index++) {

        offset1 = extract_slice(ndim, shape, stride, ar1, batch_index, r, c);

        offset2 = extract_slice(ndim, shape, stride, ar2, batch_index, r, c);

        int *kr1 = ar1 + offset1;
        
        int *kr2 = ar2 + offset2;
 
        matmul(rs + offset1, kr1, kr2, r, c, r, c);

   }
    
   return rs;


}

void batched_matmul_test() {

    int B = 200;

    int b2 = 4;

    int b3 = 5;

    int r = 100;

    int c = 100;


    int ndim = 5;

    int shape[5] = {B, b2, b3, r, c};

    int *stride = calc_stride(ndim, shape, sizeof(int));
    

    int *ar = malloc(B * b2 * b3 * r * c * sizeof(int));

    int size = B * b2 * b3 * r * c;

    for (int i = 0; i < size; i++) {
        ar[i] = i;
    }

    printf("\n");

    clock_t start = clock();

    int *rs = batched_matmul(ndim, shape, stride, size, ar, ar, B * b2 * b3, r, c);
   
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Elapsed time: %.6f seconds\n", elapsed);


    // for (int i = 0; i < size; i++) {

    //     printf("%d ", rs[i]);

    // }

    printf("\n");


}

int *matmaul_d(int *a1, int *a2, int a1_r, int a1_c, int a2_r, int a2_c) {

    if (a1_c != a2_r) {

        printf("Invalid dimensions for matmul\n");
        return NULL;
    }

    int noe = a1_r * a2_c;

    int *rs = calloc(sizeof(int), noe);

    for (int i = 0; i < a1_r; i++) {

        for (int j = 0; j < a2_c; j++) {
            
            for (int k = 0; k < a1_c; k++) {

                rs[i * a2_c + j] += a1[i * a1_c + k] * a2[k * a2_c + j]; 
            }

        }

    }


    return rs;


} 

int main() {

    int a1_r = 4096;
    int a1_c = 4096;

    int a2_r = 4096;
    int a2_c = 4096;

    int *a1 = malloc(a1_r * a1_c * sizeof(int));
    int *a2 = malloc(a2_r * a2_c * sizeof(int));

    // Fill matrices with some values (e.g., i % 100)
    for (int i = 0; i < a1_r * a1_c; i++) {
        a1[i] = i % 100;
    }
    for (int i = 0; i < a2_r * a2_c; i++) {
        a2[i] = (i * 2) % 100;
    }

    // Time measurement
    clock_t start = clock();

    int *result = matmaul_d(a1, a2, a1_r, a1_c, a2_r, a2_c);

    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Matrix multiplication completed in %.3f seconds\n", time_spent);

    // Optional: verify a few results (commented out to avoid massive output)
    // for (int i = 0; i < 10; i++) {
    //     printf("%d ", result[i]);
    // }

    free(a1);
    free(a2);
    free(result);

}