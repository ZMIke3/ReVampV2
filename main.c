#include "C:\Programming\Revamp2\Tensor\Tensor.h"
#include "C:\Programming\Revamp2\Iterator\cpu_iter.h"
#include "C:\Programming\Revamp2\Operations\ops.h"
#include "C:\Programming\Revamp2\Iterator\Iterator.h"
#include <stdlib.h>
#include <stdio.h>



int64_t *matmul_kernel(int64_t *a1, int64_t *a2, int64_t a1_r, int64_t a1_c, int64_t a2_r, int64_t a2_c) {

    if (a1_c != a2_r) {

        printf("Invalid dimensions for matmul\n");
        return NULL;
    }

    int64_t noe = a1_r * a2_c;

    int64_t *rs = calloc(sizeof(int64_t), noe);

    for (int64_t i = 0; i < a1_r; i++) {

        for (int64_t j = 0; j < a2_c; j++) {
            
            for (int64_t k = 0; k < a1_c; k++) {

                rs[i * a2_c + j] += a1[i * a1_c + k] * a2[k * a2_c + j]; 
            }

        }

    }


    return rs;


} 


Tensor *matmul(Tensor *a, Tensor *b) {

   if(!broadcast(a, b)) {

        printf("Failed to broadcast inputs\n");
   } 

    int64_t a_r = get_tensor_shape(a)[get_tensor_ndim(a) - 2];
    int64_t a_c = get_tensor_shape(a)[get_tensor_ndim(a) - 1];

    int64_t b_r = get_tensor_shape(b)[get_tensor_ndim(b) - 2];
    int64_t b_c = get_tensor_shape(b)[get_tensor_ndim(b) - 1];

    // (B1, r, c) X (B, r2, c2)
    
    if ()

    if (get_tensor_ndim(a) == 2 && get_tensor_ndim(b) == 2) {



    }


}


int main() {

    Tensor *a = arange(0, 10, 1, DTYPE_I64, CPU);

    Tensor *b = arange(0, 10, 1, DTYPE_I64, CPU);

    a =  reshape(a, (int64_t[]){5, 2}, 2);
    b =  reshape(b, (int64_t[]){2, 5}, 2);

    print(a);

    print(b);


    Tensor *res = matmul(a, b);

    print(res);
    

   
}