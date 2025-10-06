#include "C:\Programming\Revamp2\Utils\utills.h"


int64_t max_64_h(int64_t a, int64_t b) {
    return a > b ? a : b;
}

int64_t tensor_arange_helper(int64_t start, int64_t end, int64_t step) {

    int64_t size = (end - start) / step;

    if ((end - start) % step != 0) size += 1;

    return size;
    
}

int64_t tensor_size_from_shape(int64_t *shape, int64_t ndim) {
    int64_t size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

int64_t *calc_stride(int64_t ndim, int64_t *shape, size_t dtype_size) {
    int64_t *stride = malloc(ndim * sizeof(int64_t));
    assert(stride != NULL);
    stride[ndim - 1] = dtype_size;

    for (int64_t i = ndim - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }

    return stride;
}



// void copy_array_to_tensor_buffer(void *tensor, void *array) {

//     Header *t_h = (Header *)tensor;

//     void *buffer = (void *) (t_h + 1);

//     memcpy


// }