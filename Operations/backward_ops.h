#include "C:\Programming\Revamp2\Operations\ops.h"

#define MAX_NDIM 8

Iterator *return_backward_Iterator(Node *node);
void backward_add(Node *node);
void backward_mul(Node *node);

void backward_add_kernel(char **ptrs, const int64_t *inner_stride, int64_t loop_len);
Tensor *sum_to_shape(Tensor *src, Tensor *target);