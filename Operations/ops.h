#ifndef OPS_H
#define OPS_H

#include "C:\Programming\Revamp2\Tensor\Tensor.h"
#include "C:\Programming\Revamp2\Kernals\kernels.h"
#include "C:\Programming\Revamp2\Iterator\Iterator.h"
#include "C:\Programming\Revamp2\Iterator\cpu_iter.h"
#include "C:\Programming\Revamp2\Operations\ops_utils.h"
#include "C:\Programming\Revamp2\Operations\backward_ops.h"
#include "C:\Programming\Revamp2\Kernals\package.h"
#include <assert.h>



Tensor *cast_to_type(Tensor *a, dtype type, bool force_cast);
bool broadcast(Tensor *a, Tensor *b);
Tensor *broadcast_inputs_to_tensor(Tensor **inputs, dtype out_type, Backend out_backend);

Tensor *add(Tensor *a, Tensor *b);
Tensor *multiply(Tensor *a, Tensor *b);
Tensor *subtract(Tensor *a, Tensor *b);
Tensor *divide(Tensor *a, Tensor *b);
Tensor *power(Tensor *a, Tensor *b);

/* These functions are not in place, they create and return new tensors*/
Tensor *square_root(Tensor *a);
Tensor *absolute_value(Tensor *a);
// Negate is not supported for some types, need to check that
Tensor *negate(Tensor *a);

/* Comparisons */
// For greater than, less than, and equal we need to make them return type bool
Tensor *equal(Tensor *a, Tensor *b);
Tensor *maximum(Tensor *a, Tensor *b); // Element wise
Tensor *minimum(Tensor *a, Tensor *b); // Element wise
Tensor *greater_than(Tensor *a, Tensor *b);
Tensor *less_than(Tensor *a, Tensor *b);
// This function requires extra steps
Tensor *where(Tensor *condition, Tensor *a, Tensor *b);

/* Creation */
// Fill value type and tensor type must be the same, maybe this should take just numbers instead
Tensor *fill(int64_t ndim, int64_t *shape, dtype dtype, Backend backend, void *value);
// Some dtypes should not be used with arange, like bool
Tensor *arange(int64_t start, int64_t end, int64_t step, dtype dtype, Backend backend); // <- Needs checks

/* Reductions */
//int64_t calculate_output_index(cpu_iter *iter, bool *reduce_dim, Tensor *out, bool keepdim);
Tensor *sum(Tensor *a, int64_t *dims, int64_t ndims, bool keepdim);  
Tensor *max(Tensor *a, int64_t *dims, int64_t ndims, bool keepdim);
Tensor *min(Tensor *a, int64_t *dims, int64_t ndims, bool keepdim);


/* Shape Manipulation */
Tensor *reshape(Tensor *a, int64_t *new_shape, int64_t ndim);
Tensor *squeeze(Tensor *a, int64_t dim);        
Tensor *unsqueeze(Tensor *a, int64_t dim);      

/* Slicing */
Tensor *tensor_at(Tensor *a, int64_t *dims, int64_t ndim); // Returns an actual seperate tensor object from the slice
Tensor *slice(Tensor *a, int64_t *dims, int64_t ndim); // Returns a view



/* Utility */
void print_flat_i64(const Tensor *t);

#endif // OPS_H
