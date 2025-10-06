#ifndef OPSUTILS_H
#define OPSUTILS_H

#include "C:\Programming\Revamp2\Tensor\Tensor.h"
#include "C:\Programming\Revamp2\Kernals\kernels.h"
#include "C:\Programming\Revamp2\Iterator\Iterator.h"
#include "C:\Programming\Revamp2\Iterator\cpu_iter.h"
#include "C:\Programming\Revamp2\Operations\ops.h"
#include "C:\Programming\Revamp2\Error.h"

/* Binary Operations */

Tensor *binary_step_1_make_compatiable_type_and_shape(Tensor *a, Tensor *b);
Iterator *binary_step_2_make_iterator(Tensor **inputs, int numel, Tensor *out, dtype result_type);
bool binary_step_3_check_kernel_exists(Tensor_Iterator_Kernel kernel);

/* Unary Operations */

Tensor *unary_step_1_make_output_tensor(Tensor *a, dtype result_type);
Iterator *unary_step_2_make_iterator(Tensor **inputs, int numel, Tensor *out, dtype result_type);


/* Reduction Operations */

Tensor *redeuction_step_1_make_output_tensor(Tensor *a, int64_t *dims, int64_t ndims, bool keepdim);
Iterator *redeuction_step_2_make_iterator(Tensor **inputs, int numel, Tensor *out, int64_t *dims, int64_t ndims, bool keepdim, bool set_special_value, int64_t special_value);
bool reduction_step_3_check_kernel_exists(Tensor_Reduction_Kernel kernel);

#endif // OPSUTILS_H
