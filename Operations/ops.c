#include "C:\Programming\Revamp2\Operations\ops.h"
#include "C:\Programming\Revamp2\Iterator\Iterator.h"
#include "C:\Programming\Revamp2\Iterator\cpu_iter.h"
#include "C:\Programming\Revamp2\Operations\ops_utils.h"


bool broadcast(Tensor *a, Tensor *b) {
  
    Tensor **config = malloc(sizeof(Tensor *) * 2);
    config[0] = a;
    config[1] = b;


    int64_t ndim = 0;
    for (int i = 0; i < 2; i++) { ndim =  max_64_h(ndim, config[i]->ndim); } // Get the biggest dimension

    int64_t *shape = (int64_t *) malloc(ndim * sizeof(int64_t));
    bool *broadcast_dim1 = (bool * ) malloc(ndim * sizeof(bool));
    bool *broadcast_dim2 = (bool * ) malloc(ndim * sizeof(bool));

    for (int i = 0; i < ndim; i++) { shape[i] = 1; broadcast_dim1[i] = false; broadcast_dim2[i] = false; }

    for (int ops = 0; ops < 2; ops++) {
        const Tensor *op = config[ops];

        for (int dim = 0; dim < ndim; dim++) {

            int idx = dim - (ndim - op->ndim); // x = (ndim - op->ndim) skips leading dimension that are not aligned with op, 
                                                // dim - x checks if op has the corresponding dimension in the broadcast shape (src > 0)
            int64_t dim_size = (idx >= 0) ? ((int64_t*)op->shape)[idx] : 1; // If it has the dimension we use it, else we use 1

            if (shape[dim] != 1 && dim_size != 1 && shape[dim] != dim_size) {
                return false; // Shapes are not compatiable
            }

            if (shape[dim] == 1) shape[dim] = dim_size; // Only update dimensions waiting to be updated

        }

    }

    for (int i = 0; i < ndim; i++) {
        int idx_a = i - (ndim - a->ndim);
        int64_t dim_size_a = (idx_a >= 0) ? a->shape[idx_a] : 1;
        broadcast_dim1[i] = (dim_size_a == 1 && shape[i] > 1);
    
        int idx_b = i - (ndim - b->ndim);
        int64_t dim_size_b = (idx_b >= 0) ? b->shape[idx_b] : 1;
        broadcast_dim2[i] = (dim_size_b == 1 && shape[i] > 1);
    }

    set_tensor_shape_and_ndim_broadcast(a, shape, ndim, broadcast_dim1);

    set_tensor_shape_and_ndim_broadcast(b, shape, ndim, broadcast_dim2);

    free(config);

    return true;



}

Tensor *broadcast_inputs_to_tensor(Tensor **inputs, dtype out_type, Backend out_backend) {

    int64_t ndim = 0;
    for (int i = 0; i < 2; i++) { ndim =  max_64_h(ndim, inputs[i]->ndim); } // Get the biggest dimension

    int64_t *shape = (int64_t *) malloc(ndim * sizeof(int64_t));
    bool *broadcast_dim1 = (bool * ) malloc(ndim * sizeof(bool));
    bool *broadcast_dim2 = (bool * ) malloc(ndim * sizeof(bool));

    for (int i = 0; i < ndim; i++) { shape[i] = 1; broadcast_dim1[i] = false; broadcast_dim2[i] = false; }

    for (int ops = 0; ops < 2; ops++) {
        const Tensor *op = inputs[ops];

        for (int dim = 0; dim < ndim; dim++) {

            int idx = dim - (ndim - op->ndim); // x = (ndim - op->ndim) skips leading dimension that are not aligned with op, 
                                                // dim - x checks if op has the corresponding dimension in the broadcast shape (src > 0)
            int64_t dim_size = (idx >= 0) ? ((int64_t*)op->shape)[idx] : 1; // If it has the dimension we use it, else we use 1

            if (shape[dim] != 1 && dim_size != 1 && shape[dim] != dim_size) {
                return NULL; // Shapes are not compatiable
            }

            if (shape[dim] == 1) shape[dim] = dim_size; // Only update dimensions waiting to be updated

        }

    }

    Tensor *out = poisoned_blank_tensor(ndim, shape, out_type, out_backend);


    return out;



}


/* Casting */

Tensor *cast_to_type(Tensor *a, dtype type, bool force_cast) {

    dtype new_type = force_cast == true ? type : promote_to_type(get_tensor_dtype(a), type);

    // dtype new_type = promote_to_type(get_tensor_dtype(a), type);

    if (new_type == get_tensor_dtype(a)) {
        return a;
    }

    Tensor *new_t = tensor_like_with_type(a, new_type);

    Iterator *iter = tensor_iterator_make();

    Tensor_Config *cfg = tensor_config_make();

    tensor_config_add_operand(cfg, a->data, get_tensor_ndim(a), get_tensor_stride(a), get_tensor_shape(a), get_dtype_size(get_tensor_dtype(a)), TENSOR_INPUT);

    tensor_config_add_operand(cfg, new_t->data, get_tensor_ndim(new_t), get_tensor_stride(new_t), get_tensor_shape(new_t), get_dtype_size(get_tensor_dtype(new_t)), TENSOR_OUTPUT);

    
    if (!tensor_iterator_build(cfg, iter)) { printf("Failed to build iterator\n"); free(iter); free(new_t); return NULL; }

    Tensor_Iterator_Kernel kernel = get_cast_kernel(CPU, get_tensor_dtype(a), new_type);

    tensor_iterator_serial_for_each(iter, kernel);

    tensor_iterator_free(iter);

    assert(new_t != NULL);

    return new_t;

    
}

/* #region Binary Operations */

Tensor *add(Tensor *a, Tensor *b) {

    Tensor *out = binary_step_1_make_compatiable_type_and_shape(a, b);

    dtype result_type =  get_tensor_dtype(out);

    Tensor **inputs = malloc(sizeof(Tensor *) * 2);
    inputs[0] = a;
    inputs[1] = b;
    
    Iterator *iter = binary_step_2_make_iterator(inputs, 2, out, result_type);

    Tensor_Iterator_Kernel kernel = get_kernel(get_tensor_backend(out), result_type, P_ADD);

    if (!binary_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    // Encapsulate
    // out->node = node_create(out, backward_add, "ADD", "Backward_Add");
    // node_tensor_to_inputs(a, out->node);
    // node_tensor_to_inputs(b, out->node);
    // out->is_leaf = false;



    tensor_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);


    free(inputs);

    return out;

}

Tensor *subtract(Tensor *a, Tensor *b) {

    Tensor *out = binary_step_1_make_compatiable_type_and_shape(a, b);

    dtype result_type =  get_tensor_dtype(out);

    Tensor **inputs = malloc(sizeof(Tensor *) * 2);
    inputs[0] = a;
    inputs[1] = b;
    
    Iterator *iter = binary_step_2_make_iterator(inputs, 2, out, result_type);

    Tensor_Iterator_Kernel kernel = get_kernel(get_tensor_backend(out), result_type, P_SUBTRACT);

    if (!binary_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    tensor_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);


    free(inputs);

    return out;
}

Tensor *multiply(Tensor *a, Tensor *b) {

    Tensor *out = binary_step_1_make_compatiable_type_and_shape(a, b);

    dtype result_type =  get_tensor_dtype(out);

    Tensor **inputs = malloc(sizeof(Tensor *) * 2);
    inputs[0] = a;
    inputs[1] = b;
    
    Iterator *iter = binary_step_2_make_iterator(inputs, 2, out, result_type);

    Tensor_Iterator_Kernel kernel = get_kernel(get_tensor_backend(out), result_type, P_MULTIPLY);

    if (!binary_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    //     // Encapsulate
    // out->node = node_create(out, backward_mul, "MUL", "Backward_Mul");
    // node_tensor_to_inputs(a, out->node);
    // node_tensor_to_inputs(b, out->node);



    tensor_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);


    free(inputs);

    return out;
}

Tensor *divide(Tensor *a, Tensor *b) {
    
    Tensor *out = binary_step_1_make_compatiable_type_and_shape(a, b);

    dtype result_type =  get_tensor_dtype(out);

    Tensor **inputs = malloc(sizeof(Tensor *) * 2);
    inputs[0] = a;
    inputs[1] = b;
    
    Iterator *iter = binary_step_2_make_iterator(inputs, 2, out, result_type);

    Tensor_Iterator_Kernel kernel = get_kernel(get_tensor_backend(out), result_type, P_DIVIDE);
    
    if (!binary_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    tensor_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);


    free(inputs);

    return out;
}

Tensor *power(Tensor *a, Tensor *b) {

    Tensor *out = binary_step_1_make_compatiable_type_and_shape(a, b);

    dtype result_type =  get_tensor_dtype(out);

    Tensor **inputs = malloc(sizeof(Tensor *) * 2);
    inputs[0] = a;
    inputs[1] = b;
    
    Iterator *iter = binary_step_2_make_iterator(inputs, 2, out, result_type);

    Tensor_Iterator_Kernel kernel = get_kernel(get_tensor_backend(out), result_type, P_POWER);

    if (!binary_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    tensor_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);


    free(inputs);

    return out;
}

Tensor *equal(Tensor *a, Tensor *b) {
    
    Tensor *out = binary_step_1_make_compatiable_type_and_shape(a, b);

    dtype result_type =  get_tensor_dtype(out);

    Tensor **inputs = malloc(sizeof(Tensor *) * 2);
    inputs[0] = a;
    inputs[1] = b;
    
    Iterator *iter = binary_step_2_make_iterator(inputs, 2, out, result_type);

    Tensor_Iterator_Kernel kernel = get_kernel(get_tensor_backend(out), result_type, P_EQUALITY);

    if (!binary_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    tensor_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);


    free(inputs);

    return out;

}

Tensor *maximum(Tensor *a, Tensor *b) {
    
    Tensor *out = binary_step_1_make_compatiable_type_and_shape(a, b);

    dtype result_type =  get_tensor_dtype(out);

    Tensor **inputs = malloc(sizeof(Tensor *) * 2);
    inputs[0] = a;
    inputs[1] = b;
    
    Iterator *iter = binary_step_2_make_iterator(inputs, 2, out, result_type);

    Tensor_Iterator_Kernel kernel = get_kernel(get_tensor_backend(out), result_type, P_ELEMENT_WISE_MAXIMUM);

    if (!binary_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    tensor_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);


    free(inputs);

    return out;


}

Tensor *minimum(Tensor *a, Tensor *b) {
    Tensor *out = binary_step_1_make_compatiable_type_and_shape(a, b);

    dtype result_type =  get_tensor_dtype(out);

    Tensor **inputs = malloc(sizeof(Tensor *) * 2);
    inputs[0] = a;
    inputs[1] = b;
    
    Iterator *iter = binary_step_2_make_iterator(inputs, 2, out, result_type);

    Tensor_Iterator_Kernel kernel = get_kernel(get_tensor_backend(out), result_type, P_ELEMENT_WISE_MINIMUM);

    if (!binary_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    tensor_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);


    free(inputs);

    return out;

}

Tensor *greater_than(Tensor *a, Tensor *b) {
    
    Tensor *out = binary_step_1_make_compatiable_type_and_shape(a, b);

    dtype result_type =  get_tensor_dtype(out);

    Tensor **inputs = malloc(sizeof(Tensor *) * 2);
    inputs[0] = a;
    inputs[1] = b;
    
    Iterator *iter = binary_step_2_make_iterator(inputs, 2, out, result_type);

    Tensor_Iterator_Kernel kernel = get_kernel(get_tensor_backend(out), result_type, P_GREATER_THAN);

    if (!binary_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    tensor_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);


    free(inputs);

    return out;

}

Tensor *less_than(Tensor *a, Tensor *b) {
    
    Tensor *out = binary_step_1_make_compatiable_type_and_shape(a, b);

    dtype result_type =  get_tensor_dtype(out);

    Tensor **inputs = malloc(sizeof(Tensor *) * 2);
    inputs[0] = a;
    inputs[1] = b;
    
    Iterator *iter = binary_step_2_make_iterator(inputs, 2, out, result_type);

    Tensor_Iterator_Kernel kernel = get_kernel(get_tensor_backend(out), result_type, P_LESS_THAN);

    if (!binary_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    tensor_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);


    free(inputs);

    return out;

}

/* #endregion */



/* #region Unary Operations */

Tensor *square_root(Tensor *a) {

    dtype ten_a = get_tensor_dtype(a);

    dtype result_type = promote_to_type(ten_a, DTYPE_F64);

    Tensor *out =  unary_step_1_make_output_tensor(a, result_type);

    Tensor **inputs = malloc(sizeof(Tensor *));

    Iterator *iter =  unary_step_2_make_iterator(inputs, 1, out, result_type);

    Tensor_Iterator_Kernel kernel = get_kernel(get_tensor_backend(out), result_type, P_SQUARE_ROOT);

    if (!binary_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }


    tensor_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);

    free(inputs);

    return out;
}

Tensor *absolute_value(Tensor *a) {

    dtype result_type = get_tensor_dtype(a);

    Tensor *out = unary_step_1_make_output_tensor(a, result_type);

    Tensor **inputs = malloc(sizeof(Tensor *));

    inputs[0] = a;

    Iterator *iter = unary_step_2_make_iterator(inputs, 1, out, result_type);

    Tensor_Iterator_Kernel kernel = get_kernel(get_tensor_backend(out), result_type, P_ABSOLUTE_VALUE);

    if (!binary_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    tensor_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);


    free(inputs);

    return out;
}

Tensor *negate(Tensor *a) {

    dtype result_type = get_tensor_dtype(a);

    Tensor *out = unary_step_1_make_output_tensor(a, result_type);

    Tensor **inputs = malloc(sizeof(Tensor *));

    inputs[0] = a;

    Iterator *iter = unary_step_2_make_iterator(inputs, 1, out, result_type);

    Tensor_Iterator_Kernel kernel = get_kernel(get_tensor_backend(out), result_type, P_NEGATION);

    if (!binary_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    tensor_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);


    free(inputs);

    return out;
}


/* #endregion */


Tensor *where(Tensor *condition, Tensor *a, Tensor *b) {

    if (get_tensor_dtype(condition) != DTYPE_BOOL) {
        error_print("%s expected condition to have type: DTYPE_BOOL, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(DTYPE_INPUT_UNSUPPORTED),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    Tensor *out = binary_step_1_make_compatiable_type_and_shape(a, b);

    dtype result_type =  get_tensor_dtype(out);

    Tensor **inputs = malloc(sizeof(Tensor *) * 3);

    inputs[0] = condition;
    inputs[1] = a;
    inputs[2] = b;

    out = broadcast_inputs_to_tensor(inputs, result_type, get_tensor_backend(a));


    Iterator *iter = binary_step_2_make_iterator(inputs, 3, out, result_type);

    Tensor_Iterator_Kernel kernel = get_kernel(get_tensor_backend(out), result_type, P_WHERE);

    if (!binary_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    tensor_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);


    free(inputs);

    return out;
}

Tensor *fill(int64_t ndim, int64_t *shape, dtype dtype, Backend backend, void *value) {

    Tensor *a = scalar_to_tensor(value, dtype, backend);

    Tensor *out = poisoned_blank_tensor(ndim, shape, dtype, backend);

    Tensor **inputs = malloc(sizeof(Tensor *));

    inputs[0] = a;

    Iterator *iter = binary_step_2_make_iterator(inputs, 1, out, dtype);

    Tensor_Iterator_Kernel kernel = get_kernel(get_tensor_backend(out), dtype, P_FILL);

    if (!binary_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    tensor_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);


    free(inputs);

    return out;
}

Tensor *arange(int64_t start, int64_t end, int64_t step, dtype dtype, Backend backend) {

    if (step <= 0) { 
        
        printf("Step size must be >= 0\n");
        
        return NULL;

    }

    int64_t size = tensor_arange_helper(start, end, step);

    Tensor *out = poisoned_blank_tensor(1, (int64_t[]){size}, dtype, backend);

    SET_TENSOR_VALUE(dtype, out, 0, start);

    int j = start;

    for (int i = 1; i < size; i++) {
        j += step;
        SET_TENSOR_VALUE(dtype, out, i, j);
    }

    return out;

}


// /* Reductions */

Tensor *sum(Tensor *a, int64_t *dims, int64_t ndims, bool keepdim) {

    Tensor *out = redeuction_step_1_make_output_tensor(a, dims, ndims, keepdim);

    Tensor **inputs = malloc(sizeof(Tensor *));

    inputs[0] = a;

    Iterator *iter = redeuction_step_2_make_iterator(inputs, 1, out, dims, ndims, keepdim, true, 0);

    Tensor_Reduction_Kernel kernel = get_reduction_kernel(get_tensor_backend(out), get_tensor_dtype(out), P_SUM_REDUCTION);

    if (!reduction_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    tensor_reduction_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);

    free(inputs);

    return out;

}

Tensor *max(Tensor *a, int64_t *dims, int64_t ndims, bool keepdim) {

    Tensor *out = redeuction_step_1_make_output_tensor(a, dims, ndims, keepdim);

    Tensor **inputs = malloc(sizeof(Tensor *));

    inputs[0] = a;

    Iterator *iter = redeuction_step_2_make_iterator(inputs, 1, out, dims, ndims, keepdim, true, INT64_MIN);

    Tensor_Reduction_Kernel kernel = get_reduction_kernel(get_tensor_backend(out), get_tensor_dtype(out), P_MAX_REDUCTION);

    if (!reduction_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    tensor_reduction_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);

    free(inputs);

    return out;


}

Tensor *min(Tensor *a, int64_t *dims, int64_t ndims, bool keepdim) {
    
    Tensor *out = redeuction_step_1_make_output_tensor(a, dims, ndims, keepdim);

    Tensor **inputs = malloc(sizeof(Tensor *));

    inputs[0] = a;

    Iterator *iter = redeuction_step_2_make_iterator(inputs, 1, out, dims, ndims, keepdim, true, INT64_MAX);

    Tensor_Reduction_Kernel kernel = get_reduction_kernel(get_tensor_backend(out), get_tensor_dtype(out), P_MIN_REDUCTION);

    if (!reduction_step_3_check_kernel_exists(kernel)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(COULD_NOT_ACQUIRE_KERNEL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    tensor_reduction_iterator_serial_for_each(iter, kernel);
    tensor_iterator_free(iter);

    free(inputs);

    return out;

}


// /* Shape Manipulations */

Tensor *reshape(Tensor *a, int64_t *new_shape, int64_t ndim) {
    if (!a || !new_shape || ndim <= 0) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(INVALID_ARGUMENTS),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    // Calculate total size of new shape
    int64_t new_size = 1;
    int64_t infer_dim = -1;
    
    for (int64_t i = 0; i < ndim; i++) {
        if (new_shape[i] == -1) {
            if (infer_dim != -1) {
                error_print("Only one dimension can be inferred (-1) in function: %s, file: %s, line: %d, RETURN: NULL\n",  __func__, __FILE__, __LINE__);
                return NULL;
            }
            infer_dim = i;
        } else if (new_shape[i] <= 0) {
            error_print("Invalid dimension size: %lld in function: %s, file: %s, line: %d, RETURN: NULL\n", new_shape[i],  __func__, __FILE__, __LINE__);
            return NULL;
        } else {
            new_size *= new_shape[i];
        }
    }
    
    // Handle dimension inference
    if (infer_dim != -1) {
        if (a->size % new_size != 0) {
            error_print("Cannot infer dimension size in function: %s, file: %s, line: %d, RETURN: NULL\n",  __func__, __FILE__, __LINE__);
            return NULL;
        }
        new_shape[infer_dim] = a->size / new_size;
        new_size = a->size;
    }

    if (new_size != a->size) {
        error_print("Total size mismatch: expected %lld, got %lld in function: %s, file: %s, line: %d, RETURN: NULL\n", a->size, new_size,  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    Tensor *out = poisoned_blank_tensor(ndim, new_shape, get_tensor_dtype(a), get_tensor_backend(a));

    if (!out) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    // Copy data directly since reshape doesn't change data layout for contiguous tensors
    size_t data_size = a->size * get_dtype_size(get_tensor_dtype(a));
    memcpy(out->data, a->data, data_size);
    
    return out;
}

Tensor *squeeze(Tensor *a, int64_t dim) {
    if (!a) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    // Handle negative dimension
    if (dim < 0) {
        dim += a->ndim;
    }
    
    // Validate dimension
    if (dim < 0 || dim >= a->ndim) {
        error_print("Dimension %lld out of range for tensor with %lld dimensions in function: %s, file: %s, line: %d, RETURN: NULL\n",  dim, a->ndim,  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    // Check if dimension size is 1
    if (a->shape[dim] != 1) {
        error_print("Cannot squeeze dimension %lld with size %lld in function: %s, file: %s, line: %d, RETURN: NULL\n", dim, a->shape[dim],  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    // Create new shape without the squeezed dimension
    int64_t new_ndim = a->ndim - 1;
    if (new_ndim == 0) {
        // Result is a scalar
        new_ndim = 1;
        int64_t scalar_shape[] = {1};
        return reshape(a, scalar_shape, 1);
    }
    
    int64_t *new_shape = malloc(new_ndim * sizeof(int64_t));
    if (!new_shape) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(FAILED_TO_ALLOCATE_MEMORY),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    int64_t new_idx = 0;
    for (int64_t i = 0; i < a->ndim; i++) {
        if (i != dim) {
            new_shape[new_idx++] = a->shape[i];
        }
    }
    
    Tensor *result = reshape(a, new_shape, new_ndim);
    free(new_shape);
    return result;
}

Tensor *unsqueeze(Tensor *a, int64_t dim) {
    if (!a) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    int64_t new_ndim = a->ndim + 1;
    
    // Handle negative dimension
    if (dim < 0) {
        dim += new_ndim;
    }
    
    // Validate dimension
    if (dim < 0 || dim >= new_ndim) {
        error_print("Dimension %lld out of range for tensor with %lld dimensions in function: %s, file: %s, line: %d, RETURN: NULL\n",  dim, a->ndim,  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    // Create new shape with dimension of size 1 inserted
    int64_t *new_shape = malloc(new_ndim * sizeof(int64_t));
    if (!new_shape) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(FAILED_TO_ALLOCATE_MEMORY),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    int64_t old_idx = 0;
    for (int64_t i = 0; i < new_ndim; i++) {
        if (i == dim) {
            new_shape[i] = 1;
        } else {
            new_shape[i] = a->shape[old_idx++];
        }
    }
    
    Tensor *result = reshape(a, new_shape, new_ndim);
    free(new_shape);
    return result;
}


