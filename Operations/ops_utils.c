#include "C:\Programming\Revamp2\Operations\ops_utils.h"


/* Binary Operations */

Tensor *binary_step_1_make_compatiable_type_and_shape(Tensor *a, Tensor *b) {

    dtype ten_a = get_tensor_dtype(a);
    dtype ten_b = get_tensor_dtype(b);

    dtype result_type = promote_to_type(ten_a, ten_b);

    Tensor *op_a = (ten_a == result_type) ? a : cast_to_type(a, result_type, false);
    Tensor *op_b = (ten_b == result_type) ? b : cast_to_type(b, result_type, false);
    
    if (!op_a || !op_b) { if (op_a && op_a != a) tensor_free(op_a); if (op_b && op_b != b) tensor_free(op_b); return NULL; }

    Tensor *out = tensor_like_with_type(op_a, result_type);
    if (!out) { if (op_a != a) tensor_free(op_a); if (op_b != b) tensor_free(op_b); return NULL; }

    if (!tensor_broadcast_output(op_a, op_b, out)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_FAILED_TO_BROADCAST_INPUTS_WITH_OUTPUT),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_FAILED_TO_BROADCAST_INPUTS_WITH_OUTPUT, "%s, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_FAILED_TO_BROADCAST_INPUTS_WITH_OUTPUT),  __func__, __FILE__, __LINE__);
        error_set(error, a, TENSOR_FAILED_TO_BROADCAST_INPUTS_WITH_OUTPUT);
        error_set(error, b, TENSOR_FAILED_TO_BROADCAST_INPUTS_WITH_OUTPUT);
        tensor_free(out);
        if (op_a != a) tensor_free(op_a);
        if (op_b != b) tensor_free(op_b);
        return NULL;
    }

    return out;

}

Iterator *binary_step_2_make_iterator(Tensor **inputs, int numel, Tensor *out, dtype result_type) {

    Tensor_Config *cfg = tensor_config_make();

    if (!cfg) { 
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(FAILED_TO_CREATE_TENSOR_CONFIG),  __func__, __FILE__, __LINE__);
        return NULL; 
    }

    size_t elem_size = get_dtype_size(result_type);


    for (int i = 0; i < numel; i++) {
        tensor_config_add_operand(cfg, inputs[i]->data, get_tensor_ndim(inputs[i]), get_tensor_stride(inputs[i]), get_tensor_shape(inputs[i]), get_dtype_size(get_tensor_dtype(inputs[i])), TENSOR_INPUT);
    }
    
    tensor_config_add_operand(cfg, out->data, get_tensor_ndim(out),  get_tensor_stride(out), get_tensor_shape(out), get_dtype_size(get_tensor_dtype(out)), TENSOR_OUTPUT);

    Iterator *iter = tensor_iterator_make();
    if (!iter) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(FAILED_TO_CREATE_TENSOR_ITERATOR),  __func__, __FILE__, __LINE__);
        free(cfg);
        free(iter);
        return NULL;
    }

    if (!tensor_iterator_build(cfg, iter)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(FAILED_TO_BUILD_TENSOR_ITERATOR),  __func__, __FILE__, __LINE__);
        tensor_iterator_free(iter);
        free(cfg);
        free(iter);
        return NULL;
    }

    return iter;

}

bool binary_step_3_check_kernel_exists(Tensor_Iterator_Kernel kernel) {
        
    if (!kernel) {
        return false;
    }

    return true;

}


/* Unary Operations */

Tensor *unary_step_1_make_output_tensor(Tensor *a, dtype result_type) {

    Tensor *op_a = a;

    Tensor *out = tensor_like_with_type(op_a, result_type);

    if (!out) { if (op_a != a) tensor_free(op_a);  return NULL; }

}

Iterator *unary_step_2_make_iterator(Tensor **inputs, int numel, Tensor *out, dtype result_type) {
    
    Tensor_Config *cfg = tensor_config_make();

    if (!cfg) { 
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(FAILED_TO_CREATE_TENSOR_CONFIG),  __func__, __FILE__, __LINE__);
        return NULL; 
    }

    size_t elem_size = get_dtype_size(result_type);

    for (int i = 0; i < numel; i++) {
        tensor_config_add_operand(cfg, inputs[i]->data, get_tensor_ndim(inputs[i]), get_tensor_stride(inputs[i]), get_tensor_shape(inputs[i]), elem_size, TENSOR_INPUT);
    }
    
    tensor_config_add_operand(cfg, out->data, get_tensor_ndim(out),  get_tensor_stride(out), get_tensor_shape(out), elem_size, TENSOR_OUTPUT);

    
    Iterator *iter = tensor_iterator_make();

    if (!iter) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(FAILED_TO_CREATE_TENSOR_ITERATOR),  __func__, __FILE__, __LINE__);
        free(cfg);
        free(iter);
        return NULL;
    }
    
    if (!tensor_iterator_build(cfg, iter)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(FAILED_TO_BUILD_TENSOR_ITERATOR),  __func__, __FILE__, __LINE__);
        tensor_iterator_free(iter);
        free(cfg);
        free(iter);
        return NULL;
    }

    return iter;



}


/* Reduction Operations */

Tensor *redeuction_step_1_make_output_tensor(Tensor *a, int64_t *dims, int64_t ndims, bool keepdim) {
    
    dtype ten_a = get_tensor_dtype(a);

    Tensor_Config *cfg = tensor_config_make();

    int64_t *shape;
    int64_t ndim;

    compute_reduction_shape_ndim(a, dims, ndims, keepdim, &cfg->_reduce, &shape, &ndim);
    Tensor *out = poisoned_blank_tensor(ndim, shape, ten_a, get_tensor_backend(a));

    return out;
}


Iterator *redeuction_step_2_make_iterator(Tensor **inputs, int numel, Tensor *out, int64_t *dims, int64_t ndims, bool keepdim, bool set_special_value, int64_t special_value) {

    Tensor_Config *cfg = tensor_config_make();

    if (!cfg) { 
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(FAILED_TO_CREATE_TENSOR_CONFIG),  __func__, __FILE__, __LINE__);
        return NULL; 
    }

    int64_t *shape;
    int64_t ndim;

    compute_reduction_shape_ndim(inputs[0], dims, ndims, keepdim, &cfg->_reduce, &shape, &ndim);

    cfg->set_out_special_value = set_special_value;
    cfg->out_special_value = special_value;

    tensor_config_set_mode(cfg, ITERATOR_REDUCTION);
    tensor_config_set_reduction(cfg, dims, ndims, keepdim);

    for (int i = 0; i < numel; i++) {
        tensor_config_add_operand(cfg, inputs[i]->data, get_tensor_ndim(inputs[i]), get_tensor_stride(inputs[i]), get_tensor_shape(inputs[i]), get_dtype_size(get_tensor_dtype(inputs[i])), TENSOR_INPUT);
    }
    
    tensor_config_add_operand(cfg, out->data, get_tensor_ndim(out),  get_tensor_stride(out), get_tensor_shape(out), get_dtype_size(get_tensor_dtype(out)), TENSOR_REDUCTION_OUTPUT);

    Iterator *iter = tensor_iterator_make();
    if (!iter) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(FAILED_TO_CREATE_TENSOR_ITERATOR),  __func__, __FILE__, __LINE__);
        free(cfg);
        free(iter);
        return NULL;
    }

    if (!tensor_reduction_iterator_build(cfg, iter)) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(FAILED_TO_BUILD_TENSOR_ITERATOR),  __func__, __FILE__, __LINE__);
        tensor_iterator_free(iter);
        free(cfg);
        free(iter);
        return NULL;
    }

    return iter;



}


bool reduction_step_3_check_kernel_exists(Tensor_Reduction_Kernel kernel) {
        
    if (!kernel) {
        return false;
    }

    return true;

}

