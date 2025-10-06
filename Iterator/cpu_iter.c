#include "C:\Programming\Revamp2\Iterator\cpu_iter.h"

IterError cpu_iter_init(cpu_iter *iter, Tensor *tensor, bool grad) {
    if (!iter || !tensor) {
        return ITER_ERROR_INVALID_TENSOR;
    }
    
    iter->tensor_object = tensor;
    iter->ndim = tensor->ndim;
    iter->self_tensor_size = tensor->size;
    iter->counter = 0;
    
    // Allocate arrays
    iter->stride = (int64_t *)malloc(sizeof(int64_t) * tensor->ndim);
    iter->backstrides = (int64_t *)calloc(tensor->ndim, sizeof(int64_t));
    iter->coordinates = (int64_t *)calloc(tensor->ndim, sizeof(int64_t));
    iter->shape = (int64_t *)malloc(sizeof(int64_t) * tensor->ndim);
    
    if (!iter->stride || !iter->backstrides || !iter->coordinates || !iter->shape) {
        cpu_iter_free(iter);
        return ITER_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy tensor metadata
    memcpy(iter->stride, tensor->stride, sizeof(int64_t) * tensor->ndim);
    memcpy(iter->shape, tensor->shape, sizeof(int64_t) * tensor->ndim);
    
    // Calculate backstrides
    for (int i = 0; i < iter->ndim; i++) {
        iter->backstrides[i] = iter->stride[i] * (iter->shape[i] - 1);
    }

    // Set data pointer based on tensor type
    switch (tensor->mdata->type_info->dtype) {
        case DTYPE_I64:
            iter->data_ptr = (char *)tensor->data;
            break;
        case DTYPE_F32:
            iter->data_ptr = (char *)tensor->data;
            break;
        case DTYPE_F64:
            iter->data_ptr = (char *)tensor->data;
            break;
        default:
            cpu_iter_free(iter);
            return ITER_ERROR_INVALID_TENSOR;
        }

    
    return ITER_OK;
}


IterError cpu_iter_init_for_grad(cpu_iter *iter, Tensor *tensor) {
    if (!iter || !tensor) {
        return ITER_ERROR_INVALID_TENSOR;
    }
    
    iter->tensor_object = tensor;
    iter->ndim = tensor->ndim;
    iter->self_tensor_size = tensor->size;
    iter->counter = 0;
    
    // Allocate arrays
    iter->stride = (int64_t *)malloc(sizeof(int64_t) * tensor->ndim);
    iter->backstrides = (int64_t *)calloc(tensor->ndim, sizeof(int64_t));
    iter->coordinates = (int64_t *)calloc(tensor->ndim, sizeof(int64_t));
    iter->shape = (int64_t *)malloc(sizeof(int64_t) * tensor->ndim);
    
    if (!iter->stride || !iter->backstrides || !iter->coordinates || !iter->shape) {
        cpu_iter_free(iter);
        return ITER_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy tensor metadata
    memcpy(iter->stride, tensor->stride, sizeof(int64_t) * tensor->ndim);
    memcpy(iter->shape, tensor->shape, sizeof(int64_t) * tensor->ndim);
    
    // Calculate backstrides
    for (int i = 0; i < iter->ndim; i++) {
        iter->backstrides[i] = iter->stride[i] * (iter->shape[i] - 1);
    }

    // Set data pointer based on tensor type
    switch (tensor->mdata->type_info->dtype) {
        case DTYPE_I64:
            iter->data_ptr = (char *)tensor->grad;
            break;
        case DTYPE_F32:
            iter->data_ptr = (char *)tensor->grad;
            break;
        case DTYPE_F64:
            iter->data_ptr = (char *)tensor->grad;
            break;
        default:
            cpu_iter_free(iter);
            return ITER_ERROR_INVALID_TENSOR;
        }

    
    return ITER_OK;
}

void cpu_iter_free(cpu_iter *iter) {
    if (!iter) return;
    
    free(iter->stride);
    free(iter->backstrides);
    free(iter->coordinates);
    free(iter->shape);
    
    // Zero out pointers to avoid double-free
    iter->stride = NULL;
    iter->backstrides = NULL;
    iter->coordinates = NULL;
    iter->shape = NULL;
    iter->data_ptr = NULL;
}

bool cpu_iter_has_next(const cpu_iter *iter) {
    return iter->counter < iter->self_tensor_size;
}

void cpu_iter_next(cpu_iter *iter) {
    if (!iter) return;
    
    iter->counter++;
    
    // Update coordinates and data pointer
    for (int i = iter->ndim - 1; i >= 0; i--) {
        if (iter->coordinates[i] + 1 < iter->shape[i]) {
            iter->coordinates[i]++;
            iter->data_ptr += iter->stride[i];
            break;
        } else {
            iter->coordinates[i] = 0;
            iter->data_ptr -= iter->backstrides[i];
        }
    }
}

void cpu_iter_reset(cpu_iter *iter) {
    if (!iter) return;
    
    memset(iter->coordinates, 0, iter->ndim * sizeof(int64_t));
    iter->counter = 0;
    
    // Reset data pointer
    switch (iter->tensor_object->mdata->type_info->dtype) {
        case DTYPE_I64:
            iter->data_ptr = (char *)iter->tensor_object->data;
            break;
        case DTYPE_F32:
            iter->data_ptr = (char *)iter->tensor_object->data;
            break;
        case DTYPE_F64:
            iter->data_ptr = (char *)iter->tensor_object->data;
            break;
    }
}

static inline void* cpu_iter_get_current(const cpu_iter *iter) {
    return iter->data_ptr;
}

int cpu_iter_get_int(const cpu_iter *iter) {
    return *(int64_t*)iter->data_ptr;
}

float cpu_iter_get_float(const cpu_iter *iter) {
    return *(float*)iter->data_ptr;
}

double cpu_iter_get_double(const cpu_iter *iter) {
    return *(double*)iter->data_ptr;
}

static inline void cpu_iter_set_int(cpu_iter *iter, int64_t value) {
    *(int64_t*)iter->data_ptr = value;
}

static inline void cpu_iter_set_float(cpu_iter *iter, float value) {
    *(float*)iter->data_ptr = value;
}

static inline void cpu_iter_set_double(cpu_iter *iter, double value) {
    *(double*)iter->data_ptr = value;
}

IterError cpu_iter_broadcast(int num_iters, cpu_iter *iters) {
    if (!iters || num_iters <= 0) {
        return ITER_ERROR_INVALID_TENSOR;
    }
    
    // Find maximum dimension
    int max_dim = 0;
    for (int i = 0; i < num_iters; i++) {
        if (max_dim < iters[i].ndim) {
            max_dim = iters[i].ndim;
        }
    }
    
    // Allocate broadcast shape array
    int64_t *broadcast_shape = (int64_t*)calloc(max_dim, sizeof(int64_t));
    if (!broadcast_shape) {
        return ITER_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize broadcast shape to 1
    for (int i = 0; i < max_dim; i++) {
        broadcast_shape[i] = 1;
    }
    
    // Determine broadcast shape and check compatibility
    for (int i = 0; i < num_iters; i++) {
        int temp_md = max_dim;
        for (int j = iters[i].ndim - 1; j >= 0; j--) {
            temp_md--;
            
            // Check if dimensions are broadcastable
            if (broadcast_shape[temp_md] != 1 && 
                iters[i].shape[j] != 1 && 
                broadcast_shape[temp_md] != iters[i].shape[j]) {
                free(broadcast_shape);
                return ITER_ERROR_BROADCAST_INCOMPATIBLE;
            }
            
            if (broadcast_shape[temp_md] == 1) {
                broadcast_shape[temp_md] = iters[i].shape[j];
            }
        }
    }
    
    // Adjust strides and backstrides for broadcasting
    for (int i = 0; i < num_iters; i++) {
        // Reallocate arrays if needed
        if (iters[i].ndim < max_dim) {
            int64_t *new_stride = (int64_t*)realloc(iters[i].stride, max_dim * sizeof(int64_t));
            int64_t *new_backstrides = (int64_t*)realloc(iters[i].backstrides, max_dim * sizeof(int64_t));
            int64_t *new_coordinates = (int64_t*)realloc(iters[i].coordinates, max_dim * sizeof(int64_t));
            int64_t *new_shape = (int64_t*)realloc(iters[i].shape, max_dim * sizeof(int64_t));
            
            if (!new_stride || !new_backstrides || !new_coordinates || !new_shape) {
                free(broadcast_shape);
                return ITER_ERROR_OUT_OF_MEMORY;
            }
            
            iters[i].stride = new_stride;
            iters[i].backstrides = new_backstrides;
            iters[i].coordinates = new_coordinates;
            iters[i].shape = new_shape;
            
            // Initialize new dimensions
            for (int k = iters[i].ndim; k < max_dim; k++) {
                iters[i].coordinates[k] = 0;
            }
        }
        
        int temp_md = max_dim;
        for (int j = iters[i].ndim - 1; j >= 0; j--) {
            temp_md--;
            if (broadcast_shape[temp_md] != iters[i].shape[j]) {
                iters[i].stride[j] = 0;
                iters[i].backstrides[j] = 0;
            } else {
                iters[i].backstrides[j] = iters[i].stride[j] * (broadcast_shape[temp_md] - 1);
            }
        }
        
        // Handle prepended dimensions (size 1)
        for (int j = 0; j < max_dim - iters[i].ndim; j++) {
            iters[i].stride[j] = 0;
            iters[i].backstrides[j] = 0;
        }
    }
    
    // Calculate total size of broadcasted shape
    int total_size = 1;
    for (int i = 0; i < max_dim; i++) {
        total_size *= broadcast_shape[i];
    }
    
    // Update iterator metadata
    for (int i = 0; i < num_iters; i++) {
        memcpy(iters[i].shape, broadcast_shape, max_dim * sizeof(int64_t));
        iters[i].ndim = max_dim;
        iters[i].self_tensor_size = total_size;
        cpu_iter_reset(&iters[i]);
    }
    
    free(broadcast_shape);
    return ITER_OK;
}

