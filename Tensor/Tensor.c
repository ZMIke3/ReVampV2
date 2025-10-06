#include "C:\Programming\Revamp2\Tensor\Tensor.h"
#include "C:\Programming\Revamp2\Iterator\cpu_iter.h"

/* Tensor Types */

static const TypeInfo type_registry[] = {
    {DTYPE_I8,    sizeof(int8_t),    "int8",    false, true,  false},
    {DTYPE_I16,   sizeof(int16_t),   "int16",   false, true,  false},
    {DTYPE_I32,   sizeof(int32_t),   "int32",   false, true,  false},
    {DTYPE_I64,   sizeof(int64_t),   "int64",   false, true,  false},
    {DTYPE_U8,   sizeof(uint8_t),   "uint8",   false, false, false},
    {DTYPE_U16,  sizeof(uint16_t),  "uint16",  false, false, false},
    {DTYPE_U32,  sizeof(uint32_t),  "uint32",  false, false, false},
    {DTYPE_U64,  sizeof(uint64_t),  "uint64",  false, false, false},
    {DTYPE_F16, sizeof(uint16_t),  "float16", true,  true,  false},
    {DTYPE_F32, sizeof(float),     "float32", true,  true,  false},
    {DTYPE_F64, sizeof(double),    "float64", true,  true,  false},
    {DTYPE_BOOL,    sizeof(bool),      "bool",    false, false, false},
};

// Start:
// Add error checks for these

TypeInfo *get_tensor_type_info(Tensor *a) {
    return a->mdata->type_info;
}

TypeInfo *set_tensor_type_info(dtype dtype) {
    return &type_registry[dtype];
}

TypeInfo *get_type_info(dtype dtype) {
    return &type_registry[dtype];
}

dtype get_tensor_dtype(Tensor *a) {
    return a->mdata->type_info->dtype;
}

dtype promote_to_type(dtype type_1, dtype type_2) {
    if (type_1 == type_2) return type_1;
    
    // Validate input types
    if (type_1 >= DTYPE_COUNT || type_2 >= DTYPE_COUNT) {
        return DTYPE_COUNT; // Error value
    }
    
    const TypeInfo *info_a = get_type_info(type_1);
    const TypeInfo *info_b = get_type_info(type_2);
    
    // Rule 1: Float64 always wins
    if (type_1 == DTYPE_F64 || type_2 == DTYPE_F64) {
        return DTYPE_F64;
    }
    
    // Rule 2: Float32 wins over everything except Float64
    if (type_1 == DTYPE_F32 || type_2 == DTYPE_F32) {
        return DTYPE_F32;
    }
    
    // Rule 3: Float16 wins over integers and bool
    if (type_1 == DTYPE_F16 || type_2 == DTYPE_F16) {
        return DTYPE_F16;
    }
    
    // Rule 4: Handle integer promotions
    // Special cases for mixed signed/unsigned of same size
    if (info_a->dtype_size == info_b->dtype_size) {
        if (info_a->is_signed && !info_b->is_signed) {
            // Signed + unsigned of same size -> promote to next larger signed
            switch (info_a->dtype_size) {
                case 1: return DTYPE_I16;
                case 2: return DTYPE_I32;
                case 4: return DTYPE_I64;
                case 8: return DTYPE_F64; // Can't go larger than 64-bit int
            }
        } else if (!info_a->is_signed && info_b->is_signed) {
            // Same as above, but reversed
            switch (info_b->dtype_size) {
                case 1: return DTYPE_I16;
                case 2: return DTYPE_I32;
                case 4: return DTYPE_I64;
                case 8: return DTYPE_F64;
            }
        }
    }
    
    // Rule 5: Promote to larger type
    if (info_a->dtype_size > info_b->dtype_size) {
        return type_1;
    } else if (info_b->dtype_size > info_a->dtype_size) {
        return type_2;
    }
    
    // Rule 6: Same size, prefer signed over unsigned (shouldn't reach here with above rules)
    return info_a->is_signed ? type_1 : type_2;
}

size_t get_dtype_size(dtype dtype) {
    return type_registry[dtype].dtype_size;
}

// End

void print_tensor_dtype(dtype dtype) {
    switch (dtype) {
        case DTYPE_I8:
            printf("Int8\n");
            break;
        case DTYPE_I16:
            printf("Int16\n");
            break;
        case DTYPE_I32:
            printf("Int32\n");
            break;
        case DTYPE_I64:
            printf("Int64\n");
            break;
        case DTYPE_U8:
            printf("Uint8\n");
            break;
        case DTYPE_U16:
            printf("Uint16\n");
            break;
        case DTYPE_U32:
            printf("Uint32\n");
            break;
        case DTYPE_U64:
            printf("Uint64\n");
            break;
        case DTYPE_F16:
            printf("Float16\n");
            break;
        case DTYPE_F32:
            printf("Float32\n");
            break;
        case DTYPE_F64:
            printf("Float64\n");
            break;
        case DTYPE_BOOL:
            printf("Bool\n");
            break;
        case DTYPE_COUNT:    
            printf("Invalid type\n");
            break;
    }
}

/* Tensor Utility */

// Start
// Need to add error checks for these function
Tensor *new_blank_tensor() {
    Tensor *t = (Tensor *) malloc(sizeof(Tensor));
    t->mdata = (Metadata *) malloc(sizeof(Metadata));
    t->mdata->type_info = (TypeInfo *) malloc(sizeof(TypeInfo));
    t->ndim = 0;
    t->size = 0;
    // Mdata
    t->mdata->backend = CPU;      
    t->mdata->owns_data = true;
    t->mdata->dtype_set = false;
    t->mdata->data_init = false;
    t->mdata->requires_grad = false;
    t->mdata->type_info->dtype = DTYPE_NONE;
    t->mdata->type_info->dtype_size = 0;
    t->error = NULL;
    return t;
}

Tensor *poisoned_blank_tensor(int64_t ndim, int64_t *shape, dtype dtype, Backend backend) {
    Tensor *t =  new_blank_tensor();
    // Tensor data
    t->shape = malloc(ndim * sizeof(int64_t));
    memcpy(t->shape, shape, ndim * sizeof(int64_t));
    t->stride = calc_stride(ndim, shape, get_dtype_size(dtype));
    t->size = tensor_size_from_shape(shape, ndim);
    t->data = malloc(t->size * get_dtype_size(dtype));
    t->grad = calloc(t->size, get_dtype_size(dtype));
    t->ndim = ndim;
    // Metadata
    t->mdata->backend = backend;      
    t->mdata->type_info = set_tensor_type_info(dtype);

    // Node
    t->node = NULL;
    t->requires_grad = false;
    t->is_leaf = true;
    return t;
}

// Needs to be more descriptive
Tensor *tensor_like(Tensor *a) {
    Tensor *c = new_blank_tensor();
    c->ndim = a->ndim;
    c->size = a->size;
    c->shape = (int64_t *) malloc(a->ndim * sizeof(int64_t));
    c->stride = (int64_t *) malloc(a->ndim * sizeof(a->mdata->type_info->dtype_size));
    c->data = (int64_t *) malloc(a->size * sizeof(a->mdata->type_info->dtype_size));

    memcpy(c->shape, a->shape, a->ndim * sizeof(int64_t));
    memcpy(c->stride, a->stride, a->ndim * sizeof(int64_t));

    *c->mdata = *a->mdata;
    return c;
}

// Needs to be more descriptive
Tensor *tensor_like_with_type(Tensor *a, dtype dtype) {
    Tensor *tensor_with_type = poisoned_blank_tensor(get_tensor_ndim(a), get_tensor_shape(a), dtype, get_tensor_backend(a));
}
// End

const int64_t *get_tensor_shape(Tensor *a) {
    
    if (!a) {

        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    if (!a->shape) {
        error_print("%s, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL_SHAPE),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_NULL_SHAPE, "%s, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL_SHAPE),  __func__, __FILE__, __LINE__);
        error_set(error, a, TENSOR_NULL_SHAPE);
        return NULL;
    }

    return a->shape;
}

void set_tensor_shape_and_ndim_broadcast(Tensor *a, int64_t *shape, int64_t ndim, bool *broadcast_dim) {

    if (!a) {

        error_print("%s in function: %s, file: %s, line: %d, RETURN: NONE\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return;
    }

    if (!shape) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NONE\n", common_error_code_to_string(TENSOR_SHAPE_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_DATA_FROM_C_ARRAY_NULL, "%s, in function: %s, file: %s, line: %d, RETURN: NONE\n", common_error_code_to_string(TENSOR_SHAPE_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        error_set(error, a, TENSOR_SHAPE_FROM_C_ARRAY_NULL);
        return;
    }


    a->shape = shape;
    a->ndim = ndim;
    a->size = tensor_size_from_shape(shape, ndim);

    for (int i = 0; i < ndim; i++) {

        if (broadcast_dim[i]) {
            a->stride[i] = 0;
        } else {
            continue;
        }
    }

}



const int64_t *get_tensor_stride(Tensor *a) {
    
    if (!a) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return NULL;
    }

    if (!a->stride) {
        error_print("%s, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL_STRIDE),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_NULL_STRIDE, "%s, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL_STRIDE),  __func__, __FILE__, __LINE__);
        error_set(error, a, TENSOR_NULL_STRIDE);
        return NULL;
    }

    return a->stride;
}

const int64_t get_tensor_ndim(Tensor *a) {
    if (!a) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return a->ndim;
    }

    return a->ndim;
}

const Backend get_tensor_backend(Tensor *a) {
        
    if (!a) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: BACKEND_COUNT\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return BACKEND_COUNT;
    }

    if (!a->mdata) {
        Error *error = error_create(TENSOR_NULL_METADATA, "%s, in function: %s, file: %s, line: %d, RETURN: BACKEND_COUNT\n", common_error_code_to_string(TENSOR_NULL_METADATA),  __func__, __FILE__, __LINE__);
        error_set(error, a, TENSOR_NULL_METADATA);
        return BACKEND_COUNT;
    }

    return a->mdata->backend;
}


/*Tensor Utility Creation Functions*/

Tensor *new_int8_tensor(int64_t ndim, int64_t *shape, int8_t *data, Backend backend) {
    Tensor *t = poisoned_blank_tensor(ndim, shape, DTYPE_I8, backend);
    if (!t) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    if (!data) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_DATA_FROM_C_ARRAY_NULL, "%s, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        error_set(error, t, TENSOR_DATA_FROM_C_ARRAY_NULL);
        return NULL;
    }
    
    memcpy(t->data, data, t->size * sizeof(int8_t));
    return t;
}

Tensor *new_int16_tensor(int64_t ndim, int64_t *shape, int16_t *data, Backend backend) {
    Tensor *t = poisoned_blank_tensor(ndim, shape, DTYPE_I16, backend);
    if (!t) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    if (!data) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_DATA_FROM_C_ARRAY_NULL, "%s, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        error_set(error, t, TENSOR_DATA_FROM_C_ARRAY_NULL);
        return NULL;
    }
    
    memcpy(t->data, data, t->size * sizeof(int16_t));
    return t;
}

Tensor *new_int32_tensor(int64_t ndim, int64_t *shape, int32_t *data, Backend backend) {
    Tensor *t = poisoned_blank_tensor(ndim, shape, DTYPE_I32, backend);
    if (!t) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    if (!data) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_DATA_FROM_C_ARRAY_NULL, "%s, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        error_set(error, t, TENSOR_DATA_FROM_C_ARRAY_NULL);
        return NULL;
    }
    
    memcpy(t->data, data, t->size * sizeof(int32_t));
    return t;
}

Tensor *new_int64_tensor(int64_t ndim, int64_t *shape, int64_t *data, Backend backend) {
    Tensor *t = poisoned_blank_tensor(ndim, shape, DTYPE_I64, backend);
    if (!t) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    if (!data) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_DATA_FROM_C_ARRAY_NULL, "%s, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        error_set(error, t, TENSOR_DATA_FROM_C_ARRAY_NULL);
        return NULL;
    }
    
    memcpy(t->data, data, t->size * sizeof(int64_t));
    return t;
}

Tensor *new_uint8_tensor(int64_t ndim, int64_t *shape, uint8_t *data, Backend backend) {
    Tensor *t = poisoned_blank_tensor(ndim, shape, DTYPE_U8, backend);
    if (!t) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    if (!data) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_DATA_FROM_C_ARRAY_NULL, "%s, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        error_set(error, t, TENSOR_DATA_FROM_C_ARRAY_NULL);
        return NULL;
    }
    
    memcpy(t->data, data, t->size * sizeof(uint8_t));
    return t;
}

Tensor *new_uint16_tensor(int64_t ndim, int64_t *shape, uint16_t *data, Backend backend) {
    Tensor *t = poisoned_blank_tensor(ndim, shape, DTYPE_U16, backend);
    if (!t) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    if (!data) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_DATA_FROM_C_ARRAY_NULL, "%s, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        error_set(error, t, TENSOR_DATA_FROM_C_ARRAY_NULL);
        return NULL;
    }
    
    memcpy(t->data, data, t->size * sizeof(uint16_t));
    return t;
}

Tensor *new_uint32_tensor(int64_t ndim, int64_t *shape, uint32_t *data, Backend backend) {
    Tensor *t = poisoned_blank_tensor(ndim, shape, DTYPE_U32, backend);
    if (!t) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    if (!data) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_DATA_FROM_C_ARRAY_NULL, "%s, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        error_set(error, t, TENSOR_DATA_FROM_C_ARRAY_NULL);
        return NULL;
    }
    
    memcpy(t->data, data, t->size * sizeof(uint32_t));t;
}

Tensor *new_uint64_tensor(int64_t ndim, int64_t *shape, uint64_t *data, Backend backend) {
    Tensor *t = poisoned_blank_tensor(ndim, shape, DTYPE_U64, backend);
    if (!t) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    if (!data) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_DATA_FROM_C_ARRAY_NULL, "%s, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        error_set(error, t, TENSOR_DATA_FROM_C_ARRAY_NULL);
        return NULL;
    }
    
    memcpy(t->data, data, t->size * sizeof(uint64_t));
    return t;
}

Tensor *new_float32_tensor(int64_t ndim, int64_t *shape, float *data, Backend backend) {
    Tensor *t = poisoned_blank_tensor(ndim, shape, DTYPE_F32, backend);
    if (!t) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    if (!data) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_DATA_FROM_C_ARRAY_NULL, "%s, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        error_set(error, t, TENSOR_DATA_FROM_C_ARRAY_NULL);
        return NULL;
    }
    
    memcpy(t->data, data, t->size * sizeof(float));
    return t;
}

Tensor *new_float64_tensor(int64_t ndim, int64_t *shape, double *data, Backend backend) {
    Tensor *t = poisoned_blank_tensor(ndim, shape, DTYPE_F64, backend);
    if (!t) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    if (!data) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_DATA_FROM_C_ARRAY_NULL, "%s, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        error_set(error, t, TENSOR_DATA_FROM_C_ARRAY_NULL);
        return NULL;
    }
    
    memcpy(t->data, data, t->size * sizeof(double));
    return t;
}

Tensor *new_bool_tensor(int64_t ndim, int64_t *shape, bool *data, Backend backend) {
    Tensor *t = poisoned_blank_tensor(ndim, shape, DTYPE_BOOL, backend);
    if (!t) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return NULL;
    }
    
    if (!data) {
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_DATA_FROM_C_ARRAY_NULL, "%s, in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(TENSOR_DATA_FROM_C_ARRAY_NULL),  __func__, __FILE__, __LINE__);
        error_set(error, t, TENSOR_DATA_FROM_C_ARRAY_NULL);
        return NULL;
    }
    
    memcpy(t->data, data, t->size * sizeof(bool));
    return t;
}

Tensor *scalar_to_tensor(void *value, dtype dtype, Backend backend) {
    Tensor *out;

    switch (dtype) {
        case DTYPE_I8:
            return out = new_int8_tensor(1, (int64_t[]){1}, value, backend);
        case DTYPE_I16:
            return out = new_int16_tensor(1, (int64_t[]){1}, value, backend);
        case DTYPE_I32:
            return out = new_int32_tensor(1, (int64_t[]){1}, value, backend);
        case DTYPE_I64:
            return out = new_int64_tensor(1, (int64_t[]){1}, value, backend);
        case DTYPE_U8:
            return out = new_uint8_tensor(1, (int64_t[]){1}, value, backend);
        case DTYPE_U16:
            return out = new_uint16_tensor(1, (int64_t[]){1}, value, backend);
        case DTYPE_U32:
            return out = new_uint32_tensor(1, (int64_t[]){1}, value, backend);
        case DTYPE_U64:
            return out = new_uint64_tensor(1, (int64_t[]){1}, value, backend);
        case DTYPE_F32:
            return out = new_float32_tensor(1, (int64_t[]){1}, value, backend);
        case DTYPE_F64:
            return out = new_float64_tensor(1, (int64_t[]){1}, value, backend);
        case DTYPE_BOOL:
            return out = new_bool_tensor(1, (int64_t[]){1}, value, backend);
    default:
        error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(DTYPE_INPUT_UNSUPPORTED),  __func__, __FILE__, __LINE__);
        return NULL;
        break;
    }

}

Tensor *flat_array_to_tensor(int64_t size, void *value, dtype dtype, Backend backend) {
    Tensor *out;

    switch (dtype) {
        case DTYPE_I8:
            return out = new_int8_tensor(1, (int64_t[]){size}, value, backend);
        case DTYPE_I16:
            return out = new_int16_tensor(1, (int64_t[]){size}, value, backend);
        case DTYPE_I32:
            return out = new_int32_tensor(1, (int64_t[]){size}, value, backend);
        case DTYPE_I64:
            return out = new_int64_tensor(1, (int64_t[]){size}, value, backend);
        case DTYPE_U8:
            return out = new_uint8_tensor(1, (int64_t[]){size}, value, backend);
        case DTYPE_U16:
            return out = new_uint16_tensor(1, (int64_t[]){size}, value, backend);
        case DTYPE_U32:
            return out = new_uint32_tensor(1, (int64_t[]){size}, value, backend);
        case DTYPE_U64:
            return out = new_uint64_tensor(1, (int64_t[]){size}, value, backend);
        case DTYPE_F32:
            return out = new_float32_tensor(1, (int64_t[]){size}, value, backend);
        case DTYPE_F64:
            return out = new_float64_tensor(1, (int64_t[]){size}, value, backend);
        case DTYPE_BOOL:
            return out = new_bool_tensor(1, (int64_t[]){size}, value, backend);
        default:
            error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(DTYPE_INPUT_UNSUPPORTED),  __func__, __FILE__, __LINE__);
            return NULL;
        break;
    }

}

/* Tensor Display*/

void print_tensor_shape(Tensor *a) {

    if (!a) {
        error_print("%s in function: %s, file: %s, line: %d\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return;
    }

    if (!a->shape) {
        error_print("%s, in function: %s, file: %s, line: %d\n", common_error_code_to_string(TENSOR_NULL_SHAPE),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_NULL_SHAPE, "%s, in function: %s, file: %s, line: %d\n", common_error_code_to_string(TENSOR_NULL_SHAPE),  __func__, __FILE__, __LINE__);
        error_set(error, a, TENSOR_NULL_SHAPE);
        return;
    }

    for (int i = 0; i < a->ndim; i++) {
        printf("%lld ", a->shape[i]);
    }
    printf("\n");
}

void print_tensor_stride(Tensor *a) {
    
    if (!a) {
        error_print("%s in function: %s, file: %s, line: %d\n", common_error_code_to_string(TENSOR_NULL),  __func__, __FILE__, __LINE__);
        return;
    }

    if (!a->stride) {
        error_print("%s, in function: %s, file: %s, line: %d\n", common_error_code_to_string(TENSOR_NULL_STRIDE),  __func__, __FILE__, __LINE__);
        Error *error = error_create(TENSOR_NULL_STRIDE, "%s, in function: %s, file: %s, line: %d\n", common_error_code_to_string(TENSOR_NULL_STRIDE),  __func__, __FILE__, __LINE__);
        error_set(error, a, TENSOR_NULL_STRIDE);
        return;
    }

    for (int i = 0; i < a->ndim; i++) {
        printf("%lld ", a->stride[i]);
    }

    printf("\n");
}

// Need to add error checks for iter
static void print_based_on_tensor_type(cpu_iter *iter, dtype dtype) {
    switch (dtype) {
    case DTYPE_I64:
        printf("%d", cpu_iter_get_int(iter));
        break;

    case DTYPE_F64:
        printf("%f", cpu_iter_get_double(iter));
        break;
    case DTYPE_F32:
        printf("%f", cpu_iter_get_float(iter));
    default:
        break;
    }
}

void print(Tensor *tensor) {
    cpu_iter iter;
    dtype dtype = get_tensor_dtype(tensor);
    cpu_iter_init(&iter, tensor, false);

    for (int i = 0; i < iter.ndim; i++) {
        printf("[");
    }

    while (cpu_iter_has_next(&iter)) {
        if (iter.counter > 0) {
            for (int i = iter.ndim - 1; i >= 0; i--) {
                if (iter.coordinates[i] == 0) {
                    printf("[");
                }
                else {
                    break;
                }
            }
        }



           print_based_on_tensor_type(&iter, dtype);


        if (cpu_iter_has_next(&iter)) {

            for (int i = iter.ndim - 1; i >= 0; i--) {
                if (iter.coordinates[i] + 1 == iter.shape[i]) {
                    printf("]");
                }
                else {
                    break;
                }
            }
            printf(" ");
        }

        if (cpu_iter_has_next(&iter)) {
            for (int i = iter.ndim - 1; i >= 0; i--) {
                if (iter.coordinates[i] + 1 == iter.shape[i]) {
                    printf("\n");
                }
                else {
                    break;
                }
            }
        }

        cpu_iter_next(&iter);
    }

    printf("\n");
  
}

void print_grad(Tensor *tensor) {
    cpu_iter iter;
    dtype dtype = get_tensor_dtype(tensor);
    cpu_iter_init_for_grad(&iter, tensor);

    for (int i = 0; i < iter.ndim; i++) {
        printf("[");
    }

    while (cpu_iter_has_next(&iter)) {
        if (iter.counter > 0) {
            for (int i = iter.ndim - 1; i >= 0; i--) {
                if (iter.coordinates[i] == 0) {
                    printf("[");
                }
                else {
                    break;
                }
            }
        }



           print_based_on_tensor_type(&iter, dtype);


        if (cpu_iter_has_next(&iter)) {

            for (int i = iter.ndim - 1; i >= 0; i--) {
                if (iter.coordinates[i] + 1 == iter.shape[i]) {
                    printf("]");
                }
                else {
                    break;
                }
            }
            printf(" ");
        }

        if (cpu_iter_has_next(&iter)) {
            for (int i = iter.ndim - 1; i >= 0; i--) {
                if (iter.coordinates[i] + 1 == iter.shape[i]) {
                    printf("\n");
                }
                else {
                    break;
                }
            }
        }

        cpu_iter_next(&iter);
    }

    printf("\n");
  
}

void print_tensor_numpy_style(Tensor *tensor) {
    if (!tensor || tensor->size == 0) {
        printf("array([])\n");
        return;
    }
    
    printf("array(");
    
    cpu_iter iter;
    int i = cpu_iter_init(&iter, tensor, false);
    if (i != ITER_OK) {
        printf("Error)\n");
        printf("%d", i);
        return;
    }
    
    // Track previous coordinates to know when to add brackets/newlines
    int *prev_coords = calloc(iter.ndim, sizeof(int));
    bool first_element = true;
    
    while (cpu_iter_has_next(&iter)) {
        if (!first_element) {
            // Count how many dimensions changed
            int changed_dims = 0;
            for (int i = 0; i < iter.ndim; i++) {
                if (iter.coordinates[i] != prev_coords[i]) {
                    changed_dims = iter.ndim - i;
                    break;
                }
            }
            
            if (changed_dims > 1) {
                // Close brackets for dimensions that ended
                for (int i = 0; i < changed_dims - 1; i++) {
                    printf("]");
                }
                printf(",\n");
                
                // Indent
                for (int i = 0; i < iter.ndim - changed_dims + 1; i++) {
                    printf(" ");
                }
                
                // Open brackets for new dimensions
                for (int i = 0; i < changed_dims - 1; i++) {
                    printf("[");
                }
            } else if (changed_dims == 1) {
                printf(", ");
            }
        }
        
        // Open brackets at start of each dimension
        if (first_element) {
            for (int i = 0; i < iter.ndim; i++) {
                printf("[");
            }
        } else {
            // Open brackets for dimensions that started
            for (int i = 0; i < iter.ndim; i++) {
                if (iter.coordinates[i] == 0 && prev_coords[i] != 0) {
                    printf("[");
                }
            }
        }
        
        // Print element
        printf("%d", cpu_iter_get_float(&iter));
        // switch (tensor->mdata->dtype) {
        //     case INT:
        //         printf("%d", cpu_iter_get_int(&iter));
        //         break;
        //     case FLOAT:
        //         printf("%g", cpu_iter_get_float(&iter));
        //         break;
        //     case DOUBLE:
        //         printf("%g", cpu_iter_get_double(&iter));
        //         break;
        // }
        
        // Store current coordinates
        memcpy(prev_coords, iter.coordinates, iter.ndim * sizeof(int));
        first_element = false;
        
        cpu_iter_next(&iter);
    }
    
    // Close all brackets
    for (int i = 0; i < iter.ndim; i++) {
        printf("]");
    }
    printf(")\n");
    
    free(prev_coords);
    cpu_iter_free(&iter);
}

/* Tensor Memory Management */

void tensor_free(Tensor *t) {
    if (t) {
        free(t->shape);
        free(t->stride);
        free(t->data);
        free(t->mdata);
        free(t);
    }
}
