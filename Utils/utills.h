#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include "C:\Programming\Revamp2\Error.h"
// #include "C:\Programming\Revamp2\Memory.h"


#define SET_TENSOR_VALUE_TYPE_AGNOSTIC(type, tensor, idx, value) \
    do { \
        ((type *)(tensor)->data)[idx] = (value); \
    } while(0)

#define SET_TENSOR_GRAD_VALUE_TYPE_AGNOSTIC(type, tensor, idx, value) \
    do { \
        ((type *)(tensor)->grad)[idx] = (value); \
    } while(0)


#define SET_TENSOR_VALUE(dtype, tensor, idx, value) \
    do { \
        switch (dtype) { \
            case DTYPE_I8: \
                SET_TENSOR_VALUE_TYPE_AGNOSTIC(int8_t, tensor, idx, value); \
                break; \
            case DTYPE_I16: \
                SET_TENSOR_VALUE_TYPE_AGNOSTIC(int16_t, tensor, idx, value); \
                break; \
            case DTYPE_I32: \
                SET_TENSOR_VALUE_TYPE_AGNOSTIC(int32_t, tensor, idx, value); \
                break; \
            case DTYPE_I64: \
                SET_TENSOR_VALUE_TYPE_AGNOSTIC(int64_t, tensor, idx, value); \
                break; \
            case DTYPE_U8: \
                SET_TENSOR_VALUE_TYPE_AGNOSTIC(uint8_t, tensor, idx, value); \
                break; \
            case DTYPE_U16: \
                SET_TENSOR_VALUE_TYPE_AGNOSTIC(uint16_t, tensor, idx, value); \
                break; \
            case DTYPE_U32: \
                SET_TENSOR_VALUE_TYPE_AGNOSTIC(uint32_t, tensor, idx, value); \
                break; \
            case DTYPE_U64: \
                SET_TENSOR_VALUE_TYPE_AGNOSTIC(uint64_t, tensor, idx, value); \
                break; \
            case DTYPE_F16: \
                SET_TENSOR_VALUE_TYPE_AGNOSTIC(uint16_t, tensor, idx, value); \
                break; \
            case DTYPE_F32: \
                SET_TENSOR_VALUE_TYPE_AGNOSTIC(float, tensor, idx, value); \
                break; \
            case DTYPE_F64: \
                SET_TENSOR_VALUE_TYPE_AGNOSTIC(double, tensor, idx, value); \
                break; \
            case DTYPE_BOOL: \
                SET_TENSOR_VALUE_TYPE_AGNOSTIC(bool, tensor, idx, value); \
                break; \
            case DTYPE_COUNT: \
                printf("Unsupported type passed in as type for value\n"); \
                break;  \
        default: \
                printf("Unsupported type passed in as type for value\n"); \
            break; \
        } \
            \
    } while(0)

#define SET_TENSOR_GRAD_VALUE(dtype, tensor, idx, value) \
    do { \
        switch (dtype) { \
            case DTYPE_I8: \
                SET_TENSOR_GRAD_VALUE_TYPE_AGNOSTIC(int8_t, tensor, idx, value); \
                break; \
            case DTYPE_I16: \
                SET_TENSOR_GRAD_VALUE_TYPE_AGNOSTIC(int16_t, tensor, idx, value); \
                break; \
            case DTYPE_I32: \
                SET_TENSOR_GRAD_VALUE_TYPE_AGNOSTIC(int32_t, tensor, idx, value); \
                break; \
            case DTYPE_I64: \
                SET_TENSOR_GRAD_VALUE_TYPE_AGNOSTIC(int64_t, tensor, idx, value); \
                break; \
            case DTYPE_U8: \
                SET_TENSOR_GRAD_VALUE_TYPE_AGNOSTIC(uint8_t, tensor, idx, value); \
                break; \
            case DTYPE_U16: \
                SET_TENSOR_GRAD_VALUE_TYPE_AGNOSTIC(uint16_t, tensor, idx, value); \
                break; \
            case DTYPE_U32: \
                SET_TENSOR_GRAD_VALUE_TYPE_AGNOSTIC(uint32_t, tensor, idx, value); \
                break; \
            case DTYPE_U64: \
                SET_TENSOR_GRAD_VALUE_TYPE_AGNOSTIC(uint64_t, tensor, idx, value); \
                break; \
            case DTYPE_F16: \
                SET_TENSOR_GRAD_VALUE_TYPE_AGNOSTIC(uint16_t, tensor, idx, value); \
                break; \
            case DTYPE_F32: \
                SET_TENSOR_GRAD_VALUE_TYPE_AGNOSTIC(float, tensor, idx, value); \
                break; \
            case DTYPE_F64: \
                SET_TENSOR_GRAD_VALUE_TYPE_AGNOSTIC(double, tensor, idx, value); \
                break; \
            case DTYPE_BOOL: \
                SET_TENSOR_GRAD_VALUE_TYPE_AGNOSTIC(bool, tensor, idx, value); \
                break; \
            case DTYPE_COUNT: \
                printf("Unsupported type passed in as type for value\n"); \
                break;  \
        default: \
                printf("Unsupported type passed in as type for value\n"); \
            break; \
        } \
            \
    } while(0)

int64_t max_64_h(int64_t a, int64_t b);
int64_t tensor_arange_helper(int64_t start, int64_t end, int64_t step);
int64_t tensor_size_from_shape(int64_t *shape, int64_t ndim);
int64_t *calc_stride(int64_t ndim, int64_t *shape, size_t dtype_size);

void copy_array_to_tensor_buffer(void *tensor, void *array);