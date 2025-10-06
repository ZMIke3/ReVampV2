#ifndef ERROR_H
#define ERROR_H


#include <stdlib.h>
#include "C:\Programming\Revamp2\Tensor\Tensor.h"
#include <stdarg.h>
#include <string.h>


typedef enum {
    TENSOR_OK,
    TENSOR_NOT_ALLOCATED,
    TENSOR_NULL,
    TENSOR_NULL_SHAPE,
    TENSOR_NULL_STRIDE,
    TENSOR_NULL_METADATA,
    TENSOR_DATA_FROM_C_ARRAY_NULL,
    TENSOR_SHAPE_FROM_C_ARRAY_NULL,
    TENSOR_FAILED_TO_BROADCAST_INPUTS_WITH_OUTPUT,
    DTYPE_INPUT_UNSUPPORTED,
    FAILED_TO_CREATE_TENSOR_CONFIG,
    FAILED_TO_CREATE_TENSOR_ITERATOR,
    FAILED_TO_BUILD_TENSOR_ITERATOR,
    FAILED_TO_RETRIEVE_KERNEL,
    COULD_NOT_ACQUIRE_KERNEL,
    INVALID_ARGUMENTS,
    FAILED_TO_ALLOCATE_MEMORY,
} ErrorCode;

typedef enum {
    TENSOR,
} ERROR_SET_INPUT_TYPE_CODE;

typedef enum {
    CREATION,
    ADD,
    SUB,
    MULTIPLY,
    DIVIDE
} Function;

typedef struct Tensor Tensor;


typedef struct Error{
    char *msg;
    ErrorCode code;

}Error;


Error *error_create(ErrorCode code, const char *msg, ...);
void error_set(Error *error, void *input, ERROR_SET_INPUT_TYPE_CODE code);
void error_tensor_print(Tensor *tensor);
void error_print(const char *msg, ...);
char *error_code_to_string(ErrorCode code);
char *common_error_code_to_string(ErrorCode code);

#endif // ERROR_H
