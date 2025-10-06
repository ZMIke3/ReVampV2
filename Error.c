#include "C:\Programming\Revamp2\Error.h"


Error *error_create(ErrorCode code, const char *msg, ...) {
    Error *err = (Error *)malloc(sizeof(Error));

    if (!err) {
        printf("Failed to create Error struct in function: %s. Msg was: %s\n", __func__, msg);
        return NULL;
    }

    va_list args;
    va_start(args, msg);

    int msg_len = vsnprintf(NULL, 0, msg, args) + 1; 
    
    err->msg = (char *)malloc(msg_len);

    if (!err->msg) {
        printf("Failed to allocate memory for error message in function: %s\n", __func__);
        free(err);
        return NULL;
    }

    vsnprintf(err->msg, msg_len, msg, args);
    va_end(args);

    err->code = code;

    return err;
}

void error_tensor_print(Tensor *tensor) {

    if (!tensor) {
        printf("Input is null in function: %s\n", __func__);
    }

    if (!tensor->error) {
        printf("Input error field is null in function: %s\n", __func__);
    }

    printf("%s\n", tensor->error->msg);

}


void error_print(const char *msg, ...) {
    va_list args;
    va_start(args, msg);
    vprintf(msg, args);
    va_end(args);
}

void error_set(Error *error, void *input, ERROR_SET_INPUT_TYPE_CODE code) {

    switch (code) {

        case TENSOR:
            Tensor *tensor = (Tensor *) input;
            tensor->error = error;
        break;
    
    default:
        break;
    }
}


char *error_code_to_string(ErrorCode code) {

    switch (code) {
        case TENSOR_NULL:
            return "TENSOR_NULL_ERROR:";
        break;

        case TENSOR_NULL_SHAPE:
            return "TENSOR_NULL_SHAPE_ERROR:";
        break;

        case TENSOR_NULL_STRIDE:
            return "TENSOR_NULL_STRIDE_ERROR:";
        break;        

        case TENSOR_NULL_METADATA:
            return "TENSOR_NULL_METADATA_ERROR:";
        break;   

        case TENSOR_DATA_FROM_C_ARRAY_NULL:
            return "TENSOR_DATA_FROM_C_ARRAY_NULL_ERROR:";
        break;     

        case TENSOR_SHAPE_FROM_C_ARRAY_NULL:
            return "TENSOR_SHAPE_FROM_C_ARRAY_NULL_ERROR:";
        break;     

        case DTYPE_INPUT_UNSUPPORTED:
            return "DTYPE_INPUT_UNSUPPORTED_ERROR:";
        break;

        case TENSOR_FAILED_TO_BROADCAST_INPUTS_WITH_OUTPUT:
            return "TENSOR_FAILED_TO_BROADCAST_INPUTS_WITH_OUTPUT_ERROR:";
        break;

        case FAILED_TO_CREATE_TENSOR_CONFIG:
            return "FAILED_TO_CREATE_TENSOR_CONFIG_ERROR:";
        break;

        case FAILED_TO_CREATE_TENSOR_ITERATOR:
            return "FAILED_TO_CREATE_TENSOR_ITERATOR_ERROR:";
        break;

        case FAILED_TO_RETRIEVE_KERNEL:
            return "FAILED_TO_RETRIEVE_KERNEL_ERROR:";
        break;

        case INVALID_ARGUMENTS:
            return "INVALID_ARGUMENTS_ERROR:";
        break;

        case FAILED_TO_ALLOCATE_MEMORY:
            return "FAILED_TO_ALLOCATE_MEMORY_ERROR:";
        break;


    default:
        break;
    }
}


char *common_error_code_to_string(ErrorCode code) {

    switch (code) {
        case TENSOR_NULL:
            return "TENSOR_NULL_ERROR: Tensor is not allocated";
        break;
        
        case TENSOR_NULL_SHAPE:
            return "TENSOR_NULL_SHAPE_ERROR: Tensor's shape is not allocated";
        break;

        case TENSOR_NULL_STRIDE:
            return "TENSOR_NULL_STRIDE_ERROR: Tensor's stride is not allocated";
        break;       

        case TENSOR_NULL_METADATA:
            return "TENSOR_NULL_METADATA_ERROR: Tensor metadata is not allocated";
        break;   

        case TENSOR_DATA_FROM_C_ARRAY_NULL:
            return "TENSOR_DATA_FROM_C_ARRAY_NULL_ERROR: The C array as input for tensor's data is not allocated";
        break;    

        case TENSOR_SHAPE_FROM_C_ARRAY_NULL:
            return "TENSOR_SHAPE_FROM_C_ARRAY_NULL_ERROR: The C array as input for tensor's shape is not allocated";
        break;    

        case TENSOR_FAILED_TO_BROADCAST_INPUTS_WITH_OUTPUT:
            return "TENSOR_FAILED_TO_BROADCAST_INPUTS_WITH_OUTPUT_ERROR: The input tensors could not be broadcasted to resolve output tensor shape";
        break;

        case DTYPE_INPUT_UNSUPPORTED:
            return "DTYPE_INPUT_UNSUPPORTED_ERROR: The input data type is not supported";
        break;


        case FAILED_TO_CREATE_TENSOR_CONFIG:
            return "FAILED_TO_CREATE_TENSOR_CONFIG_ERROR: Tensor config is null";
        break;

        case FAILED_TO_CREATE_TENSOR_ITERATOR:
            return "FAILED_TO_CREATE_TENSOR_ITERATOR_ERROR: Tensor iterator is null";
        break;


        case FAILED_TO_BUILD_TENSOR_ITERATOR:
            return "FAILED_TO_BUILD_TENSOR_ITERATOR_ERROR: Tensor iterator failed to build. It seems that it was allocated though";
        break;

        case FAILED_TO_RETRIEVE_KERNEL:
            return "FAILED_TO_RETRIEVE_KERNEL_ERROR: It seems that the requested kernel does not exist in kernel table";
        break;

        case COULD_NOT_ACQUIRE_KERNEL:
            return "COULD_NOT_ACQUIRE_KERNEL_ERROR: The kernel for the corresponding dtypes or operation might not exist";
        break;

        case INVALID_ARGUMENTS:
            return "INVALID_ARGUMENTS_ERROR: The arguments are inappropriate as inputs for the function";
        break;

        case FAILED_TO_ALLOCATE_MEMORY:
            return "FAILED_TO_ALLOCATE_MEMORY_ERROR: Failed to allocate memory for value";
        break;

    default:
        break;
    }
}