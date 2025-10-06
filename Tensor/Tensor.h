#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdbool.h>
#include <errno.h>
#include <assert.h>
#include <stdint.h>
#include "C:\Programming\Revamp2\Utils\utills.h"
#include "C:\Programming\Revamp2\AutoGrad.h"
#include "C:\Programming\Revamp2\Error.h"


typedef enum{
    CPU,
    SIMD,
    GPU,
    BACKEND_COUNT
}Backend;

typedef enum {
    DTYPE_I8,
    DTYPE_I16, 
    DTYPE_I32,
    DTYPE_I64,
    DTYPE_U8,
    DTYPE_U16,
    DTYPE_U32,
    DTYPE_U64,
    DTYPE_F16,
    DTYPE_F32,
    DTYPE_F64,
    DTYPE_BOOL,
    DTYPE_NONE,
    DTYPE_COUNT
} dtype;


typedef struct Tensor Tensor;

typedef struct Node Node;

typedef struct Error Error;


typedef struct {
    dtype dtype;
    size_t dtype_size;
    const char *name;
    bool is_floating_point;
    bool is_signed;
    bool is_complex;
} TypeInfo;

typedef struct Metadata {
    bool owns_data;
    bool dtype_set;
    bool data_init;
    bool requires_grad;
    TypeInfo *type_info;
    Backend backend;
}Metadata;

typedef struct Tensor {
    int64_t size;
    int64_t ndim; 
    int64_t *shape;
    int64_t *stride;
    Metadata *mdata;
    Node *node;
    void *grad;
    void *data;
    bool is_leaf;
    bool requires_grad;
    Error *error;
} Tensor;




/* Tensor Types*/
TypeInfo *get_tensor_type_info(Tensor *a);
TypeInfo *set_tensor_type_info(dtype dtype);
TypeInfo *get_type_info(dtype dtype);
dtype promote_to_type(dtype type_1, dtype type_2);
dtype get_tensor_dtype(Tensor *a);
size_t get_dtype_size(dtype dtype);
void print_tensor_dtype(dtype dtype);

/* Tensor Utility */
Tensor *new_blank_tensor();
Tensor *poisoned_blank_tensor(int64_t ndim, int64_t *shape, dtype dtype, Backend backend);
Tensor *tensor_like(Tensor *a);
Tensor *tensor_like_with_type(Tensor *a, dtype dtype);
const int64_t *get_tensor_shape(Tensor *a);
void set_tensor_shape_and_ndim_broadcast(Tensor *a, int64_t *shape, int64_t ndim, bool *broadcast_dim);
const int64_t *get_tensor_stride(Tensor *a);
const int64_t get_tensor_ndim(Tensor *a);
const Backend get_tensor_backend(Tensor *a);
void print_tensor_shape(Tensor *a);
void print_tensor_stride(Tensor *a);
void print(Tensor *tensor);
void print_grad(Tensor *tensor);

/* Tensor Utility Creation Functions */
Tensor *new_int8_tensor(int64_t ndim, int64_t *shape, int8_t *data, Backend backend);
Tensor *new_int16_tensor(int64_t ndim, int64_t *shape, int16_t *data, Backend backend);
Tensor *new_int32_tensor(int64_t ndim, int64_t *shape, int32_t *data, Backend backend);
Tensor *new_int64_tensor(int64_t ndim, int64_t *shape, int64_t *data, Backend backend);
Tensor *new_uint8_tensor(int64_t ndim, int64_t *shape, uint8_t *data, Backend backend);
Tensor *new_uint16_tensor(int64_t ndim, int64_t *shape, uint16_t *data, Backend backend);
Tensor *new_uint32_tensor(int64_t ndim, int64_t *shape, uint32_t *data, Backend backend);
Tensor *new_uint64_tensor(int64_t ndim, int64_t *shape, uint64_t *data, Backend backend);
Tensor *new_float32_tensor(int64_t ndim, int64_t *shape, float *data, Backend backend);
Tensor *new_float64_tensor(int64_t ndim, int64_t *shape, double *data, Backend backend);
Tensor *new_bool_tensor(int64_t ndim, int64_t *shape, bool *data, Backend backend);
Tensor *scalar_to_tensor(void *value, dtype dtype, Backend backend);
Tensor *flat_array_to_tensor(int64_t size, void *value, dtype dtype, Backend backend);

/* Tensor Memory Management */
void tensor_free(Tensor *t);

/*Miscallenous <- Can't spell*/
int size_from_shape(int64_t* shape, int64_t ndim);
void print_tensor_numpy_style(Tensor *tensor);

#endif // TENSOR_H
