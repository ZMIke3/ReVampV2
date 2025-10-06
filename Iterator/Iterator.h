#pragma once
#include "C:\Programming\Revamp2\Tensor\Tensor.h"
#include "C:\Programming\Revamp2\Utils\utills.h"

// Iterator rewrite
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#define _MAX_OPERANDS 8
#define _MAX_STACK_NDIM 8
#define MAX_DIMS 45

typedef enum {
    TENSOR_INPUT,
    TENSOR_OUTPUT,
    TENSOR_REDUCTION_OUTPUT
}Tensor_FLAG;

typedef enum {
    ITERATOR_REDUCTION,
    ITERATOR_ELEMENT_WISE
}IteratorMode;


typedef struct {
    char *data;
    int64_t *stride;
    int64_t *shape;
    int64_t ndim;
    size_t size_of_elem;
    Tensor_FLAG flag;
}Tensor_Operand_Metadata;

typedef struct {
    Tensor_Operand_Metadata ops[_MAX_OPERANDS];
    int nops;
    // For Reductions
    IteratorMode mode;
    int64_t *reduce_dims;   
    int64_t reduce_ndims;
    bool *_reduce;
    bool keepdim;
    bool set_out_special_value;
    int64_t out_special_value; // <- Should really be a union of all possible tensor types     
}Tensor_Config;

typedef struct {
    bool *_reduce;
    int input_index;
    int output_index;
    int64_t *coordinates; // For input
    int64_t *backstrides; // For input
    int64_t *ReductionDims;     
    int64_t Reduction_ndims;
    int64_t *output_shape;  
    int64_t output_ndim;
    int64_t input_size;
    int64_t counter;
    bool keepdim;
    bool set_out_special_value;
    int64_t out_special_value; // <- Should really be a union of all possible tensor types     
}ReductionCrate;


typedef struct {
    int nops;
    int ndim;
    int64_t *shape;
    char *base_ptrs[_MAX_OPERANDS];
    int64_t *strides[_MAX_OPERANDS];
    int64_t inner_size;
    int64_t outer_size;
    bool shape_on_stack;
    size_t size_of_elem;
    IteratorMode mode;

    /* Reductions */
    ReductionCrate *ReducCrate;
}Iterator;

typedef void (*Tensor_Iterator_Kernel) (char **ptrs, const int64_t *inner_stride, int64_t loop_len);
typedef void (*Tensor_Reduction_Kernel) (char **ptrs,  void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);

Tensor_Config *tensor_config_make();
void tensor_config_add_operand(Tensor_Config *config, void *data, int64_t ndim, int64_t *stride, int64_t *shape, int size_of_elem, Tensor_FLAG flag);
void tensor_config_set_mode(Tensor_Config *config, IteratorMode mode);
void tensor_iterator_free(Iterator *iter);
bool compute_broadcast_shape(const Tensor_Config *config, int64_t **out_shape, int64_t *out_ndim);
void tensor_config_add_operand(Tensor_Config *config, void *data, int64_t ndim, int64_t *stride, int64_t *shape, int size_of_elem, Tensor_FLAG flag);
void coalesce_dims(Iterator *iter);
void calculate_inner_outer_blocks(Iterator *iter);
bool tensor_iterator_build(const Tensor_Config *config, Iterator *iter);
void tensor_iterator_serial_for_each(Iterator *iter, Tensor_Iterator_Kernel kernal);
void ti_parallel_for_each(Iterator *it, Tensor_Iterator_Kernel f, int nthreads);
void tensor_iterator_free(Iterator *iter);
bool tensor_broadcast_output(Tensor *a, Tensor *b, Tensor *out);
Iterator *tensor_iterator_make();

/* Reductions */
void tensor_config_set_reduction(Tensor_Config *config, int64_t *reduce_dims, int64_t reduce_ndims, bool keepdim);
int64_t map_input_coordinates_to_output_idx(Iterator *iter);
char *advance_coordinates(Iterator *iter, char **data_ptr, int op_idx);
bool tensor_reduction_iterator_build(Tensor_Config *config, Iterator *iter);
void compute_reduction_shape_ndim(Tensor *a, int64_t *dims, int ndims, bool keedim, bool **_reduce, int64_t **out_shape, int64_t *out_ndims);
void tensor_reduction_iterator_serial_for_each(Iterator *iter, Tensor_Reduction_Kernel kernel);
void advance_input_pointer_and_coordinates(Iterator *iter, char **input_ptr);

