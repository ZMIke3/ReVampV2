#include "C:\Programming\Revamp2\Iterator\Iterator.h"
#include <omp.h>

Tensor_Config *tensor_config_make() {
  //  printf("In tensor_config_make\n");
    Tensor_Config *cfg = (Tensor_Config *) malloc(sizeof(Tensor_Config));
    if (cfg) { cfg->nops = 0; return cfg; }
}

void tensor_config_add_operand(Tensor_Config *config, void *data, int64_t ndim, int64_t *stride, int64_t *shape, int size_of_elem, Tensor_FLAG flag) {
  //  printf("In tensor_config_add_operand\n");
    Tensor_Operand_Metadata *op = &config->ops[config->nops++]; // Get a reference to current metadata in config before incrementing
    op->data = (char *) data;
    op->ndim = ndim;
    op->stride = (int64_t *) malloc(ndim * sizeof(int64_t));
    op->shape = (int64_t *) malloc(ndim * sizeof(int64_t));
    memcpy(op->shape, shape, ndim * sizeof(int64_t));
    memcpy(op->stride, stride, ndim * sizeof(int64_t));
    op->size_of_elem = size_of_elem;
    op->flag = flag;
}

void tensor_config_set_mode(Tensor_Config *config, IteratorMode mode) {
    if (!config) return;
    config->mode = mode;
}

bool compute_broadcast_shape(const Tensor_Config *config, int64_t **out_shape, int64_t *out_ndim) {
  //  printf("In compute_broadcast_shape\n");
    int64_t ndim = 0;
    for (int i = 0; i < config->nops; i++) { ndim =  max_64_h(ndim, config->ops[i].ndim); } // Get the biggest dimension

    int64_t stack_shape[_MAX_STACK_NDIM];
    int64_t *shape = (ndim <= _MAX_STACK_NDIM) ? stack_shape : (int64_t *) malloc(ndim * sizeof(int64_t)); // Use stack if ndims is small else, dynamic

    for (int i = 0; i < ndim; i++) shape[i] = 1;

    for (int ops = 0; ops < config->nops; ops++) {
        const Tensor_Operand_Metadata *op = &config->ops[ops];

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

    *out_shape = (ndim <= _MAX_STACK_NDIM) ? (int64_t *) malloc(ndim * sizeof(int64_t)) : shape;

    memcpy(*out_shape, shape, ndim * sizeof(int64_t));

    *out_ndim = ndim;

    if (shape == stack_shape) {} // Shape is on stack

    return true;



}

void coalesce_dims(Iterator *iter) {
   // printf("In coalesce_dims_type_1\n");
    // For op can the current dimension and it's adjacent be coalesced
    for (int dim = 0; dim < iter->ndim; dim++) {
        bool merge = true;
        for (int ops = 0; ops < iter->nops && merge; ops++) {
            int64_t sh0 = iter->shape[dim];
            int64_t sh1 = iter->shape[dim + 1];
            int64_t st0 = iter->strides[ops][dim];
            int64_t st1 = iter->strides[ops][dim + 1];

            // Rule for coalescing, shape[curr] == 1 || shape[curr + 1] = 1 || shape[curr] * stride[curr] == stride[curr + 1]
            if (sh0 != 1 && sh1 != 1 && sh0 * st0 != st1) merge = false;
        }


        // If it can be coalesced
        if (merge) {

            // Shape[curr] *= Shape[curr + 1], we're basically putting the elements in the same array to my understanding
            iter->shape[dim] *= iter->shape[dim + 1];

            // Update the strides for each op
            for (int ops = 0; ops < iter->nops; ops++) {
                iter->strides[ops][dim] = iter->strides[ops][dim + 1];
            }

            // Pull the remaining shapes and strides to the left
            for (int d = dim + 1; d < iter->ndim - 1; ++d) {
                iter->shape[d] = iter->shape[d + 1];
            for (int ops = 0; ops < iter->nops; ops++)
                iter->strides[ops][d] = iter->strides[ops][d + 1];
            }
            // Iter has one less dim
            iter->ndim--; // 
            // Check if the coalesced dim can be coalesced with the next dim
            dim--;
        }
    }
    
}

void calculate_inner_outer_blocks(Iterator *iter) {
 //   printf("In calculate_inner_outer_blocks\n");
    /*
        We're trying to simplify our computation
        say we have a tensor of shape [2, 3, 4]
        by choosing inner and outer blocks we can operate on this tensor
        as if it were a 6 by 4.
        We do this by finding the dimension in which all tensors are contiguous meaning that 
        the elements in that dimension are all adjacent
        we can then set that as an inner block so computation there will be extremely quick
        everythin else is considered outer dimesions, so if we have 6 outer and 4 inner we're basically processing
        6 batches of 4 inner blocks, we've reduced [2, 3, 4] to two loops! with the inner one being faster!
    
    */

    int64_t inner_dim = iter->ndim - 1;

    int64_t smallest_stride = 1;

    for (int op = 0; op < iter->nops; op++) {
        if (iter->strides[op][inner_dim] != 0 && iter->strides[op][inner_dim] < smallest_stride) { // iter->strides[ops][inner_dim] < smallest_stride <-- That's probbaly not needed as stride should not be less than 1
            smallest_stride = iter->strides[op][inner_dim];
        }
    }

    // Now we can look for a fully contiguous dimension (Meaning that that dimension is contiguous for all tensors)

    while (inner_dim >= 0) {
        
        bool _true = true;

        for (int op = 0; op < iter->nops; op++) {
            int64_t st =  iter->strides[op][inner_dim];
            int64_t s = iter->shape[inner_dim];
            if (st == 1) continue; // We're not worried about broadcated dimensions, they only ever access the first index, they're trival
            if (s != smallest_stride) _true = false; // This means that at least one tensor is not contiguous on this dimension
        }

        if (_true) break; // We found a dimension where they're all contiguous
        inner_dim--; // If not we proceed to an outer dimension
    }
    
    if (inner_dim < 0) { inner_dim = iter->ndim - 1; } // If we find no contiguous dimension fall back to the last so we can maintaint two loop status quo

    // Everything to the right of that contiguous dimension becomes inner block
    iter->inner_size = 1;
    for (int dim = inner_dim; dim < iter->ndim; dim++) {
        iter->inner_size *= iter->shape[dim];
    }

    // Everything to the left of that contiguous dimension becomes outer block
    iter->outer_size = 1;
    for (int dim = 0; dim < inner_dim; dim++) {
        iter->outer_size *= iter->shape[dim];
    }

    // Question I have though is, why do we assume everything to the right of the contiguous dimension
    // is contiguous

}

bool tensor_iterator_build(const Tensor_Config *config, Iterator *iter) {
   // printf("In tensor_iterator_build\n");
    // Probbably some error function
    if (config->nops == 0 || config->nops > _MAX_OPERANDS) return false;

    int64_t *shape;
    int64_t ndim;

    if (!compute_broadcast_shape(config, &shape, &ndim)) return false;

    memset(iter, 0, sizeof(* iter));
    iter->nops = config->nops;
    iter->ndim = ndim;
    iter->shape_on_stack = false;
    iter->shape = shape;


    for (int op = 0; op < config->nops; op++) {
        Tensor_Operand_Metadata *c = &config->ops[op];
        for (int dim = 0; dim < ndim; dim++) {
                if (c->shape[dim] == 1) {
                    c->stride[dim] = 0;
                }
            }
        }


    // Here we're just broadcasting the strides, missing dimensions go to 0
    for (int op = 0; op < config->nops; op++) {
        const Tensor_Operand_Metadata *src = &config->ops[op];
        int64_t *stride =  (int64_t *) malloc(ndim * sizeof(int64_t));

        for (int dim = 0; dim < ndim; dim++) {
            stride[dim] = 0;
        }

        int64_t lead = ndim - src->ndim; // So we can get to the dimension where the specific tensors strides begin

        for (int dim = 0; dim < src->ndim; dim++) {
            int64_t st = src->stride[dim];
            int64_t broadcast_dim = lead + dim;
            stride[broadcast_dim] = (iter->shape[broadcast_dim] == 1) ? 0 : st; // Dimensions of 1 have a stride of 0
        }

        iter->strides[op] = stride;
        iter->base_ptrs[op] = src->data;
    }

    coalesce_dims(iter);
    calculate_inner_outer_blocks(iter);
   // printf("Leaving tensor_iterator_build\n");
    return true;
}

void run_block(Tensor_Iterator_Kernel kernal, char **ptrs, const int64_t *inner_strides, int64_t inner_size, int nops) {
 //   printf("In run_block\n");
    kernal(ptrs, inner_strides, inner_size);
}

int64_t compute_advance_and_coordinates(Iterator *iter, int64_t **coordinates, int64_t op, int idx) {
        int64_t *coordinatess = calloc(iter->ndim, sizeof(int64_t));
        int64_t ptr = 0;
        for (int dim = iter->ndim - 1; dim >= 0; dim--) {
            int64_t coord = idx % iter->shape[dim]; // Get the coordinate at that dimension
            coordinatess[dim] = coord;
            ptr += coord * iter->strides[op][dim]; // Advance the ptr by the strides in that dimension
            idx /= iter->shape[dim];
        }

        *coordinates = coordinatess;
        return ptr;

}

void tensor_iterator_serial_for_each(Iterator *iter, Tensor_Iterator_Kernel kernal) {
  //  printf("In tensor_iterator_serial_for_each\n");
    int nops = iter->nops;
    int64_t inner = iter->inner_size;
    int64_t blocks = iter->outer_size;

    int64_t *inner_strides = (int64_t *)malloc(nops * sizeof(int64_t));

    for (int op = 0; op < nops; op++) {
        inner_strides[op] = iter->strides[op][iter->ndim-1];
    }

    char *ptrs[_MAX_OPERANDS];
    memcpy(ptrs, iter->base_ptrs, nops * sizeof(char *));
    int64_t *coordinates;
    for (int64_t block = 0; block < blocks; block++) {
        run_block(kernal, ptrs, inner_strides, inner, nops);

        // Advancing the blocks
        for (int op = 0; op < nops; op++) {
            int64_t idx =  block + 1;
            char *ptr = iter->base_ptrs[op];

            //ptr += compute_advance_and_coordinates(iter, &coordinates, op, idx);
            for (int dim = iter->ndim - 2; dim >= 0; dim--) {
                int64_t coord = idx % iter->shape[dim]; // Get the coordinate at that dimension
                ptr += coord * iter->strides[op][dim]; // Advance the ptr by the strides in that dimension
                idx /= iter->shape[dim];
            }
            ptrs[op] = ptr;
        }
    }
}

void ti_parallel_for_each(Iterator *it, Tensor_Iterator_Kernel f, int nthreads) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
    for (int64_t blk = 0; blk < it->outer_size; ++blk) {
        char *ptrs[_MAX_OPERANDS];
        for (int op = 0; op < it->nops; ++op) {
            char *p = it->base_ptrs[op];
            int64_t idx = blk;
            for (int d = it->ndim - 2; d >= 0; --d) {
                int64_t coord = idx % it->shape[d];
                p += coord * it->strides[op][d];
                idx /= it->shape[d];
            }
            ptrs[op] = p;
        }
        int64_t inner_stride[_MAX_OPERANDS];
        for (int op = 0; op < it->nops; ++op)
            inner_stride[op] = it->strides[op][it->ndim - 1];

        run_block(f, ptrs, inner_stride, it->inner_size, it->nops);
    }
}


void tensor_iterator_free(Iterator *it) {
    for (int op = 0; op < it->nops; ++op)
        free(it->strides[op]);
    free(it->shape);
}

bool tensor_broadcast_output(Tensor *a, Tensor *b, Tensor *out) {
   // printf("In tensor_broadcast_output\n");
    
    int64_t out_dim = a->ndim > b->ndim ? a->ndim : b->ndim;
    
    int64_t *a_padded = calloc(out_dim, sizeof(int64_t));
    int64_t *b_padded = calloc(out_dim, sizeof(int64_t));
    
    memcpy(a_padded + (out_dim - a->ndim), a->shape, a->ndim * sizeof(int64_t));
    memcpy(b_padded + (out_dim - b->ndim), b->shape, b->ndim * sizeof(int64_t));
    
    for (int i = 0; i < out_dim - a->ndim; i++) a_padded[i] = 1;
    for (int i = 0; i < out_dim - b->ndim; i++) b_padded[i] = 1;
    
    int64_t *shape = calloc(out_dim, sizeof(int64_t));
    
    for (int i = 0; i < out_dim; i++) {
        if (a_padded[i] == b_padded[i] || a_padded[i] == 1 || b_padded[i] == 1) {
            shape[i] = max_64_h(a_padded[i], b_padded[i]);
        } else {
            free(a_padded);
            free(b_padded);
            free(shape);
            return false;
        }
    }
    
    if (out_dim != out->ndim) {
        out->ndim = out_dim;
        free(out->shape);
        free(out->stride);
        out->shape = (int64_t *) malloc(out->ndim * sizeof(int64_t));
        assert(out->shape != NULL);
    }
    
    memcpy(out->shape, shape, out->ndim * sizeof(int64_t));
    
    int64_t new_size = tensor_size_from_shape(shape, out_dim);
    if (new_size > out->size) {
        out->size = new_size;
        if (out->data) {
            free(out->data);
        }
        out->data = (int64_t *) malloc(new_size * sizeof(int64_t));
        assert((int64_t *) out->data != NULL);
    }
    
    if (out->stride) {
        free(out->stride);
    }
    out->stride = calc_stride(out->ndim, out->shape, out->mdata->type_info->dtype_size);
    assert(out->stride != NULL);
    
    free(a_padded);
    free(b_padded);
    free(shape);
    
    return true;
}

Iterator *tensor_iterator_make() {
    Iterator *iter = (Iterator *)malloc(sizeof(Iterator));
    if (!iter) { 
        printf("Failed to allocate iterator\n"); 
        return NULL; 
    }

    iter->ReducCrate = (ReductionCrate *)malloc(sizeof(ReductionCrate));
    if (!iter->ReducCrate) { 
        printf("Failed to allocate ReductionCrate\n");
        free(iter);
        return NULL; 
    }

    return iter;
}

/* Reductions */

void tensor_config_set_reduction(Tensor_Config *config, int64_t *reduce_dims, int64_t reduce_ndims, bool keepdim) {
    assert(config != NULL);
    config->reduce_dims = reduce_dims;
    config->reduce_ndims = reduce_ndims;
    config->keepdim = keepdim;
}

void compute_reduction_shape_ndim(Tensor *a, int64_t *dims, int ndims, bool keepdim, bool **_reduce, int64_t **out_shape, int64_t *out_ndims) {

    // Mark dimensions to reduce
    bool *reduce = calloc(a->ndim, sizeof(bool));
    for (int dim = 0; dim < ndims; dim++) {
        int64_t d = dims[dim];
        if (d < 0) d += a->ndim;
        reduce[d] = true;
    }
    
    int64_t *output_shape = calloc(a->ndim, sizeof(int64_t));
    int64_t out_ndim = 0;
    for (int dim = 0; dim < a->ndim; dim++) {
        if (keepdim) {
            output_shape[out_ndim++] = reduce[dim] ? 1 : a->shape[dim];
        } else if (!reduce[dim]) {
            output_shape[out_ndim++] = a->shape[dim];
        }
    }

    *_reduce = reduce;
    *out_shape = output_shape;
    *out_ndims = out_ndim;
}

bool tensor_reduction_iterator_build(Tensor_Config *config, Iterator *iter) {

    if (config->nops == 0 || config->nops > _MAX_OPERANDS) { 
        return false; 
    }

    // Tensor_Operand_Metadata *src0 = &config->ops[0];
    // Tensor_Operand_Metadata *src1 = &config->ops[1];

    if (config->nops > 2) {
        printf("Can only support 2 tensors for reductions\n");
        return false;
    }

    // Find input and output indices
    for (int i = 0; i < config->nops; i++) {
        Tensor_Operand_Metadata *src = &config->ops[i];
        
        if (src->flag == TENSOR_INPUT) {
            iter->ReducCrate->input_index = i;
        } else if (src->flag == TENSOR_REDUCTION_OUTPUT) {
            iter->ReducCrate->output_index = i;
        } else {
            printf("Failed to retrieve index of Input Tensor and output Tensor\n");
            return false;
        }
    }

    // Initialize iterator (but preserve ReducCrate pointer)
//     ReductionCrate *temp_crate = iter->ReducCrate;
//    // memset(iter, 0, sizeof(*iter));
//     iter->ReducCrate = temp_crate;

    // Initialize ReductionCrate
    //memset(iter->ReducCrate, 0, sizeof(*iter->ReducCrate));
    iter->ReducCrate->input_index = iter->ReducCrate->input_index;
    iter->ReducCrate->output_index = iter->ReducCrate->output_index;

    if (config->set_out_special_value) {
        iter->ReducCrate->out_special_value = config->out_special_value;
    }

    // Set up iterator basic fields
    iter->nops = config->nops;
    iter->ndim = config->ops[iter->ReducCrate->input_index].ndim;
    iter->shape_on_stack = false;
    iter->mode = config->mode;
    
    // Set up ReductionCrate fields
    iter->ReducCrate->keepdim = config->keepdim;
    iter->ReducCrate->_reduce = config->_reduce;
    iter->ReducCrate->ReductionDims = config->reduce_dims;
    iter->ReducCrate->Reduction_ndims = config->reduce_ndims;
    iter->ReducCrate->output_shape = config->ops[iter->ReducCrate->output_index].shape;
    iter->ReducCrate->output_ndim = config->ops[iter->ReducCrate->output_index].ndim;
    iter->ReducCrate->counter = 0;
    
    // Allocate shape and copy
    iter->shape = malloc(iter->ndim * sizeof(int64_t));
    if (!iter->shape) {
        return false;
    }
    memcpy(iter->shape, config->ops[iter->ReducCrate->input_index].shape, iter->ndim * sizeof(int64_t));

    // Allocate ReductionCrate arrays
    iter->ReducCrate->coordinates = calloc(iter->ndim, sizeof(int64_t));
    iter->ReducCrate->backstrides = calloc(iter->ndim, sizeof(int64_t));
    if (!iter->ReducCrate->coordinates || !iter->ReducCrate->backstrides) {
        return false;
    }
    
    iter->ReducCrate->input_size = tensor_size_from_shape(config->ops[iter->ReducCrate->input_index].shape, config->ops[iter->ReducCrate->input_index].ndim);
    
    // Set up base pointers and strides
    for (int op = 0; op < iter->nops; op++) {
        Tensor_Operand_Metadata *src = &config->ops[op];
        iter->base_ptrs[op] = src->data;
        iter->strides[op] = malloc(iter->ndim * sizeof(int64_t));
        if (!iter->strides[op]) {
            return false;
        }
        memcpy(iter->strides[op], src->stride, iter->ndim * sizeof(int64_t));
    }
    
    // Calculate backstrides for coordinate wrapping
    for (int i = 0; i < iter->ndim; i++) {
        iter->ReducCrate->backstrides[i] = iter->strides[iter->ReducCrate->input_index][i] * (iter->shape[i] - 1);
    }

    return true;
}

char *advance_coordinates(Iterator *iter, char **data_ptr, int op_idx) {
    iter->ReducCrate->counter++;
    for (int i = iter->ndim - 1; i >= 0; i--) {
        if (iter->ReducCrate->coordinates[i] + 1 < iter->shape[i]) {
            iter->ReducCrate->coordinates[i]++;
            *data_ptr += iter->strides[op_idx][i];
            break;
        } else {
            *data_ptr -= iter->ReducCrate->backstrides[i];
            iter->ReducCrate->coordinates[i] = 0;
        }
    }

    return *data_ptr;
}

int64_t map_input_coordinates_to_output_idx(Iterator *iter) {
    int64_t out_flat_idx = 0;
    int out_coord_idx = 0;
    
    for (int i = 0; i < iter->ndim; i++) {
        if (iter->ReducCrate->keepdim) {
            // Keep all dimensions, but reduced dims use coord = 0
            int64_t coord = iter->ReducCrate->_reduce[i] ? 0 : iter->ReducCrate->coordinates[i];
            out_flat_idx += coord * (iter->strides[iter->ReducCrate->output_index][i] / sizeof(int64_t));
        } else if (!iter->ReducCrate->_reduce[i]) {
            // Only include non-reduced dimensions
            out_flat_idx += iter->ReducCrate->coordinates[i] * 
                (iter->strides[iter->ReducCrate->output_index][out_coord_idx] / sizeof(int64_t));
            out_coord_idx++;
        }
    }
    
    return out_flat_idx;
}

void tensor_reduction_iterator_serial_for_each(Iterator *iter, Tensor_Reduction_Kernel kernel) {
    int64_t total = iter->ReducCrate->input_size;

    char *input_ptr = iter->base_ptrs[iter->ReducCrate->input_index];

    char *output_ptr = iter->base_ptrs[iter->ReducCrate->output_index];

    int64_t output_size = tensor_size_from_shape(iter->ReducCrate->output_shape, iter->ReducCrate->output_ndim);

    if (iter->ReducCrate->set_out_special_value) {

        int64_t *out_ptr = (int64_t*)iter->base_ptrs[iter->ReducCrate->output_index];

        for (int64_t i = 0; i < output_size; i++) {
            out_ptr[i] = iter->ReducCrate->out_special_value;
        }

    } else {
        memset(iter->base_ptrs[iter->ReducCrate->output_index], 0, output_size * sizeof(int64_t));
    }

    memset(iter->ReducCrate->coordinates, 0, iter->ndim * sizeof(int64_t));

    int64_t OUTER = 20;
    int64_t INNER = total / OUTER;
    int64_t remainder = total % OUTER;

    int64_t *out_indexes = calloc(INNER, sizeof(int64_t));
    int64_t *input_values = calloc(INNER, sizeof(int64_t));

    for (int64_t i = 0; i < OUTER; i++) {

        int64_t current_inner = (i == OUTER - 1) ? INNER + remainder : INNER;

        for (int64_t j = 0; j < current_inner; j++) {

            out_indexes[j] = map_input_coordinates_to_output_idx(iter);
            input_values[j] = *(int64_t *) input_ptr;
            input_ptr = advance_coordinates(iter, &input_ptr, iter->ReducCrate->input_index);

        }

        kernel(iter->base_ptrs, input_values, out_indexes, INNER, iter->ReducCrate->input_index, iter->ReducCrate->output_index);
        
    }
}