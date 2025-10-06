#ifndef CPU_ITERATOR_H
#define CPU_ITERATOR_H

#include "C:\Programming\Revamp2\Tensor\Tensor.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

typedef enum {
    ITER_OK,
    ITER_ERROR_OUT_OF_MEMORY,
    ITER_ERROR_INVALID_TENSOR,
    ITER_ERROR_BROADCAST_INCOMPATIBLE
} IterError;

typedef struct {
    Tensor *tensor_object;
    int64_t *stride;
    int64_t *backstrides;
    int64_t *coordinates;
    int64_t *shape;
    int64_t self_tensor_size;
    int64_t counter;
    int64_t ndim;    
    char *data_ptr;
} cpu_iter;

// Initialize iterator
IterError cpu_iter_init(cpu_iter *iter, Tensor *tensor, bool grad);
IterError cpu_iter_init_for_grad(cpu_iter *iter, Tensor *tensor);

// Free iterator memory
void cpu_iter_free(cpu_iter *iter);

// // Check if iterator has more elements
bool cpu_iter_has_next(const cpu_iter *iter);

// Advance iterator to next element
void cpu_iter_next(cpu_iter *iter);

// Reset iterator to beginning
void cpu_iter_reset(cpu_iter *iter);

// // Type-specific getters (keep as inline for performance)
// static inline void* cpu_iter_get_current(const cpu_iter *iter);

int cpu_iter_get_int(const cpu_iter *iter);

float cpu_iter_get_float(const cpu_iter *iter);

double cpu_iter_get_double(const cpu_iter *iter);

// Type-specific setters
static inline void cpu_iter_set_int(cpu_iter *iter, int64_t value);

static inline void cpu_iter_set_float(cpu_iter *iter, float value);

static inline void cpu_iter_set_double(cpu_iter *iter, double value);

// Broadcasting function
IterError cpu_iter_broadcast(int num_iters, cpu_iter *iters);

// Convenience function for iterating over tensor
#define CPU_ITER_FOREACH(iter, tensor) \
    for (cpu_iter iter; cpu_iter_init(&iter, tensor) == ITER_OK && cpu_iter_has_next(&iter); \
         cpu_iter_next(&iter))

#endif // CPU_ITERATOR_H