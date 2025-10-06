#ifndef PACKAGE_H
#define PACKAGE_H

#include "C:\Programming\Revamp2\Kernals\kernels.h"
#include "C:\Programming\Revamp2\Iterator\Iterator.h"
#include "C:\Programming\Revamp2\Tensor\Tensor.h"


typedef enum {
    P_ADD = 0, 
    P_SUBTRACT,
    P_MULTIPLY,
    P_DIVIDE,
    P_SQUARE_ROOT,
    P_ABSOLUTE_VALUE,
    P_NEGATION,
    P_POWER,
    P_EQUALITY,
    P_GREATER_THAN,
    P_LESS_THAN,
    P_ELEMENT_WISE_MAXIMUM,
    P_ELEMENT_WISE_MINIMUM,
    P_WHERE,
    P_ZEROS,
    P_ONES,
    P_FILL,
    P_SUM_REDUCTION,
    P_MAX_REDUCTION,
    P_MIN_REDUCTION,

    P_INT8_TO_FLOAT,
    P_INT8_TO_DOUBLE,
    P_INT16_TO_FLOAT,
    P_INT16_TO_DOUBLE,
    P_INT32_TO_FLOAT,
    P_INT32_TO_DOUBLE,
    P_INT64_TO_FLOAT,
    P_INT64_TO_DOUBLE,

    P_UINT8_TO_FLOAT,
    P_UINT8_TO_DOUBLE,
    P_UINT16_TO_FLOAT,
    P_UINT16_TO_DOUBLE,
    P_UINT32_TO_FLOAT,
    P_UINT32_TO_DOUBLE,
    P_UINT64_TO_FLOAT,
    P_UINT64_TO_DOUBLE,

    P_FLOAT_TO_INT8,
    P_FLOAT_TO_INT16,
    P_FLOAT_TO_INT32,
    P_FLOAT_TO_INT64,

    P_FLOAT_TO_UINT8,
    P_FLOAT_TO_UINT16,
    P_FLOAT_TO_UINT32,
    P_FLOAT_TO_UINT64,
    P_FLOAT_TO_DOUBLE,

    P_DOUBLE_TO_FLOAT,
    P_DOUBLE_TO_INT8,
    P_DOUBLE_TO_INT16,
    P_DOUBLE_TO_INT32,
    P_DOUBLE_TO_INT64,

    P_DOUBLE_TO_UINT8,
    P_DOUBLE_TO_UINT16,
    P_DOUBLE_TO_UINT32,
    P_DOUBLE_TO_UINT64,
    P_KERNELS_COUNT
} kernels;


typedef struct package {
    Tensor_Iterator_Kernel *kernels;
    Tensor_Reduction_Kernel *reduc_kernels;
    int capacity;
    int element_wise_numel;
    int reduction_numel;
}package;


typedef struct CPU_backend {
    package packages[DTYPE_COUNT];
    bool initialized;
} CPU_backend;

typedef void (*package_initializer)(package *pkg);

void new_cpu_backend(CPU_backend *CPU);
void init_packages();
void init_cpu_packages(CPU_backend *CPU);

package* get_package_cpu(CPU_backend *cpu, dtype dtype);
Tensor_Iterator_Kernel get_kernel(Backend backend, dtype result_type, kernels op);
Tensor_Reduction_Kernel get_reduction_kernel(Backend backend, dtype result_type, kernels op);
Tensor_Iterator_Kernel get_cast_kernel(Backend backend, dtype from_type, dtype to_type);

void init_package_bool(package *pkg);
void init_package_i8(package *pkg);
void init_package_i16(package *pkg);
void init_package_i32(package *pkg);
void init_package_i64(package *pkg);
void init_package_u8(package *pkg);
void init_package_u16(package *pkg);
void init_package_u32(package *pkg);
void init_package_u64(package *pkg);
void init_package_f32(package *pkg);
void init_package_f64(package *pkg);




#endif // PACKAGE_H
