#include "C:\Programming\Revamp2\Kernals\kernels.h"
#include "C:\Programming\Revamp2\Kernals\package.h"

#include <stdlib.h>
#include <string.h>

CPU_backend cpu;

#define INIT_COMMON_KERNELS(pkg, type_prefix) do { \
    (pkg)->kernels[P_ADD] = type_prefix##_kernel_add; \
    (pkg)->kernels[P_SUBTRACT] = type_prefix##_kernel_subtract; \
    (pkg)->kernels[P_MULTIPLY] = type_prefix##_kernel_multiply; \
    (pkg)->kernels[P_DIVIDE] = type_prefix##_kernel_divide; \
    (pkg)->kernels[P_SQUARE_ROOT] = type_prefix##_kernel_square_root; \
    (pkg)->kernels[P_ABSOLUTE_VALUE] = type_prefix##_kernel_absolute_value; \
    (pkg)->kernels[P_NEGATION] = type_prefix##_kernel_negate; \
    (pkg)->kernels[P_POWER] = type_prefix##_kernel_power; \
    (pkg)->kernels[P_EQUALITY] = type_prefix##_kernel_equality; \
    (pkg)->kernels[P_GREATER_THAN] = type_prefix##_kernel_greater_than; \
    (pkg)->kernels[P_LESS_THAN] = type_prefix##_kernel_less_than; \
    (pkg)->kernels[P_ELEMENT_WISE_MAXIMUM] = type_prefix##_kernel_maximum; \
    (pkg)->kernels[P_ELEMENT_WISE_MINIMUM] = type_prefix##_kernel_minimum; \
    (pkg)->kernels[P_WHERE] = type_prefix##_kernel_where; \
    (pkg)->kernels[P_FILL] = type_prefix##_kernel_fill; \
    (pkg)->element_wise_numel += 16; \
} while(0)

#define INIT_COMMON_KERNELS_BOOL(pkg, type_prefix) do { \
    (pkg)->kernels[P_EQUALITY] = type_prefix##_kernel_equality; \
    (pkg)->kernels[P_ELEMENT_WISE_MAXIMUM] = type_prefix##_kernel_maximum; \
    (pkg)->kernels[P_ELEMENT_WISE_MINIMUM] = type_prefix##_kernel_minimum; \
    (pkg)->kernels[P_WHERE] = type_prefix##_kernel_where; \
    (pkg)->kernels[P_FILL] = type_prefix##_kernel_fill; \
    (pkg)->element_wise_numel += 5; \
} while(0)


#define INIT_COMMON_KERNELS_SIGNED(pkg, type_prefix) do { \
    (pkg)->kernels[P_ADD] = type_prefix##_kernel_add; \
    (pkg)->kernels[P_SUBTRACT] = type_prefix##_kernel_subtract; \
    (pkg)->kernels[P_MULTIPLY] = type_prefix##_kernel_multiply; \
    (pkg)->kernels[P_DIVIDE] = type_prefix##_kernel_divide; \
    (pkg)->kernels[P_ABSOLUTE_VALUE] = type_prefix##_kernel_absolute_value; \
    (pkg)->kernels[P_NEGATION] = type_prefix##_kernel_negate; \
    (pkg)->kernels[P_EQUALITY] = type_prefix##_kernel_equality; \
    (pkg)->kernels[P_GREATER_THAN] = type_prefix##_kernel_greater_than; \
    (pkg)->kernels[P_LESS_THAN] = type_prefix##_kernel_less_than; \
    (pkg)->kernels[P_ELEMENT_WISE_MAXIMUM] = type_prefix##_kernel_maximum; \
    (pkg)->kernels[P_ELEMENT_WISE_MINIMUM] = type_prefix##_kernel_minimum; \
    (pkg)->kernels[P_WHERE] = type_prefix##_kernel_where; \
    (pkg)->kernels[P_FILL] = type_prefix##_kernel_fill; \
    (pkg)->element_wise_numel += 13; \
} while(0)

#define INIT_COMMON_KERNELS_UNSIGNED(pkg, type_prefix) do { \
    (pkg)->kernels[P_ADD] = type_prefix##_kernel_add; \
    (pkg)->kernels[P_SUBTRACT] = type_prefix##_kernel_subtract; \
    (pkg)->kernels[P_MULTIPLY] = type_prefix##_kernel_multiply; \
    (pkg)->kernels[P_DIVIDE] = type_prefix##_kernel_divide; \
    (pkg)->kernels[P_EQUALITY] = type_prefix##_kernel_equality; \
    (pkg)->kernels[P_GREATER_THAN] = type_prefix##_kernel_greater_than; \
    (pkg)->kernels[P_LESS_THAN] = type_prefix##_kernel_less_than; \
    (pkg)->kernels[P_ELEMENT_WISE_MAXIMUM] = type_prefix##_kernel_maximum; \
    (pkg)->kernels[P_ELEMENT_WISE_MINIMUM] = type_prefix##_kernel_minimum; \
    (pkg)->kernels[P_WHERE] = type_prefix##_kernel_where; \
    (pkg)->kernels[P_FILL] = type_prefix##_kernel_fill; \
    (pkg)->element_wise_numel += 11; \
} while(0)


#define INIT_REDUCTION_KERNELS(pkg, type_prefix) do { \
    (pkg)->reduc_kernels[P_SUM_REDUCTION] = type_prefix##_kernel_sum; \
    (pkg)->reduc_kernels[P_MAX_REDUCTION] = type_prefix##_kernel_max; \
    (pkg)->reduc_kernels[P_MIN_REDUCTION] = type_prefix##_kernel_min; \
    (pkg)->reduction_numel += 3; \
} while(0)

#define INIT_CAST_KERNELS_I8(pkg) do { \
    (pkg)->kernels[P_INT8_TO_FLOAT] = int8_t_float_kernel_cast; \
    (pkg)->kernels[P_INT8_TO_DOUBLE] = int8_t_double_kernel_cast; \
    (pkg)->element_wise_numel += 2; \
} while(0)

#define INIT_CAST_KERNELS_I16(pkg) do { \
    (pkg)->kernels[P_INT16_TO_FLOAT] = int16_t_float_kernel_cast; \
    (pkg)->kernels[P_INT16_TO_DOUBLE] = int16_t_double_kernel_cast; \
    (pkg)->element_wise_numel += 2; \
} while(0)


#define INIT_CAST_KERNELS_I32(pkg) do { \
    (pkg)->kernels[P_INT32_TO_FLOAT] = int32_t_float_kernel_cast; \
    (pkg)->kernels[P_INT32_TO_DOUBLE] = int32_t_double_kernel_cast; \
    (pkg)->element_wise_numel += 2; \
} while(0)

#define INIT_CAST_KERNELS_I64(pkg) do { \
    (pkg)->kernels[P_INT64_TO_FLOAT] = int64_t_float_kernel_cast; \
    (pkg)->kernels[P_INT64_TO_DOUBLE] = int64_t_double_kernel_cast; \
    (pkg)->element_wise_numel += 2; \
} while(0)

#define INIT_CAST_KERNELS_U8(pkg) do { \
    (pkg)->kernels[P_UINT8_TO_FLOAT] = uint8_t_float_kernel_cast; \
    (pkg)->kernels[P_UINT8_TO_DOUBLE] = uint8_t_double_kernel_cast; \
    (pkg)->element_wise_numel += 2; \
} while(0)

#define INIT_CAST_KERNELS_U16(pkg) do { \
    (pkg)->kernels[P_UINT16_TO_FLOAT] = uint16_t_float_kernel_cast; \
    (pkg)->kernels[P_UINT16_TO_DOUBLE] = uint16_t_double_kernel_cast; \
    (pkg)->element_wise_numel += 2; \
} while(0)

#define INIT_CAST_KERNELS_U32(pkg) do { \
    (pkg)->kernels[P_UINT32_TO_FLOAT] = uint32_t_float_kernel_cast; \
    (pkg)->kernels[P_UINT32_TO_DOUBLE] = uint32_t_double_kernel_cast; \
    (pkg)->element_wise_numel += 2; \
} while(0)

#define INIT_CAST_KERNELS_U64(pkg) do { \
    (pkg)->kernels[P_UINT64_TO_FLOAT] = uint64_t_float_kernel_cast; \
    (pkg)->kernels[P_UINT64_TO_DOUBLE] = uint64_t_double_kernel_cast; \
    (pkg)->element_wise_numel += 2; \
} while(0)

#define INIT_CAST_KERNELS_F32(pkg) do { \
    (pkg)->kernels[P_FLOAT_TO_INT8] = float_int8_t_kernel_cast; \
    (pkg)->kernels[P_FLOAT_TO_INT16] = float_int16_t_kernel_cast; \
    (pkg)->kernels[P_FLOAT_TO_INT32] = float_int32_t_kernel_cast; \
    (pkg)->kernels[P_FLOAT_TO_INT64] = float_int64_t_kernel_cast; \
    (pkg)->kernels[P_FLOAT_TO_UINT8] = float_uint8_t_kernel_cast; \
    (pkg)->kernels[P_FLOAT_TO_UINT16] = float_uint16_t_kernel_cast; \
    (pkg)->kernels[P_FLOAT_TO_UINT32] = float_uint32_t_kernel_cast; \
    (pkg)->kernels[P_FLOAT_TO_UINT64] = float_uint64_t_kernel_cast; \
    (pkg)->kernels[P_FLOAT_TO_DOUBLE] = float_double_kernel_cast; \
    (pkg)->element_wise_numel += 9; \
} while(0)

#define INIT_CAST_KERNELS_F64(pkg) do { \
    (pkg)->kernels[P_DOUBLE_TO_FLOAT] = double_float_kernel_cast; \
    (pkg)->kernels[P_DOUBLE_TO_INT8] = double_int8_t_kernel_cast; \
    (pkg)->kernels[P_DOUBLE_TO_INT16] = double_int16_t_kernel_cast; \
    (pkg)->kernels[P_DOUBLE_TO_INT32] = double_int32_t_kernel_cast; \
    (pkg)->kernels[P_DOUBLE_TO_INT64] = double_int64_t_kernel_cast; \
    (pkg)->kernels[P_DOUBLE_TO_UINT8] = double_uint8_t_kernel_cast; \
    (pkg)->kernels[P_DOUBLE_TO_UINT16] = double_uint16_t_kernel_cast; \
    (pkg)->kernels[P_DOUBLE_TO_UINT32] = double_uint32_t_kernel_cast; \
    (pkg)->kernels[P_DOUBLE_TO_UINT64] = double_uint64_t_kernel_cast; \
    (pkg)->element_wise_numel += 9; \
} while(0)

void init_package_bool(package *pkg) {
    INIT_COMMON_KERNELS_BOOL(pkg, bool);
}

void init_package_i8(package *pkg) {
    INIT_COMMON_KERNELS_SIGNED(pkg, i8);
    INIT_REDUCTION_KERNELS(pkg, i8);
    INIT_CAST_KERNELS_I8(pkg);
}

void init_package_i16(package *pkg) {
    INIT_COMMON_KERNELS_SIGNED(pkg, i16);
    INIT_REDUCTION_KERNELS(pkg, i16);
    INIT_CAST_KERNELS_I16(pkg);
}

void init_package_i32(package *pkg) {
    INIT_COMMON_KERNELS_SIGNED(pkg, i32);
    INIT_REDUCTION_KERNELS(pkg, i32);
    INIT_CAST_KERNELS_I32(pkg);
}

void init_package_i64(package *pkg) {
    INIT_COMMON_KERNELS_SIGNED(pkg, i64);
    INIT_REDUCTION_KERNELS(pkg, i64);
    INIT_CAST_KERNELS_I64(pkg);
}

void init_package_u8(package *pkg) {
    INIT_COMMON_KERNELS_UNSIGNED(pkg, u8);
    INIT_REDUCTION_KERNELS(pkg, u8);
    INIT_CAST_KERNELS_U8(pkg);
}

void init_package_u16(package *pkg) {
    INIT_COMMON_KERNELS_UNSIGNED(pkg, u16);
    INIT_REDUCTION_KERNELS(pkg, u16);
    INIT_CAST_KERNELS_U16(pkg);
}

void init_package_u32(package *pkg) {
    INIT_COMMON_KERNELS_UNSIGNED(pkg, u32);
    INIT_REDUCTION_KERNELS(pkg, u32);
    INIT_CAST_KERNELS_U32(pkg);
}

void init_package_u64(package *pkg) {
    INIT_COMMON_KERNELS_UNSIGNED(pkg, u64);
    INIT_REDUCTION_KERNELS(pkg, u64);
    INIT_CAST_KERNELS_U64(pkg);
}

void init_package_f32(package *pkg) {
    INIT_COMMON_KERNELS(pkg, f32);
    INIT_REDUCTION_KERNELS(pkg, f32);
    INIT_CAST_KERNELS_F32(pkg);
}

void init_package_f64(package *pkg) {
    INIT_COMMON_KERNELS(pkg, f64);
    INIT_REDUCTION_KERNELS(pkg, f64);
    INIT_CAST_KERNELS_F64(pkg);
}


static const package_initializer cpu_initializers[DTYPE_COUNT] = {
    [DTYPE_BOOL] = init_package_bool,
    [DTYPE_I8] = init_package_i8,
    [DTYPE_I16] = init_package_i16,
    [DTYPE_I32] = init_package_i32,
    [DTYPE_I64] = init_package_i64,
    [DTYPE_U8] = init_package_u8,
    [DTYPE_U16] = init_package_u16,
    [DTYPE_U32] = init_package_u32,
    [DTYPE_U64] = init_package_u64,
    [DTYPE_F32] = init_package_f32,
    [DTYPE_F64] = init_package_f64,
};


static void init_package_base(package *pkg) {
    pkg->capacity = P_KERNELS_COUNT;
    pkg->kernels = calloc(pkg->capacity, sizeof(Tensor_Iterator_Kernel *));
    pkg->reduc_kernels = calloc(pkg->capacity, sizeof(Tensor_Reduction_Kernel *));
    pkg->element_wise_numel = 0;
    pkg->reduction_numel = 0;
}

void init_cpu_backend(CPU_backend *cpu) {
    if (cpu->initialized) return;


    
    for (int i = 0; i < DTYPE_COUNT; i++) {
        init_package_base(&cpu->packages[i]);
        // if (cpu_initializers[i]) {
        //     cpu_initializers[i](&cpu->packages[i]);
        // }
    }
    
    cpu->packages[DTYPE_I64].kernels[P_ADD] = i64_kernel_add;
    cpu->initialized = true;
}

void cleanup_cpu_backend(CPU_backend *cpu) {
    for (int i = 0; i < DTYPE_COUNT; i++) {
        free(cpu->packages[i].kernels);
        free(cpu->packages[i].reduc_kernels);
        memset(&cpu->packages[i], 0, sizeof(package));
    }
    cpu->initialized = false;
}

package* get_package_cpu(CPU_backend *cpu, dtype dtype) {
    if (dtype >= DTYPE_COUNT) return NULL;
    return &cpu->packages[dtype];
}

Tensor_Iterator_Kernel get_kernel(Backend backend, dtype dtype, kernels kernel) {

      init_cpu_backend(&cpu);

    switch (backend) {
        
        case CPU:
            package *pkg = get_package_cpu(&cpu, dtype);
            if (!pkg || kernel >= pkg->capacity) {
                error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(FAILED_TO_RETRIEVE_KERNEL),  __func__, __FILE__, __LINE__);
                return NULL;
            }
            return pkg->kernels[kernel];
        break;
    
    default:
        break;
    }

}

Tensor_Reduction_Kernel get_reduction_kernel(Backend backend, dtype dtype, kernels kernel) {
    
    switch (backend) {
        
        case CPU:
            package *pkg = get_package_cpu(&cpu, dtype);
            if (!pkg || kernel >= pkg->capacity) {
                error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(FAILED_TO_RETRIEVE_KERNEL),  __func__, __FILE__, __LINE__);
                return NULL;
            }
            return pkg->reduc_kernels[kernel];
        break;
    
    default:
        break;
    }

}


Tensor_Iterator_Kernel get_cast_kernel(Backend backend, dtype from_type, dtype to_type) {
    static const kernels cast_op_map[DTYPE_COUNT][DTYPE_COUNT] = {

        [DTYPE_BOOL] = {0},
        
        [DTYPE_I8] = {
            [DTYPE_F32] = P_INT8_TO_FLOAT,
            [DTYPE_F64] = P_INT8_TO_DOUBLE,
        },
        
        [DTYPE_I16] = {
            [DTYPE_F32] = P_INT16_TO_FLOAT,
            [DTYPE_F64] = P_INT16_TO_DOUBLE,
        },
        
        [DTYPE_I32] = {
            [DTYPE_F32] = P_INT32_TO_FLOAT,
            [DTYPE_F64] = P_INT32_TO_DOUBLE,
        },
        
        [DTYPE_I64] = {
            [DTYPE_F32] = P_INT64_TO_FLOAT,
            [DTYPE_F64] = P_INT64_TO_DOUBLE,
        },
        
        [DTYPE_U8] = {
            [DTYPE_F32] = P_UINT8_TO_FLOAT,
            [DTYPE_F64] = P_UINT8_TO_DOUBLE,
        },
         
        [DTYPE_U16] = {
            [DTYPE_F32] = P_UINT16_TO_FLOAT,
            [DTYPE_F64] = P_UINT16_TO_DOUBLE,
        },
        
        [DTYPE_U32] = {
            [DTYPE_F32] = P_UINT32_TO_FLOAT,
            [DTYPE_F64] = P_UINT32_TO_DOUBLE,
        },
        
        [DTYPE_U64] = {
            [DTYPE_F32] = P_UINT64_TO_FLOAT,
            [DTYPE_F64] = P_UINT64_TO_DOUBLE,
        },
        
        [DTYPE_F32] = {
            [DTYPE_I8] = P_FLOAT_TO_INT8,
            [DTYPE_I16] = P_FLOAT_TO_INT16,
            [DTYPE_I32] = P_FLOAT_TO_INT32,
            [DTYPE_I64] = P_FLOAT_TO_INT64,
            [DTYPE_U8] = P_FLOAT_TO_UINT8,
            [DTYPE_U16] = P_FLOAT_TO_UINT16,
            [DTYPE_U32] = P_FLOAT_TO_UINT32,
            [DTYPE_U64] = P_FLOAT_TO_UINT64,
            [DTYPE_F64] = P_FLOAT_TO_DOUBLE,
        },
        
        [DTYPE_F64] = {
            [DTYPE_I8] = P_DOUBLE_TO_INT8,
            [DTYPE_I16] = P_DOUBLE_TO_INT16,
            [DTYPE_I32] = P_DOUBLE_TO_INT32,
            [DTYPE_I64] = P_DOUBLE_TO_INT64,
            [DTYPE_U8] = P_DOUBLE_TO_UINT8,
            [DTYPE_U16] = P_DOUBLE_TO_UINT16,
            [DTYPE_U32] = P_DOUBLE_TO_UINT32,
            [DTYPE_U64] = P_DOUBLE_TO_UINT64,
            [DTYPE_F32] = P_DOUBLE_TO_FLOAT,
        },
    };
    
    if (from_type >= DTYPE_COUNT || to_type >= DTYPE_COUNT) return NULL;
    
    kernels cast_op = cast_op_map[from_type][to_type];
    if (cast_op == 0) return NULL; 

    
    switch (backend) {
        
        case CPU:
            package *pkg = get_package_cpu(&cpu, from_type);
            if (!pkg || cast_op >= pkg->capacity) {
                error_print("%s in function: %s, file: %s, line: %d, RETURN: NULL\n", common_error_code_to_string(FAILED_TO_RETRIEVE_KERNEL),  __func__, __FILE__, __LINE__);
                return NULL;
            }
            return pkg->kernels[cast_op];
        break;
    
    default:
        break;
    }
}

