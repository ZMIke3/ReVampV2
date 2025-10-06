#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

typedef enum {
    ADD = 0,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    EQUALITY,
    GREATER_THAN,
    LESS_THAN,
    SQUARE_ROOT,
    ABSOLUTE_VALUE,
    NEGATION,
    POWER,
    MAXIMUM,
    MINIMUM,
    WHERE,
    FILL,
    SUM_REDUCTION,
    MAX_REDUCTION,
    MIN_REDUCTION,
    OPCOUNT
} OpTypes;


typedef enum {
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    F32, F64,
    BOOL,
    NUM_DTYPES
} DType;

static const char *dtype_names[NUM_DTYPES] = {
    "i8","i16","i32","i64",
    "u8","u16","u32","u64",
    "f32","f64","bool"
};

static const char *ctype_names[NUM_DTYPES] = {
    "int8_t","int16_t","int32_t","int64_t",
    "uint8_t","uint16_t","uint32_t","uint64_t",
    "float","double","bool"
};

typedef enum {
    REGULAR_S,
    WHERE_S,
    FILL_S,
    REDUCTION_S,
    CAST_S,
}Specialized;

typedef struct {
    const char *name;
    int arity;
    bool (*is_valid)(DType);
    const char *(*expr)(DType);
    Specialized special;
} OpRule;

typedef struct {
    const char *from;
    const char *to;
}CastRule;


bool valid_all(DType t) {
    return true;
}

bool valid_numeric(DType t) {
    return t != BOOL;
}

bool valid_float(DType t) {
    return t == F32 || t == F64;
}

bool valid_abs(DType t) {
    return true; 
}

bool valid_signed(DType t) {
    switch (t) {
        case I8:
        case I16: 
        case I32:
        case I64:
        case F32: 
        case F64:
        return true;
        default: return false;
    }
}


const char* add_expr(DType t) {
    return "*out = *in0 + *in1";
}

const char* sub_expr(DType t) {
    return "*out = *in0 - *in1";
}


const char* mul_expr(DType t) {
    return "*out = *in0 * *in1";
}

const char* div_expr(DType t) {
    return "*out = *in0 / *in1";
}

const char* sqrt_expr(DType t) {
    return (t == F32) ? "*out = sqrtf(*in0)" : "*out = sqrt(*in0)";
}

const char* abs_expr(DType t) {
    switch (t) {
        case I8: case I16: case I32: return "*out = abs(*in0)";
        case I64: return "*out = llabs(*in0)";
        case F32: return "*out = fabsf(*in0)";
        case F64: return "*out = fabs(*in0)";
        default: return NULL;
    }
}

const char* eq_expr(DType t) {
    return "*out = *in0 == *in1";
}


const char* gt_expr(DType t) {
    return "*out = *in0 > *in1";
}

const char* lt_expr(DType t) {
    return "*out = *in0 < *in1";
}


const char *neg_expr(DType t) {
    return "*out = -*in0";
}

const char *pow_expr(DType t) {
    switch (t) {
        case F32: return "*out = powf(*in0, *in1)";
        case F64: return "*out = pow(*in0, *in1)";
        default: return NULL;
    }
}

const char *emax_expr(DType t) {
    switch (t) {
        case I8: return  "*out = max_8(*in0, *in1)";
        case I16: return "*out = max_16(*in0, *in1)";
        case I32: return "*out = max_32(*in0, *in1)";
        case I64: return "*out = max_64(*in0, *in1)";
        case U8: return  "*out = max_u8(*in0, *in1)";
        case U16: return "*out = max_u16(*in0, *in1)";
        case U32: return "*out = max_u32(*in0, *in1)";
        case U64: return "*out = max_u64(*in0, *in1)";
        case F32: return "*out = fmaxf(*in0, *in1)";
        case F64: return "*out = fmax(*in0, *in1)";
        case BOOL: return "*out = max_bool(*in0, *in1)";
        default: return NULL;
    }
}


const char *emin_expr(DType t) {
    switch (t) {
        case I8: return  "*out = min_8(*in0, *in1)";
        case I16: return "*out = min_16(*in0, *in1)";
        case I32: return "*out = min_32(*in0, *in1)";
        case I64: return "*out = min_64(*in0, *in1)";
        case U8: return  "*out = min_u8(*in0, *in1)";
        case U16: return "*out = min_u16(*in0, *in1)";
        case U32: return "*out = min_u32(*in0, *in1)";
        case U64: return "*out = min_u64(*in0, *in1)";
        case F32: return "*out = fminf(*in0, *in1)";
        case F64: return "*out = fmin(*in0, *in1)";
        case BOOL: return "*out = min_bool(*in0, *in1)";
        default: return NULL;
    }
}

const char *where_expr(DType t) {
    return "*out = *in0 ? *in1 : *in2";
}

const char *fill_expr(DType t) {
    return "*out = fill_value";
}

const char *sum_expr(DType t) {
    return "out[output_indexes[i]] + current_value";
}

const char *max_expr(DType t) {
    return "current_value > out[output_indexes[i]] ? current_value : out[output_indexes[i]]";
}

const char *min_expr(DType t) {
    return "current_value < out[output_indexes[i]] ? current_value : out[output_indexes[i]]";
}

OpRule ops[] = {
    { "add",        2, valid_numeric, add_expr,  REGULAR_S},
    { "subtract",   2, valid_numeric, sub_expr,  REGULAR_S },
    { "multiply",   2, valid_numeric, mul_expr,  REGULAR_S },
    { "divide",     2, valid_numeric, div_expr,  REGULAR_S },
    { "equality",   2, valid_all, eq_expr,  REGULAR_S },
    { "greater_than", 2, valid_numeric, gt_expr,  REGULAR_S },
    { "less_than",  2, valid_numeric, lt_expr,  REGULAR_S },
    { "square_root", 1, valid_float,   sqrt_expr,  REGULAR_S },
    { "absolute_value", 1, valid_abs,  abs_expr,  REGULAR_S },
    { "negate", 1, valid_signed, neg_expr,  REGULAR_S},
    { "power", 2, valid_float, pow_expr,  REGULAR_S},
    { "maximum", 2, valid_all, emax_expr,  REGULAR_S},
    { "minimum", 2, valid_all, emin_expr,  REGULAR_S},
    { "where", 3, valid_all, where_expr,  WHERE_S},
    { "fill", 1, valid_all, fill_expr,  FILL_S},
    { "sum", 2, valid_numeric, sum_expr,  REDUCTION_S},
    { "max", 1, valid_numeric, max_expr,  REDUCTION_S},
    { "min", 1, valid_numeric, min_expr,  REDUCTION_S},
    
};

CastRule casts[] = {
    {"int8_t", "float"},
    {"int8_t", "double"},
    {"int16_t", "float"},
    {"int16_t", "double"},
    {"int32_t", "float"},
    {"int32_t", "double"},
    {"int64_t", "float"},
    {"int64_t", "double"},
    {"uint8_t", "float"},
    {"uint8_t", "double"},
    {"uint16_t", "float"},
    {"uint16_t", "double"},
    {"uint32_t", "float"},
    {"uint32_t", "double"},
    {"uint64_t", "float"},
    {"uint64_t", "double"},
    {"float", "int8_t"},
    {"float", "int16_t"},
    {"float", "int32_t"},
    {"float", "int64_t"},
    {"float", "uint8_t"},
    {"float", "uint16_t"},
    {"float", "uint32_t"},
    {"float", "uint64_t"},
    {"double", "int8_t"},
    {"double", "int16_t"},
    {"double", "int32_t"},
    {"double", "int64_t"},
    {"double", "uint8_t"},
    {"double", "uint16_t"},
    {"double", "uint32_t"},
    {"double", "uint64_t"},
    {"float", "double"},
    {"double", "float"},
    {"int8_t", "int16_t"},
    {"int8_t", "int32_t"},
    {"int8_t", "int64_t"},
    {"int16_t", "int32_t"},
    {"int16_t", "int64_t"},
    {"int32_t", "int64_t"},
    {"uint8_t", "uint16_t"},
    {"uint8_t", "uint32_t"},
    {"uint8_t", "uint64_t"},
    {"uint16_t", "uint32_t"},
    {"uint16_t", "uint64_t"},
    {"uint32_t", "uint64_t"},
    {"int8_t", "uint8_t"},
    {"int16_t", "uint16_t"},
    {"int32_t", "uint32_t"},
    {"int64_t", "uint64_t"},
    {"uint8_t", "int8_t"},
    {"uint16_t", "int16_t"},
    {"uint32_t", "int32_t"},
    {"uint64_t", "int64_t"},


};



void kernel_gen(FILE *file, const char *name, const char *type, const char *ctype, const char *operation, int nargs, Specialized specialization) {


    switch (specialization) {
        case REGULAR_S:
            fprintf(file, "void %s_kernel_%s(char **ptrs, const int64_t *stride, int64_t n) {\n", type, name);

            for (int a = 0; a < nargs; a++) {
                fprintf(file, "    %s *in%d = (%s*)ptrs[%d];\n", ctype, a, ctype, a);
            }

            fprintf(file, "    %s *out = (%s*)ptrs[%d];\n", ctype, ctype, nargs);
            
            fprintf(file, "    for (int64_t i = 0; i < n; i++) {\n");
            fprintf(file, "        %s;\n", operation);

            for (int a = 0; a < nargs; a++) {
                fprintf(file, "        in%d = (%s*)((char*)in%d + stride[%d]);\n", a, ctype, a, a);
            }
            fprintf(file, "        out = (%s*)((char*)out + stride[%d]);\n", ctype, nargs);

            fprintf(file, "    }\n}\n\n");
        break;


        case WHERE_S:
            fprintf(file, "void %s_kernel_%s(char **ptrs, const int64_t *stride, int64_t n) {\n", type, name);

            fprintf(file, "    bool *in0 = (bool*)ptrs[0];\n");

            for (int a = 1; a < nargs; a++) {
                fprintf(file, "    %s *in%d = (%s*)ptrs[%d];\n", ctype, a, ctype, a);
            }

            fprintf(file, "    %s *out = (%s*)ptrs[%d];\n", ctype, ctype, nargs);


            fprintf(file, "    for (int64_t i = 0; i < n; i++) {\n");
            fprintf(file, "        %s;\n", operation);

            fprintf(file, "        in0 = (bool*)((char*)in0 + stride[0]);\n");
            for (int a = 1; a < nargs; a++) {
                fprintf(file, "        in%d = (%s*)((char*)in%d + stride[%d]);\n", a, ctype, a, a);
            }
            fprintf(file, "        out = (%s*)((char*)out + stride[%d]);\n", ctype, nargs);

            fprintf(file, "    }\n}\n\n");
        break;


        case FILL_S:
            fprintf(file, "void %s_kernel_%s(char **ptrs, const int64_t *stride, int64_t n) {\n", type, name);

            for (int a = 0; a < nargs; a++) {
                fprintf(file, "    %s *in%d = (%s*)ptrs[%d];\n", ctype, a, ctype, a);
            }

            fprintf(file, "    %s *out = (%s*)ptrs[%d];\n", ctype, ctype, nargs);
            fprintf(file, "    %s fill_value = *in0;\n", ctype);

            fprintf(file, "    for (int64_t i = 0; i < n; i++) {\n");
            fprintf(file, "       %s;\n", operation);

            fprintf(file, "        out = (%s*)((char*)out + stride[1]);\n", ctype);

            fprintf(file, "    }\n}\n\n");

        break;

        case REDUCTION_S:
            fprintf(file, "void %s_kernel_%s(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {\n", type, name);

            fprintf(file, "    %s *in0 = (%s*)ptrs[idx_in];\n", ctype, ctype);
            fprintf(file, "    %s *out = (%s*)ptrs[idx_out];\n", ctype, ctype);

            fprintf(file, "    %s *values = (%s*)input_values;\n", ctype, ctype);
            fprintf(file,  "    %s current_value = 0;\n", ctype);

            fprintf(file, "    for (int64_t i = 0; i < length; i++) {\n");
            fprintf(file, "         current_value = values[i];\n");

            fprintf(file, "         out[output_indexes[i]] = %s;\n", operation);

            fprintf(file, "    }\n}\n\n");
        break;

        case CAST_S:
            fprintf(file, "void %s_kernel_%s(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {\n", type, name);

            fprintf(file, "    %s *in0 = (%s*)ptrs[idx_in];\n", ctype, ctype);
            fprintf(file, "    %s *out = (%s*)ptrs[idx_out];\n", ctype, ctype);

            fprintf(file, "    %s *values = (%s*)input_values;\n", ctype, ctype);
            fprintf(file,  "    %s current_value = 0;\n", ctype);

            fprintf(file, "    for (int64_t i = 0; i < length; i++) {\n");
            fprintf(file, "         current_value = values[i];\n");

            fprintf(file, "         out[output_indexes[i]] = %s;\n", operation);

            fprintf(file, "    }\n}\n\n");
        break;

    default:
        break;
    }
   

    
}

void generate_boilerplate(const char *filename, char *mode) {

    FILE *file = fopen(filename, mode);
    if (!file) {
        fprintf(stderr, "Could not open %s for writing\n", filename);
        return;
    }

    char *functions = "int8_t max_8(int8_t a, int8_t b) { return a > b ? a : b; }\n"
                      "int8_t min_8(int8_t a, int8_t b) { return a < b ? a : b; }\n"
                      "int16_t max_16(int16_t a, int16_t b) { return a > b ? a : b; }\n"
                      "int16_t min_16(int16_t a, int16_t b) { return a < b ? a : b; }\n"
                      "int32_t max_32(int32_t a, int32_t b) { return a > b ? a : b; }\n"
                      "int32_t min_32(int32_t a, int32_t b) { return a < b ? a : b; }\n"
                      "int64_t max_64(int64_t a, int64_t b) { return a > b ? a : b; }\n"
                      "int64_t min_64(int64_t a, int64_t b) { return a < b ? a : b; }\n"
                      "uint8_t max_u8(uint8_t a, uint8_t b) { return a > b ? a : b; }\n"
                      "uint8_t min_u8(uint8_t a, uint8_t b) { return a < b ? a : b; }\n"
                      "uint16_t max_u16(uint16_t a, uint16_t b) { return a > b ? a : b; }\n"
                      "uint16_t min_u16(uint16_t a, uint16_t b) { return a < b ? a : b; }\n"
                      "uint32_t max_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }\n"
                      "uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }\n"
                      "uint64_t max_u64(uint64_t a, uint64_t b) { return a > b ? a : b; }\n"
                      "uint64_t min_u64(uint64_t a, uint64_t b) { return a < b ? a : b; }\n"
                      "bool max_bool(bool a, bool b) { return a || b; }\n"
                      "bool min_bool(bool a, bool b) { return a && b; }\n";

    fprintf(file, "#include <stdlib.h>\n");
    fprintf(file, "#include <stdint.h>\n");
    fprintf(file, "#include <inttypes.h>\n");
    fprintf(file, "#include <math.h>\n");
    fprintf(file, "#include <stdbool.h>\n");
    fprintf(file, "\n");
    fprintf(file, "%s", functions);
    fprintf(file, "\n\n\n\n");

    fflush(file);
    fclose(file);

}

void generate_kernels(const char *filename, char *mode) {

    FILE *file = fopen(filename, mode);
    if (!file) {
        fprintf(stderr, "Could not open %s for writing\n", filename);
        return;
    }

    for (int i = 0; i < OPCOUNT; i++) {

        for (int t = 0; t < NUM_DTYPES; t++) {
            if (!ops[i].is_valid(t)) continue;

            const char *expr = ops[i].expr(t);
            if (!expr) continue;

            kernel_gen(file, ops[i].name, dtype_names[t], ctype_names[t], expr, ops[i].arity, ops[i].special);
            fflush(file);
        }   

    }

    fflush(file);
    fclose(file);

}

void generate_cast_kernels(const char *filename, char *mode) {

    int numofcast = sizeof(casts) / sizeof(casts[0]);

    FILE *file = fopen(filename, mode);
    if (!file) {
        fprintf(stderr, "Could not open %s for writing\n", filename);
        return;
    }

    for (int i = 0; i < numofcast; i++) {
       
        fprintf(file, "void %s_%s_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {\n", casts[i].from, casts[i].to);
        fprintf(file, "      %s *in0 = (%s*)ptrs[0];\n", casts[i].from, casts[i].from);
        fprintf(file, "      %s *out = (%s*)ptrs[1];\n", casts[i].to, casts[i].to);
        fprintf(file, "      for (int64_t i = 0; i < n; i++) {\n");
        fprintf(file, "           *out = (%s)(*in0);\n", casts[i].to);
        fprintf(file, "            in0 = (%s*)((char*)in0 + stride[0]); \n", casts[i].from);
        fprintf(file, "            out = (%s*)((char*)out + stride[1]); \n", casts[i].to);
        fprintf(file, "    }\n}\n\n");
    }

    fprintf(file, "\n");
    fflush(file);
    fclose(file);

}

void generate_kernels_header(const char *filename, char *mode) {
    FILE *file = fopen(filename, mode);
    if (!file) {
        fprintf(stderr, "Could not open %s for writing\n", filename);
        return;
    }

    fprintf(file, "#ifndef KERNELS_H\n#define KERNELS_H\n\n");
    fprintf(file, "#include <stdint.h>\n\n");

    for (int i = 0; i < OPCOUNT; i++) {
        for (int t = 0; t < NUM_DTYPES; t++) {
            const char *name = ops[i].name;
            const char *dtype = dtype_names[t];

       //     printf("This is name: %s, dtype: %s, OP: %d\n", name, dtype, i);

            if (((ops[i].special == REGULAR_S || ops[i].special == WHERE_S || ops[i].special == FILL_S) && ops[i].is_valid(t))) {
                fprintf(file, "void %s_kernel_%s(char **ptrs, const int64_t *stride, int64_t n);\n", dtype, name);
            } else if (ops[i].special == REDUCTION_S && ops[i].is_valid(t)) {
                fprintf(file, "void %s_kernel_%s(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);\n", dtype, name);
            }

        }

        fprintf(file, "\n\n");
    }

    int numofcast = sizeof(casts) / sizeof(casts[0]);
    for (int i = 0; i < numofcast; i++) {
        fprintf(file, "void %s_%s_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);\n", casts[i].from, casts[i].to);
    }

    fprintf(file, "\n#endif // KERNELS_H\n");

    fflush(file);
    fclose(file);
}

int main () {

    char *kernels_c_file = "kernels.c";
    
    char *kernels_h_file = "kernels.h";
    
    generate_kernels_header(kernels_h_file, "w");

    generate_boilerplate(kernels_c_file, "w");

    generate_cast_kernels(kernels_c_file, "a");

    generate_kernels(kernels_c_file, "a");


}

