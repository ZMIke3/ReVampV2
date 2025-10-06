#ifndef KERNELS_H
#define KERNELS_H

#include <stdint.h>

void i8_kernel_add(char **ptrs, const int64_t *stride, int64_t n);
void i16_kernel_add(char **ptrs, const int64_t *stride, int64_t n);
void i32_kernel_add(char **ptrs, const int64_t *stride, int64_t n);
void i64_kernel_add(char **ptrs, const int64_t *stride, int64_t n);
void u8_kernel_add(char **ptrs, const int64_t *stride, int64_t n);
void u16_kernel_add(char **ptrs, const int64_t *stride, int64_t n);
void u32_kernel_add(char **ptrs, const int64_t *stride, int64_t n);
void u64_kernel_add(char **ptrs, const int64_t *stride, int64_t n);
void f32_kernel_add(char **ptrs, const int64_t *stride, int64_t n);
void f64_kernel_add(char **ptrs, const int64_t *stride, int64_t n);


void i8_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n);
void i16_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n);
void i32_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n);
void i64_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n);
void u8_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n);
void u16_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n);
void u32_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n);
void u64_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n);
void f32_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n);
void f64_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n);


void i8_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n);
void i16_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n);
void i32_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n);
void i64_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n);
void u8_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n);
void u16_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n);
void u32_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n);
void u64_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n);
void f32_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n);
void f64_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n);


void i8_kernel_divide(char **ptrs, const int64_t *stride, int64_t n);
void i16_kernel_divide(char **ptrs, const int64_t *stride, int64_t n);
void i32_kernel_divide(char **ptrs, const int64_t *stride, int64_t n);
void i64_kernel_divide(char **ptrs, const int64_t *stride, int64_t n);
void u8_kernel_divide(char **ptrs, const int64_t *stride, int64_t n);
void u16_kernel_divide(char **ptrs, const int64_t *stride, int64_t n);
void u32_kernel_divide(char **ptrs, const int64_t *stride, int64_t n);
void u64_kernel_divide(char **ptrs, const int64_t *stride, int64_t n);
void f32_kernel_divide(char **ptrs, const int64_t *stride, int64_t n);
void f64_kernel_divide(char **ptrs, const int64_t *stride, int64_t n);


void i8_kernel_equality(char **ptrs, const int64_t *stride, int64_t n);
void i16_kernel_equality(char **ptrs, const int64_t *stride, int64_t n);
void i32_kernel_equality(char **ptrs, const int64_t *stride, int64_t n);
void i64_kernel_equality(char **ptrs, const int64_t *stride, int64_t n);
void u8_kernel_equality(char **ptrs, const int64_t *stride, int64_t n);
void u16_kernel_equality(char **ptrs, const int64_t *stride, int64_t n);
void u32_kernel_equality(char **ptrs, const int64_t *stride, int64_t n);
void u64_kernel_equality(char **ptrs, const int64_t *stride, int64_t n);
void f32_kernel_equality(char **ptrs, const int64_t *stride, int64_t n);
void f64_kernel_equality(char **ptrs, const int64_t *stride, int64_t n);
void bool_kernel_equality(char **ptrs, const int64_t *stride, int64_t n);


void i8_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n);
void i16_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n);
void i32_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n);
void i64_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n);
void u8_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n);
void u16_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n);
void u32_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n);
void u64_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n);
void f32_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n);
void f64_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n);


void i8_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n);
void i16_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n);
void i32_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n);
void i64_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n);
void u8_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n);
void u16_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n);
void u32_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n);
void u64_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n);
void f32_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n);
void f64_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n);


void f32_kernel_square_root(char **ptrs, const int64_t *stride, int64_t n);
void f64_kernel_square_root(char **ptrs, const int64_t *stride, int64_t n);


void i8_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n);
void i16_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n);
void i32_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n);
void i64_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n);
void u8_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n);
void u16_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n);
void u32_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n);
void u64_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n);
void f32_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n);
void f64_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n);
void bool_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n);


void i8_kernel_negate(char **ptrs, const int64_t *stride, int64_t n);
void i16_kernel_negate(char **ptrs, const int64_t *stride, int64_t n);
void i32_kernel_negate(char **ptrs, const int64_t *stride, int64_t n);
void i64_kernel_negate(char **ptrs, const int64_t *stride, int64_t n);
void f32_kernel_negate(char **ptrs, const int64_t *stride, int64_t n);
void f64_kernel_negate(char **ptrs, const int64_t *stride, int64_t n);


void f32_kernel_power(char **ptrs, const int64_t *stride, int64_t n);
void f64_kernel_power(char **ptrs, const int64_t *stride, int64_t n);


void i8_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n);
void i16_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n);
void i32_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n);
void i64_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n);
void u8_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n);
void u16_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n);
void u32_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n);
void u64_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n);
void f32_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n);
void f64_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n);
void bool_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n);


void i8_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n);
void i16_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n);
void i32_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n);
void i64_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n);
void u8_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n);
void u16_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n);
void u32_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n);
void u64_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n);
void f32_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n);
void f64_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n);
void bool_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n);


void i8_kernel_where(char **ptrs, const int64_t *stride, int64_t n);
void i16_kernel_where(char **ptrs, const int64_t *stride, int64_t n);
void i32_kernel_where(char **ptrs, const int64_t *stride, int64_t n);
void i64_kernel_where(char **ptrs, const int64_t *stride, int64_t n);
void u8_kernel_where(char **ptrs, const int64_t *stride, int64_t n);
void u16_kernel_where(char **ptrs, const int64_t *stride, int64_t n);
void u32_kernel_where(char **ptrs, const int64_t *stride, int64_t n);
void u64_kernel_where(char **ptrs, const int64_t *stride, int64_t n);
void f32_kernel_where(char **ptrs, const int64_t *stride, int64_t n);
void f64_kernel_where(char **ptrs, const int64_t *stride, int64_t n);
void bool_kernel_where(char **ptrs, const int64_t *stride, int64_t n);


void i8_kernel_fill(char **ptrs, const int64_t *stride, int64_t n);
void i16_kernel_fill(char **ptrs, const int64_t *stride, int64_t n);
void i32_kernel_fill(char **ptrs, const int64_t *stride, int64_t n);
void i64_kernel_fill(char **ptrs, const int64_t *stride, int64_t n);
void u8_kernel_fill(char **ptrs, const int64_t *stride, int64_t n);
void u16_kernel_fill(char **ptrs, const int64_t *stride, int64_t n);
void u32_kernel_fill(char **ptrs, const int64_t *stride, int64_t n);
void u64_kernel_fill(char **ptrs, const int64_t *stride, int64_t n);
void f32_kernel_fill(char **ptrs, const int64_t *stride, int64_t n);
void f64_kernel_fill(char **ptrs, const int64_t *stride, int64_t n);
void bool_kernel_fill(char **ptrs, const int64_t *stride, int64_t n);


void i8_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void i16_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void i32_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void i64_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void u8_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void u16_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void u32_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void u64_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void f32_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void f64_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);


void i8_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void i16_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void i32_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void i64_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void u8_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void u16_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void u32_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void u64_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void f32_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void f64_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);


void i8_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void i16_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void i32_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void i64_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void u8_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void u16_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void u32_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void u64_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void f32_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);
void f64_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out);


void int8_t_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int8_t_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int16_t_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int16_t_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int32_t_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int32_t_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int64_t_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int64_t_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint8_t_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint8_t_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint16_t_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint16_t_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint32_t_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint32_t_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint64_t_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint64_t_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void float_int8_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void float_int16_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void float_int32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void float_int64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void float_uint8_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void float_uint16_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void float_uint32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void float_uint64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void double_int8_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void double_int16_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void double_int32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void double_int64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void double_uint8_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void double_uint16_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void double_uint32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void double_uint64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void float_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void double_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int8_t_int16_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int8_t_int32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int8_t_int64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int16_t_int32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int16_t_int64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int32_t_int64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint8_t_uint16_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint8_t_uint32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint8_t_uint64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint16_t_uint32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint16_t_uint64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint32_t_uint64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int8_t_uint8_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int16_t_uint16_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int32_t_uint32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void int64_t_uint64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint8_t_int8_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint16_t_int16_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint32_t_int32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);
void uint64_t_int64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n);

#endif // KERNELS_H
