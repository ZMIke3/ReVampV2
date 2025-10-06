#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>

int8_t max_8(int8_t a, int8_t b) { return a > b ? a : b; }
int8_t min_8(int8_t a, int8_t b) { return a < b ? a : b; }
int16_t max_16(int16_t a, int16_t b) { return a > b ? a : b; }
int16_t min_16(int16_t a, int16_t b) { return a < b ? a : b; }
int32_t max_32(int32_t a, int32_t b) { return a > b ? a : b; }
int32_t min_32(int32_t a, int32_t b) { return a < b ? a : b; }
int64_t max_64(int64_t a, int64_t b) { return a > b ? a : b; }
int64_t min_64(int64_t a, int64_t b) { return a < b ? a : b; }
uint8_t max_u8(uint8_t a, uint8_t b) { return a > b ? a : b; }
uint8_t min_u8(uint8_t a, uint8_t b) { return a < b ? a : b; }
uint16_t max_u16(uint16_t a, uint16_t b) { return a > b ? a : b; }
uint16_t min_u16(uint16_t a, uint16_t b) { return a < b ? a : b; }
uint32_t max_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }
uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }
uint64_t max_u64(uint64_t a, uint64_t b) { return a > b ? a : b; }
uint64_t min_u64(uint64_t a, uint64_t b) { return a < b ? a : b; }
bool max_bool(bool a, bool b) { return a || b; }
bool min_bool(bool a, bool b) { return a && b; }




void int8_t_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int8_t *in0 = (int8_t*)ptrs[0];
      float *out = (float*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (float)(*in0);
            in0 = (int8_t*)((char*)in0 + stride[0]); 
            out = (float*)((char*)out + stride[1]); 
    }
}

void int8_t_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int8_t *in0 = (int8_t*)ptrs[0];
      double *out = (double*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (double)(*in0);
            in0 = (int8_t*)((char*)in0 + stride[0]); 
            out = (double*)((char*)out + stride[1]); 
    }
}

void int16_t_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int16_t *in0 = (int16_t*)ptrs[0];
      float *out = (float*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (float)(*in0);
            in0 = (int16_t*)((char*)in0 + stride[0]); 
            out = (float*)((char*)out + stride[1]); 
    }
}

void int16_t_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int16_t *in0 = (int16_t*)ptrs[0];
      double *out = (double*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (double)(*in0);
            in0 = (int16_t*)((char*)in0 + stride[0]); 
            out = (double*)((char*)out + stride[1]); 
    }
}

void int32_t_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int32_t *in0 = (int32_t*)ptrs[0];
      float *out = (float*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (float)(*in0);
            in0 = (int32_t*)((char*)in0 + stride[0]); 
            out = (float*)((char*)out + stride[1]); 
    }
}

void int32_t_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int32_t *in0 = (int32_t*)ptrs[0];
      double *out = (double*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (double)(*in0);
            in0 = (int32_t*)((char*)in0 + stride[0]); 
            out = (double*)((char*)out + stride[1]); 
    }
}

void int64_t_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int64_t *in0 = (int64_t*)ptrs[0];
      float *out = (float*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (float)(*in0);
            in0 = (int64_t*)((char*)in0 + stride[0]); 
            out = (float*)((char*)out + stride[1]); 
    }
}

void int64_t_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int64_t *in0 = (int64_t*)ptrs[0];
      double *out = (double*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (double)(*in0);
            in0 = (int64_t*)((char*)in0 + stride[0]); 
            out = (double*)((char*)out + stride[1]); 
    }
}

void uint8_t_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint8_t *in0 = (uint8_t*)ptrs[0];
      float *out = (float*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (float)(*in0);
            in0 = (uint8_t*)((char*)in0 + stride[0]); 
            out = (float*)((char*)out + stride[1]); 
    }
}

void uint8_t_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint8_t *in0 = (uint8_t*)ptrs[0];
      double *out = (double*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (double)(*in0);
            in0 = (uint8_t*)((char*)in0 + stride[0]); 
            out = (double*)((char*)out + stride[1]); 
    }
}

void uint16_t_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint16_t *in0 = (uint16_t*)ptrs[0];
      float *out = (float*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (float)(*in0);
            in0 = (uint16_t*)((char*)in0 + stride[0]); 
            out = (float*)((char*)out + stride[1]); 
    }
}

void uint16_t_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint16_t *in0 = (uint16_t*)ptrs[0];
      double *out = (double*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (double)(*in0);
            in0 = (uint16_t*)((char*)in0 + stride[0]); 
            out = (double*)((char*)out + stride[1]); 
    }
}

void uint32_t_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint32_t *in0 = (uint32_t*)ptrs[0];
      float *out = (float*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (float)(*in0);
            in0 = (uint32_t*)((char*)in0 + stride[0]); 
            out = (float*)((char*)out + stride[1]); 
    }
}

void uint32_t_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint32_t *in0 = (uint32_t*)ptrs[0];
      double *out = (double*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (double)(*in0);
            in0 = (uint32_t*)((char*)in0 + stride[0]); 
            out = (double*)((char*)out + stride[1]); 
    }
}

void uint64_t_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint64_t *in0 = (uint64_t*)ptrs[0];
      float *out = (float*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (float)(*in0);
            in0 = (uint64_t*)((char*)in0 + stride[0]); 
            out = (float*)((char*)out + stride[1]); 
    }
}

void uint64_t_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint64_t *in0 = (uint64_t*)ptrs[0];
      double *out = (double*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (double)(*in0);
            in0 = (uint64_t*)((char*)in0 + stride[0]); 
            out = (double*)((char*)out + stride[1]); 
    }
}

void float_int8_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      float *in0 = (float*)ptrs[0];
      int8_t *out = (int8_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int8_t)(*in0);
            in0 = (float*)((char*)in0 + stride[0]); 
            out = (int8_t*)((char*)out + stride[1]); 
    }
}

void float_int16_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      float *in0 = (float*)ptrs[0];
      int16_t *out = (int16_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int16_t)(*in0);
            in0 = (float*)((char*)in0 + stride[0]); 
            out = (int16_t*)((char*)out + stride[1]); 
    }
}

void float_int32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      float *in0 = (float*)ptrs[0];
      int32_t *out = (int32_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int32_t)(*in0);
            in0 = (float*)((char*)in0 + stride[0]); 
            out = (int32_t*)((char*)out + stride[1]); 
    }
}

void float_int64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      float *in0 = (float*)ptrs[0];
      int64_t *out = (int64_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int64_t)(*in0);
            in0 = (float*)((char*)in0 + stride[0]); 
            out = (int64_t*)((char*)out + stride[1]); 
    }
}

void float_uint8_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      float *in0 = (float*)ptrs[0];
      uint8_t *out = (uint8_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint8_t)(*in0);
            in0 = (float*)((char*)in0 + stride[0]); 
            out = (uint8_t*)((char*)out + stride[1]); 
    }
}

void float_uint16_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      float *in0 = (float*)ptrs[0];
      uint16_t *out = (uint16_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint16_t)(*in0);
            in0 = (float*)((char*)in0 + stride[0]); 
            out = (uint16_t*)((char*)out + stride[1]); 
    }
}

void float_uint32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      float *in0 = (float*)ptrs[0];
      uint32_t *out = (uint32_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint32_t)(*in0);
            in0 = (float*)((char*)in0 + stride[0]); 
            out = (uint32_t*)((char*)out + stride[1]); 
    }
}

void float_uint64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      float *in0 = (float*)ptrs[0];
      uint64_t *out = (uint64_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint64_t)(*in0);
            in0 = (float*)((char*)in0 + stride[0]); 
            out = (uint64_t*)((char*)out + stride[1]); 
    }
}

void double_int8_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      double *in0 = (double*)ptrs[0];
      int8_t *out = (int8_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int8_t)(*in0);
            in0 = (double*)((char*)in0 + stride[0]); 
            out = (int8_t*)((char*)out + stride[1]); 
    }
}

void double_int16_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      double *in0 = (double*)ptrs[0];
      int16_t *out = (int16_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int16_t)(*in0);
            in0 = (double*)((char*)in0 + stride[0]); 
            out = (int16_t*)((char*)out + stride[1]); 
    }
}

void double_int32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      double *in0 = (double*)ptrs[0];
      int32_t *out = (int32_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int32_t)(*in0);
            in0 = (double*)((char*)in0 + stride[0]); 
            out = (int32_t*)((char*)out + stride[1]); 
    }
}

void double_int64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      double *in0 = (double*)ptrs[0];
      int64_t *out = (int64_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int64_t)(*in0);
            in0 = (double*)((char*)in0 + stride[0]); 
            out = (int64_t*)((char*)out + stride[1]); 
    }
}

void double_uint8_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      double *in0 = (double*)ptrs[0];
      uint8_t *out = (uint8_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint8_t)(*in0);
            in0 = (double*)((char*)in0 + stride[0]); 
            out = (uint8_t*)((char*)out + stride[1]); 
    }
}

void double_uint16_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      double *in0 = (double*)ptrs[0];
      uint16_t *out = (uint16_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint16_t)(*in0);
            in0 = (double*)((char*)in0 + stride[0]); 
            out = (uint16_t*)((char*)out + stride[1]); 
    }
}

void double_uint32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      double *in0 = (double*)ptrs[0];
      uint32_t *out = (uint32_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint32_t)(*in0);
            in0 = (double*)((char*)in0 + stride[0]); 
            out = (uint32_t*)((char*)out + stride[1]); 
    }
}

void double_uint64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      double *in0 = (double*)ptrs[0];
      uint64_t *out = (uint64_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint64_t)(*in0);
            in0 = (double*)((char*)in0 + stride[0]); 
            out = (uint64_t*)((char*)out + stride[1]); 
    }
}

void float_double_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      float *in0 = (float*)ptrs[0];
      double *out = (double*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (double)(*in0);
            in0 = (float*)((char*)in0 + stride[0]); 
            out = (double*)((char*)out + stride[1]); 
    }
}

void double_float_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      double *in0 = (double*)ptrs[0];
      float *out = (float*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (float)(*in0);
            in0 = (double*)((char*)in0 + stride[0]); 
            out = (float*)((char*)out + stride[1]); 
    }
}

void int8_t_int16_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int8_t *in0 = (int8_t*)ptrs[0];
      int16_t *out = (int16_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int16_t)(*in0);
            in0 = (int8_t*)((char*)in0 + stride[0]); 
            out = (int16_t*)((char*)out + stride[1]); 
    }
}

void int8_t_int32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int8_t *in0 = (int8_t*)ptrs[0];
      int32_t *out = (int32_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int32_t)(*in0);
            in0 = (int8_t*)((char*)in0 + stride[0]); 
            out = (int32_t*)((char*)out + stride[1]); 
    }
}

void int8_t_int64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int8_t *in0 = (int8_t*)ptrs[0];
      int64_t *out = (int64_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int64_t)(*in0);
            in0 = (int8_t*)((char*)in0 + stride[0]); 
            out = (int64_t*)((char*)out + stride[1]); 
    }
}

void int16_t_int32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int16_t *in0 = (int16_t*)ptrs[0];
      int32_t *out = (int32_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int32_t)(*in0);
            in0 = (int16_t*)((char*)in0 + stride[0]); 
            out = (int32_t*)((char*)out + stride[1]); 
    }
}

void int16_t_int64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int16_t *in0 = (int16_t*)ptrs[0];
      int64_t *out = (int64_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int64_t)(*in0);
            in0 = (int16_t*)((char*)in0 + stride[0]); 
            out = (int64_t*)((char*)out + stride[1]); 
    }
}

void int32_t_int64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int32_t *in0 = (int32_t*)ptrs[0];
      int64_t *out = (int64_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int64_t)(*in0);
            in0 = (int32_t*)((char*)in0 + stride[0]); 
            out = (int64_t*)((char*)out + stride[1]); 
    }
}

void uint8_t_uint16_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint8_t *in0 = (uint8_t*)ptrs[0];
      uint16_t *out = (uint16_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint16_t)(*in0);
            in0 = (uint8_t*)((char*)in0 + stride[0]); 
            out = (uint16_t*)((char*)out + stride[1]); 
    }
}

void uint8_t_uint32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint8_t *in0 = (uint8_t*)ptrs[0];
      uint32_t *out = (uint32_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint32_t)(*in0);
            in0 = (uint8_t*)((char*)in0 + stride[0]); 
            out = (uint32_t*)((char*)out + stride[1]); 
    }
}

void uint8_t_uint64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint8_t *in0 = (uint8_t*)ptrs[0];
      uint64_t *out = (uint64_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint64_t)(*in0);
            in0 = (uint8_t*)((char*)in0 + stride[0]); 
            out = (uint64_t*)((char*)out + stride[1]); 
    }
}

void uint16_t_uint32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint16_t *in0 = (uint16_t*)ptrs[0];
      uint32_t *out = (uint32_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint32_t)(*in0);
            in0 = (uint16_t*)((char*)in0 + stride[0]); 
            out = (uint32_t*)((char*)out + stride[1]); 
    }
}

void uint16_t_uint64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint16_t *in0 = (uint16_t*)ptrs[0];
      uint64_t *out = (uint64_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint64_t)(*in0);
            in0 = (uint16_t*)((char*)in0 + stride[0]); 
            out = (uint64_t*)((char*)out + stride[1]); 
    }
}

void uint32_t_uint64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint32_t *in0 = (uint32_t*)ptrs[0];
      uint64_t *out = (uint64_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint64_t)(*in0);
            in0 = (uint32_t*)((char*)in0 + stride[0]); 
            out = (uint64_t*)((char*)out + stride[1]); 
    }
}

void int8_t_uint8_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int8_t *in0 = (int8_t*)ptrs[0];
      uint8_t *out = (uint8_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint8_t)(*in0);
            in0 = (int8_t*)((char*)in0 + stride[0]); 
            out = (uint8_t*)((char*)out + stride[1]); 
    }
}

void int16_t_uint16_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int16_t *in0 = (int16_t*)ptrs[0];
      uint16_t *out = (uint16_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint16_t)(*in0);
            in0 = (int16_t*)((char*)in0 + stride[0]); 
            out = (uint16_t*)((char*)out + stride[1]); 
    }
}

void int32_t_uint32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int32_t *in0 = (int32_t*)ptrs[0];
      uint32_t *out = (uint32_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint32_t)(*in0);
            in0 = (int32_t*)((char*)in0 + stride[0]); 
            out = (uint32_t*)((char*)out + stride[1]); 
    }
}

void int64_t_uint64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      int64_t *in0 = (int64_t*)ptrs[0];
      uint64_t *out = (uint64_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (uint64_t)(*in0);
            in0 = (int64_t*)((char*)in0 + stride[0]); 
            out = (uint64_t*)((char*)out + stride[1]); 
    }
}

void uint8_t_int8_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint8_t *in0 = (uint8_t*)ptrs[0];
      int8_t *out = (int8_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int8_t)(*in0);
            in0 = (uint8_t*)((char*)in0 + stride[0]); 
            out = (int8_t*)((char*)out + stride[1]); 
    }
}

void uint16_t_int16_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint16_t *in0 = (uint16_t*)ptrs[0];
      int16_t *out = (int16_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int16_t)(*in0);
            in0 = (uint16_t*)((char*)in0 + stride[0]); 
            out = (int16_t*)((char*)out + stride[1]); 
    }
}

void uint32_t_int32_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint32_t *in0 = (uint32_t*)ptrs[0];
      int32_t *out = (int32_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int32_t)(*in0);
            in0 = (uint32_t*)((char*)in0 + stride[0]); 
            out = (int32_t*)((char*)out + stride[1]); 
    }
}

void uint64_t_int64_t_kernel_cast(char **ptrs, const int64_t *stride, int64_t n) {
      uint64_t *in0 = (uint64_t*)ptrs[0];
      int64_t *out = (int64_t*)ptrs[1];
      for (int64_t i = 0; i < n; i++) {
           *out = (int64_t)(*in0);
            in0 = (uint64_t*)((char*)in0 + stride[0]); 
            out = (int64_t*)((char*)out + stride[1]); 
    }
}


void i8_kernel_add(char **ptrs, const int64_t *stride, int64_t n) {
    int8_t *in0 = (int8_t*)ptrs[0];
    int8_t *in1 = (int8_t*)ptrs[1];
    int8_t *out = (int8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 + *in1;
        in0 = (int8_t*)((char*)in0 + stride[0]);
        in1 = (int8_t*)((char*)in1 + stride[1]);
        out = (int8_t*)((char*)out + stride[2]);
    }
}

void i16_kernel_add(char **ptrs, const int64_t *stride, int64_t n) {
    int16_t *in0 = (int16_t*)ptrs[0];
    int16_t *in1 = (int16_t*)ptrs[1];
    int16_t *out = (int16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 + *in1;
        in0 = (int16_t*)((char*)in0 + stride[0]);
        in1 = (int16_t*)((char*)in1 + stride[1]);
        out = (int16_t*)((char*)out + stride[2]);
    }
}

void i32_kernel_add(char **ptrs, const int64_t *stride, int64_t n) {
    int32_t *in0 = (int32_t*)ptrs[0];
    int32_t *in1 = (int32_t*)ptrs[1];
    int32_t *out = (int32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 + *in1;
        in0 = (int32_t*)((char*)in0 + stride[0]);
        in1 = (int32_t*)((char*)in1 + stride[1]);
        out = (int32_t*)((char*)out + stride[2]);
    }
}

void i64_kernel_add(char **ptrs, const int64_t *stride, int64_t n) {
    int64_t *in0 = (int64_t*)ptrs[0];
    int64_t *in1 = (int64_t*)ptrs[1];
    int64_t *out = (int64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 + *in1;
        in0 = (int64_t*)((char*)in0 + stride[0]);
        in1 = (int64_t*)((char*)in1 + stride[1]);
        out = (int64_t*)((char*)out + stride[2]);
    }
}

void u8_kernel_add(char **ptrs, const int64_t *stride, int64_t n) {
    uint8_t *in0 = (uint8_t*)ptrs[0];
    uint8_t *in1 = (uint8_t*)ptrs[1];
    uint8_t *out = (uint8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 + *in1;
        in0 = (uint8_t*)((char*)in0 + stride[0]);
        in1 = (uint8_t*)((char*)in1 + stride[1]);
        out = (uint8_t*)((char*)out + stride[2]);
    }
}

void u16_kernel_add(char **ptrs, const int64_t *stride, int64_t n) {
    uint16_t *in0 = (uint16_t*)ptrs[0];
    uint16_t *in1 = (uint16_t*)ptrs[1];
    uint16_t *out = (uint16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 + *in1;
        in0 = (uint16_t*)((char*)in0 + stride[0]);
        in1 = (uint16_t*)((char*)in1 + stride[1]);
        out = (uint16_t*)((char*)out + stride[2]);
    }
}

void u32_kernel_add(char **ptrs, const int64_t *stride, int64_t n) {
    uint32_t *in0 = (uint32_t*)ptrs[0];
    uint32_t *in1 = (uint32_t*)ptrs[1];
    uint32_t *out = (uint32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 + *in1;
        in0 = (uint32_t*)((char*)in0 + stride[0]);
        in1 = (uint32_t*)((char*)in1 + stride[1]);
        out = (uint32_t*)((char*)out + stride[2]);
    }
}

void u64_kernel_add(char **ptrs, const int64_t *stride, int64_t n) {
    uint64_t *in0 = (uint64_t*)ptrs[0];
    uint64_t *in1 = (uint64_t*)ptrs[1];
    uint64_t *out = (uint64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 + *in1;
        in0 = (uint64_t*)((char*)in0 + stride[0]);
        in1 = (uint64_t*)((char*)in1 + stride[1]);
        out = (uint64_t*)((char*)out + stride[2]);
    }
}

void f32_kernel_add(char **ptrs, const int64_t *stride, int64_t n) {
    float *in0 = (float*)ptrs[0];
    float *in1 = (float*)ptrs[1];
    float *out = (float*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 + *in1;
        in0 = (float*)((char*)in0 + stride[0]);
        in1 = (float*)((char*)in1 + stride[1]);
        out = (float*)((char*)out + stride[2]);
    }
}

void f64_kernel_add(char **ptrs, const int64_t *stride, int64_t n) {
    double *in0 = (double*)ptrs[0];
    double *in1 = (double*)ptrs[1];
    double *out = (double*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 + *in1;
        in0 = (double*)((char*)in0 + stride[0]);
        in1 = (double*)((char*)in1 + stride[1]);
        out = (double*)((char*)out + stride[2]);
    }
}

void i8_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n) {
    int8_t *in0 = (int8_t*)ptrs[0];
    int8_t *in1 = (int8_t*)ptrs[1];
    int8_t *out = (int8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 - *in1;
        in0 = (int8_t*)((char*)in0 + stride[0]);
        in1 = (int8_t*)((char*)in1 + stride[1]);
        out = (int8_t*)((char*)out + stride[2]);
    }
}

void i16_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n) {
    int16_t *in0 = (int16_t*)ptrs[0];
    int16_t *in1 = (int16_t*)ptrs[1];
    int16_t *out = (int16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 - *in1;
        in0 = (int16_t*)((char*)in0 + stride[0]);
        in1 = (int16_t*)((char*)in1 + stride[1]);
        out = (int16_t*)((char*)out + stride[2]);
    }
}

void i32_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n) {
    int32_t *in0 = (int32_t*)ptrs[0];
    int32_t *in1 = (int32_t*)ptrs[1];
    int32_t *out = (int32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 - *in1;
        in0 = (int32_t*)((char*)in0 + stride[0]);
        in1 = (int32_t*)((char*)in1 + stride[1]);
        out = (int32_t*)((char*)out + stride[2]);
    }
}

void i64_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n) {
    int64_t *in0 = (int64_t*)ptrs[0];
    int64_t *in1 = (int64_t*)ptrs[1];
    int64_t *out = (int64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 - *in1;
        in0 = (int64_t*)((char*)in0 + stride[0]);
        in1 = (int64_t*)((char*)in1 + stride[1]);
        out = (int64_t*)((char*)out + stride[2]);
    }
}

void u8_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n) {
    uint8_t *in0 = (uint8_t*)ptrs[0];
    uint8_t *in1 = (uint8_t*)ptrs[1];
    uint8_t *out = (uint8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 - *in1;
        in0 = (uint8_t*)((char*)in0 + stride[0]);
        in1 = (uint8_t*)((char*)in1 + stride[1]);
        out = (uint8_t*)((char*)out + stride[2]);
    }
}

void u16_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n) {
    uint16_t *in0 = (uint16_t*)ptrs[0];
    uint16_t *in1 = (uint16_t*)ptrs[1];
    uint16_t *out = (uint16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 - *in1;
        in0 = (uint16_t*)((char*)in0 + stride[0]);
        in1 = (uint16_t*)((char*)in1 + stride[1]);
        out = (uint16_t*)((char*)out + stride[2]);
    }
}

void u32_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n) {
    uint32_t *in0 = (uint32_t*)ptrs[0];
    uint32_t *in1 = (uint32_t*)ptrs[1];
    uint32_t *out = (uint32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 - *in1;
        in0 = (uint32_t*)((char*)in0 + stride[0]);
        in1 = (uint32_t*)((char*)in1 + stride[1]);
        out = (uint32_t*)((char*)out + stride[2]);
    }
}

void u64_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n) {
    uint64_t *in0 = (uint64_t*)ptrs[0];
    uint64_t *in1 = (uint64_t*)ptrs[1];
    uint64_t *out = (uint64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 - *in1;
        in0 = (uint64_t*)((char*)in0 + stride[0]);
        in1 = (uint64_t*)((char*)in1 + stride[1]);
        out = (uint64_t*)((char*)out + stride[2]);
    }
}

void f32_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n) {
    float *in0 = (float*)ptrs[0];
    float *in1 = (float*)ptrs[1];
    float *out = (float*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 - *in1;
        in0 = (float*)((char*)in0 + stride[0]);
        in1 = (float*)((char*)in1 + stride[1]);
        out = (float*)((char*)out + stride[2]);
    }
}

void f64_kernel_subtract(char **ptrs, const int64_t *stride, int64_t n) {
    double *in0 = (double*)ptrs[0];
    double *in1 = (double*)ptrs[1];
    double *out = (double*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 - *in1;
        in0 = (double*)((char*)in0 + stride[0]);
        in1 = (double*)((char*)in1 + stride[1]);
        out = (double*)((char*)out + stride[2]);
    }
}

void i8_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n) {
    int8_t *in0 = (int8_t*)ptrs[0];
    int8_t *in1 = (int8_t*)ptrs[1];
    int8_t *out = (int8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 * *in1;
        in0 = (int8_t*)((char*)in0 + stride[0]);
        in1 = (int8_t*)((char*)in1 + stride[1]);
        out = (int8_t*)((char*)out + stride[2]);
    }
}

void i16_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n) {
    int16_t *in0 = (int16_t*)ptrs[0];
    int16_t *in1 = (int16_t*)ptrs[1];
    int16_t *out = (int16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 * *in1;
        in0 = (int16_t*)((char*)in0 + stride[0]);
        in1 = (int16_t*)((char*)in1 + stride[1]);
        out = (int16_t*)((char*)out + stride[2]);
    }
}

void i32_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n) {
    int32_t *in0 = (int32_t*)ptrs[0];
    int32_t *in1 = (int32_t*)ptrs[1];
    int32_t *out = (int32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 * *in1;
        in0 = (int32_t*)((char*)in0 + stride[0]);
        in1 = (int32_t*)((char*)in1 + stride[1]);
        out = (int32_t*)((char*)out + stride[2]);
    }
}

void i64_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n) {
    int64_t *in0 = (int64_t*)ptrs[0];
    int64_t *in1 = (int64_t*)ptrs[1];
    int64_t *out = (int64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 * *in1;
        in0 = (int64_t*)((char*)in0 + stride[0]);
        in1 = (int64_t*)((char*)in1 + stride[1]);
        out = (int64_t*)((char*)out + stride[2]);
    }
}

void u8_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n) {
    uint8_t *in0 = (uint8_t*)ptrs[0];
    uint8_t *in1 = (uint8_t*)ptrs[1];
    uint8_t *out = (uint8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 * *in1;
        in0 = (uint8_t*)((char*)in0 + stride[0]);
        in1 = (uint8_t*)((char*)in1 + stride[1]);
        out = (uint8_t*)((char*)out + stride[2]);
    }
}

void u16_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n) {
    uint16_t *in0 = (uint16_t*)ptrs[0];
    uint16_t *in1 = (uint16_t*)ptrs[1];
    uint16_t *out = (uint16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 * *in1;
        in0 = (uint16_t*)((char*)in0 + stride[0]);
        in1 = (uint16_t*)((char*)in1 + stride[1]);
        out = (uint16_t*)((char*)out + stride[2]);
    }
}

void u32_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n) {
    uint32_t *in0 = (uint32_t*)ptrs[0];
    uint32_t *in1 = (uint32_t*)ptrs[1];
    uint32_t *out = (uint32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 * *in1;
        in0 = (uint32_t*)((char*)in0 + stride[0]);
        in1 = (uint32_t*)((char*)in1 + stride[1]);
        out = (uint32_t*)((char*)out + stride[2]);
    }
}

void u64_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n) {
    uint64_t *in0 = (uint64_t*)ptrs[0];
    uint64_t *in1 = (uint64_t*)ptrs[1];
    uint64_t *out = (uint64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 * *in1;
        in0 = (uint64_t*)((char*)in0 + stride[0]);
        in1 = (uint64_t*)((char*)in1 + stride[1]);
        out = (uint64_t*)((char*)out + stride[2]);
    }
}

void f32_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n) {
    float *in0 = (float*)ptrs[0];
    float *in1 = (float*)ptrs[1];
    float *out = (float*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 * *in1;
        in0 = (float*)((char*)in0 + stride[0]);
        in1 = (float*)((char*)in1 + stride[1]);
        out = (float*)((char*)out + stride[2]);
    }
}

void f64_kernel_multiply(char **ptrs, const int64_t *stride, int64_t n) {
    double *in0 = (double*)ptrs[0];
    double *in1 = (double*)ptrs[1];
    double *out = (double*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 * *in1;
        in0 = (double*)((char*)in0 + stride[0]);
        in1 = (double*)((char*)in1 + stride[1]);
        out = (double*)((char*)out + stride[2]);
    }
}

void i8_kernel_divide(char **ptrs, const int64_t *stride, int64_t n) {
    int8_t *in0 = (int8_t*)ptrs[0];
    int8_t *in1 = (int8_t*)ptrs[1];
    int8_t *out = (int8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 / *in1;
        in0 = (int8_t*)((char*)in0 + stride[0]);
        in1 = (int8_t*)((char*)in1 + stride[1]);
        out = (int8_t*)((char*)out + stride[2]);
    }
}

void i16_kernel_divide(char **ptrs, const int64_t *stride, int64_t n) {
    int16_t *in0 = (int16_t*)ptrs[0];
    int16_t *in1 = (int16_t*)ptrs[1];
    int16_t *out = (int16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 / *in1;
        in0 = (int16_t*)((char*)in0 + stride[0]);
        in1 = (int16_t*)((char*)in1 + stride[1]);
        out = (int16_t*)((char*)out + stride[2]);
    }
}

void i32_kernel_divide(char **ptrs, const int64_t *stride, int64_t n) {
    int32_t *in0 = (int32_t*)ptrs[0];
    int32_t *in1 = (int32_t*)ptrs[1];
    int32_t *out = (int32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 / *in1;
        in0 = (int32_t*)((char*)in0 + stride[0]);
        in1 = (int32_t*)((char*)in1 + stride[1]);
        out = (int32_t*)((char*)out + stride[2]);
    }
}

void i64_kernel_divide(char **ptrs, const int64_t *stride, int64_t n) {
    int64_t *in0 = (int64_t*)ptrs[0];
    int64_t *in1 = (int64_t*)ptrs[1];
    int64_t *out = (int64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 / *in1;
        in0 = (int64_t*)((char*)in0 + stride[0]);
        in1 = (int64_t*)((char*)in1 + stride[1]);
        out = (int64_t*)((char*)out + stride[2]);
    }
}

void u8_kernel_divide(char **ptrs, const int64_t *stride, int64_t n) {
    uint8_t *in0 = (uint8_t*)ptrs[0];
    uint8_t *in1 = (uint8_t*)ptrs[1];
    uint8_t *out = (uint8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 / *in1;
        in0 = (uint8_t*)((char*)in0 + stride[0]);
        in1 = (uint8_t*)((char*)in1 + stride[1]);
        out = (uint8_t*)((char*)out + stride[2]);
    }
}

void u16_kernel_divide(char **ptrs, const int64_t *stride, int64_t n) {
    uint16_t *in0 = (uint16_t*)ptrs[0];
    uint16_t *in1 = (uint16_t*)ptrs[1];
    uint16_t *out = (uint16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 / *in1;
        in0 = (uint16_t*)((char*)in0 + stride[0]);
        in1 = (uint16_t*)((char*)in1 + stride[1]);
        out = (uint16_t*)((char*)out + stride[2]);
    }
}

void u32_kernel_divide(char **ptrs, const int64_t *stride, int64_t n) {
    uint32_t *in0 = (uint32_t*)ptrs[0];
    uint32_t *in1 = (uint32_t*)ptrs[1];
    uint32_t *out = (uint32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 / *in1;
        in0 = (uint32_t*)((char*)in0 + stride[0]);
        in1 = (uint32_t*)((char*)in1 + stride[1]);
        out = (uint32_t*)((char*)out + stride[2]);
    }
}

void u64_kernel_divide(char **ptrs, const int64_t *stride, int64_t n) {
    uint64_t *in0 = (uint64_t*)ptrs[0];
    uint64_t *in1 = (uint64_t*)ptrs[1];
    uint64_t *out = (uint64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 / *in1;
        in0 = (uint64_t*)((char*)in0 + stride[0]);
        in1 = (uint64_t*)((char*)in1 + stride[1]);
        out = (uint64_t*)((char*)out + stride[2]);
    }
}

void f32_kernel_divide(char **ptrs, const int64_t *stride, int64_t n) {
    float *in0 = (float*)ptrs[0];
    float *in1 = (float*)ptrs[1];
    float *out = (float*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 / *in1;
        in0 = (float*)((char*)in0 + stride[0]);
        in1 = (float*)((char*)in1 + stride[1]);
        out = (float*)((char*)out + stride[2]);
    }
}

void f64_kernel_divide(char **ptrs, const int64_t *stride, int64_t n) {
    double *in0 = (double*)ptrs[0];
    double *in1 = (double*)ptrs[1];
    double *out = (double*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 / *in1;
        in0 = (double*)((char*)in0 + stride[0]);
        in1 = (double*)((char*)in1 + stride[1]);
        out = (double*)((char*)out + stride[2]);
    }
}

void i8_kernel_equality(char **ptrs, const int64_t *stride, int64_t n) {
    int8_t *in0 = (int8_t*)ptrs[0];
    int8_t *in1 = (int8_t*)ptrs[1];
    int8_t *out = (int8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 == *in1;
        in0 = (int8_t*)((char*)in0 + stride[0]);
        in1 = (int8_t*)((char*)in1 + stride[1]);
        out = (int8_t*)((char*)out + stride[2]);
    }
}

void i16_kernel_equality(char **ptrs, const int64_t *stride, int64_t n) {
    int16_t *in0 = (int16_t*)ptrs[0];
    int16_t *in1 = (int16_t*)ptrs[1];
    int16_t *out = (int16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 == *in1;
        in0 = (int16_t*)((char*)in0 + stride[0]);
        in1 = (int16_t*)((char*)in1 + stride[1]);
        out = (int16_t*)((char*)out + stride[2]);
    }
}

void i32_kernel_equality(char **ptrs, const int64_t *stride, int64_t n) {
    int32_t *in0 = (int32_t*)ptrs[0];
    int32_t *in1 = (int32_t*)ptrs[1];
    int32_t *out = (int32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 == *in1;
        in0 = (int32_t*)((char*)in0 + stride[0]);
        in1 = (int32_t*)((char*)in1 + stride[1]);
        out = (int32_t*)((char*)out + stride[2]);
    }
}

void i64_kernel_equality(char **ptrs, const int64_t *stride, int64_t n) {
    int64_t *in0 = (int64_t*)ptrs[0];
    int64_t *in1 = (int64_t*)ptrs[1];
    int64_t *out = (int64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 == *in1;
        in0 = (int64_t*)((char*)in0 + stride[0]);
        in1 = (int64_t*)((char*)in1 + stride[1]);
        out = (int64_t*)((char*)out + stride[2]);
    }
}

void u8_kernel_equality(char **ptrs, const int64_t *stride, int64_t n) {
    uint8_t *in0 = (uint8_t*)ptrs[0];
    uint8_t *in1 = (uint8_t*)ptrs[1];
    uint8_t *out = (uint8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 == *in1;
        in0 = (uint8_t*)((char*)in0 + stride[0]);
        in1 = (uint8_t*)((char*)in1 + stride[1]);
        out = (uint8_t*)((char*)out + stride[2]);
    }
}

void u16_kernel_equality(char **ptrs, const int64_t *stride, int64_t n) {
    uint16_t *in0 = (uint16_t*)ptrs[0];
    uint16_t *in1 = (uint16_t*)ptrs[1];
    uint16_t *out = (uint16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 == *in1;
        in0 = (uint16_t*)((char*)in0 + stride[0]);
        in1 = (uint16_t*)((char*)in1 + stride[1]);
        out = (uint16_t*)((char*)out + stride[2]);
    }
}

void u32_kernel_equality(char **ptrs, const int64_t *stride, int64_t n) {
    uint32_t *in0 = (uint32_t*)ptrs[0];
    uint32_t *in1 = (uint32_t*)ptrs[1];
    uint32_t *out = (uint32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 == *in1;
        in0 = (uint32_t*)((char*)in0 + stride[0]);
        in1 = (uint32_t*)((char*)in1 + stride[1]);
        out = (uint32_t*)((char*)out + stride[2]);
    }
}

void u64_kernel_equality(char **ptrs, const int64_t *stride, int64_t n) {
    uint64_t *in0 = (uint64_t*)ptrs[0];
    uint64_t *in1 = (uint64_t*)ptrs[1];
    uint64_t *out = (uint64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 == *in1;
        in0 = (uint64_t*)((char*)in0 + stride[0]);
        in1 = (uint64_t*)((char*)in1 + stride[1]);
        out = (uint64_t*)((char*)out + stride[2]);
    }
}

void f32_kernel_equality(char **ptrs, const int64_t *stride, int64_t n) {
    float *in0 = (float*)ptrs[0];
    float *in1 = (float*)ptrs[1];
    float *out = (float*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 == *in1;
        in0 = (float*)((char*)in0 + stride[0]);
        in1 = (float*)((char*)in1 + stride[1]);
        out = (float*)((char*)out + stride[2]);
    }
}

void f64_kernel_equality(char **ptrs, const int64_t *stride, int64_t n) {
    double *in0 = (double*)ptrs[0];
    double *in1 = (double*)ptrs[1];
    double *out = (double*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 == *in1;
        in0 = (double*)((char*)in0 + stride[0]);
        in1 = (double*)((char*)in1 + stride[1]);
        out = (double*)((char*)out + stride[2]);
    }
}

void bool_kernel_equality(char **ptrs, const int64_t *stride, int64_t n) {
    bool *in0 = (bool*)ptrs[0];
    bool *in1 = (bool*)ptrs[1];
    bool *out = (bool*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 == *in1;
        in0 = (bool*)((char*)in0 + stride[0]);
        in1 = (bool*)((char*)in1 + stride[1]);
        out = (bool*)((char*)out + stride[2]);
    }
}

void i8_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n) {
    int8_t *in0 = (int8_t*)ptrs[0];
    int8_t *in1 = (int8_t*)ptrs[1];
    int8_t *out = (int8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 > *in1;
        in0 = (int8_t*)((char*)in0 + stride[0]);
        in1 = (int8_t*)((char*)in1 + stride[1]);
        out = (int8_t*)((char*)out + stride[2]);
    }
}

void i16_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n) {
    int16_t *in0 = (int16_t*)ptrs[0];
    int16_t *in1 = (int16_t*)ptrs[1];
    int16_t *out = (int16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 > *in1;
        in0 = (int16_t*)((char*)in0 + stride[0]);
        in1 = (int16_t*)((char*)in1 + stride[1]);
        out = (int16_t*)((char*)out + stride[2]);
    }
}

void i32_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n) {
    int32_t *in0 = (int32_t*)ptrs[0];
    int32_t *in1 = (int32_t*)ptrs[1];
    int32_t *out = (int32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 > *in1;
        in0 = (int32_t*)((char*)in0 + stride[0]);
        in1 = (int32_t*)((char*)in1 + stride[1]);
        out = (int32_t*)((char*)out + stride[2]);
    }
}

void i64_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n) {
    int64_t *in0 = (int64_t*)ptrs[0];
    int64_t *in1 = (int64_t*)ptrs[1];
    int64_t *out = (int64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 > *in1;
        in0 = (int64_t*)((char*)in0 + stride[0]);
        in1 = (int64_t*)((char*)in1 + stride[1]);
        out = (int64_t*)((char*)out + stride[2]);
    }
}

void u8_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n) {
    uint8_t *in0 = (uint8_t*)ptrs[0];
    uint8_t *in1 = (uint8_t*)ptrs[1];
    uint8_t *out = (uint8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 > *in1;
        in0 = (uint8_t*)((char*)in0 + stride[0]);
        in1 = (uint8_t*)((char*)in1 + stride[1]);
        out = (uint8_t*)((char*)out + stride[2]);
    }
}

void u16_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n) {
    uint16_t *in0 = (uint16_t*)ptrs[0];
    uint16_t *in1 = (uint16_t*)ptrs[1];
    uint16_t *out = (uint16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 > *in1;
        in0 = (uint16_t*)((char*)in0 + stride[0]);
        in1 = (uint16_t*)((char*)in1 + stride[1]);
        out = (uint16_t*)((char*)out + stride[2]);
    }
}

void u32_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n) {
    uint32_t *in0 = (uint32_t*)ptrs[0];
    uint32_t *in1 = (uint32_t*)ptrs[1];
    uint32_t *out = (uint32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 > *in1;
        in0 = (uint32_t*)((char*)in0 + stride[0]);
        in1 = (uint32_t*)((char*)in1 + stride[1]);
        out = (uint32_t*)((char*)out + stride[2]);
    }
}

void u64_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n) {
    uint64_t *in0 = (uint64_t*)ptrs[0];
    uint64_t *in1 = (uint64_t*)ptrs[1];
    uint64_t *out = (uint64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 > *in1;
        in0 = (uint64_t*)((char*)in0 + stride[0]);
        in1 = (uint64_t*)((char*)in1 + stride[1]);
        out = (uint64_t*)((char*)out + stride[2]);
    }
}

void f32_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n) {
    float *in0 = (float*)ptrs[0];
    float *in1 = (float*)ptrs[1];
    float *out = (float*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 > *in1;
        in0 = (float*)((char*)in0 + stride[0]);
        in1 = (float*)((char*)in1 + stride[1]);
        out = (float*)((char*)out + stride[2]);
    }
}

void f64_kernel_greater_than(char **ptrs, const int64_t *stride, int64_t n) {
    double *in0 = (double*)ptrs[0];
    double *in1 = (double*)ptrs[1];
    double *out = (double*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 > *in1;
        in0 = (double*)((char*)in0 + stride[0]);
        in1 = (double*)((char*)in1 + stride[1]);
        out = (double*)((char*)out + stride[2]);
    }
}

void i8_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n) {
    int8_t *in0 = (int8_t*)ptrs[0];
    int8_t *in1 = (int8_t*)ptrs[1];
    int8_t *out = (int8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 < *in1;
        in0 = (int8_t*)((char*)in0 + stride[0]);
        in1 = (int8_t*)((char*)in1 + stride[1]);
        out = (int8_t*)((char*)out + stride[2]);
    }
}

void i16_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n) {
    int16_t *in0 = (int16_t*)ptrs[0];
    int16_t *in1 = (int16_t*)ptrs[1];
    int16_t *out = (int16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 < *in1;
        in0 = (int16_t*)((char*)in0 + stride[0]);
        in1 = (int16_t*)((char*)in1 + stride[1]);
        out = (int16_t*)((char*)out + stride[2]);
    }
}

void i32_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n) {
    int32_t *in0 = (int32_t*)ptrs[0];
    int32_t *in1 = (int32_t*)ptrs[1];
    int32_t *out = (int32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 < *in1;
        in0 = (int32_t*)((char*)in0 + stride[0]);
        in1 = (int32_t*)((char*)in1 + stride[1]);
        out = (int32_t*)((char*)out + stride[2]);
    }
}

void i64_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n) {
    int64_t *in0 = (int64_t*)ptrs[0];
    int64_t *in1 = (int64_t*)ptrs[1];
    int64_t *out = (int64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 < *in1;
        in0 = (int64_t*)((char*)in0 + stride[0]);
        in1 = (int64_t*)((char*)in1 + stride[1]);
        out = (int64_t*)((char*)out + stride[2]);
    }
}

void u8_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n) {
    uint8_t *in0 = (uint8_t*)ptrs[0];
    uint8_t *in1 = (uint8_t*)ptrs[1];
    uint8_t *out = (uint8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 < *in1;
        in0 = (uint8_t*)((char*)in0 + stride[0]);
        in1 = (uint8_t*)((char*)in1 + stride[1]);
        out = (uint8_t*)((char*)out + stride[2]);
    }
}

void u16_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n) {
    uint16_t *in0 = (uint16_t*)ptrs[0];
    uint16_t *in1 = (uint16_t*)ptrs[1];
    uint16_t *out = (uint16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 < *in1;
        in0 = (uint16_t*)((char*)in0 + stride[0]);
        in1 = (uint16_t*)((char*)in1 + stride[1]);
        out = (uint16_t*)((char*)out + stride[2]);
    }
}

void u32_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n) {
    uint32_t *in0 = (uint32_t*)ptrs[0];
    uint32_t *in1 = (uint32_t*)ptrs[1];
    uint32_t *out = (uint32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 < *in1;
        in0 = (uint32_t*)((char*)in0 + stride[0]);
        in1 = (uint32_t*)((char*)in1 + stride[1]);
        out = (uint32_t*)((char*)out + stride[2]);
    }
}

void u64_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n) {
    uint64_t *in0 = (uint64_t*)ptrs[0];
    uint64_t *in1 = (uint64_t*)ptrs[1];
    uint64_t *out = (uint64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 < *in1;
        in0 = (uint64_t*)((char*)in0 + stride[0]);
        in1 = (uint64_t*)((char*)in1 + stride[1]);
        out = (uint64_t*)((char*)out + stride[2]);
    }
}

void f32_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n) {
    float *in0 = (float*)ptrs[0];
    float *in1 = (float*)ptrs[1];
    float *out = (float*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 < *in1;
        in0 = (float*)((char*)in0 + stride[0]);
        in1 = (float*)((char*)in1 + stride[1]);
        out = (float*)((char*)out + stride[2]);
    }
}

void f64_kernel_less_than(char **ptrs, const int64_t *stride, int64_t n) {
    double *in0 = (double*)ptrs[0];
    double *in1 = (double*)ptrs[1];
    double *out = (double*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 < *in1;
        in0 = (double*)((char*)in0 + stride[0]);
        in1 = (double*)((char*)in1 + stride[1]);
        out = (double*)((char*)out + stride[2]);
    }
}

void f32_kernel_square_root(char **ptrs, const int64_t *stride, int64_t n) {
    float *in0 = (float*)ptrs[0];
    float *out = (float*)ptrs[1];
    for (int64_t i = 0; i < n; i++) {
        *out = sqrtf(*in0);
        in0 = (float*)((char*)in0 + stride[0]);
        out = (float*)((char*)out + stride[1]);
    }
}

void f64_kernel_square_root(char **ptrs, const int64_t *stride, int64_t n) {
    double *in0 = (double*)ptrs[0];
    double *out = (double*)ptrs[1];
    for (int64_t i = 0; i < n; i++) {
        *out = sqrt(*in0);
        in0 = (double*)((char*)in0 + stride[0]);
        out = (double*)((char*)out + stride[1]);
    }
}

void i8_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n) {
    int8_t *in0 = (int8_t*)ptrs[0];
    int8_t *out = (int8_t*)ptrs[1];
    for (int64_t i = 0; i < n; i++) {
        *out = abs(*in0);
        in0 = (int8_t*)((char*)in0 + stride[0]);
        out = (int8_t*)((char*)out + stride[1]);
    }
}

void i16_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n) {
    int16_t *in0 = (int16_t*)ptrs[0];
    int16_t *out = (int16_t*)ptrs[1];
    for (int64_t i = 0; i < n; i++) {
        *out = abs(*in0);
        in0 = (int16_t*)((char*)in0 + stride[0]);
        out = (int16_t*)((char*)out + stride[1]);
    }
}

void i32_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n) {
    int32_t *in0 = (int32_t*)ptrs[0];
    int32_t *out = (int32_t*)ptrs[1];
    for (int64_t i = 0; i < n; i++) {
        *out = abs(*in0);
        in0 = (int32_t*)((char*)in0 + stride[0]);
        out = (int32_t*)((char*)out + stride[1]);
    }
}

void i64_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n) {
    int64_t *in0 = (int64_t*)ptrs[0];
    int64_t *out = (int64_t*)ptrs[1];
    for (int64_t i = 0; i < n; i++) {
        *out = llabs(*in0);
        in0 = (int64_t*)((char*)in0 + stride[0]);
        out = (int64_t*)((char*)out + stride[1]);
    }
}

void f32_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n) {
    float *in0 = (float*)ptrs[0];
    float *out = (float*)ptrs[1];
    for (int64_t i = 0; i < n; i++) {
        *out = fabsf(*in0);
        in0 = (float*)((char*)in0 + stride[0]);
        out = (float*)((char*)out + stride[1]);
    }
}

void f64_kernel_absolute_value(char **ptrs, const int64_t *stride, int64_t n) {
    double *in0 = (double*)ptrs[0];
    double *out = (double*)ptrs[1];
    for (int64_t i = 0; i < n; i++) {
        *out = fabs(*in0);
        in0 = (double*)((char*)in0 + stride[0]);
        out = (double*)((char*)out + stride[1]);
    }
}

void i8_kernel_negate(char **ptrs, const int64_t *stride, int64_t n) {
    int8_t *in0 = (int8_t*)ptrs[0];
    int8_t *out = (int8_t*)ptrs[1];
    for (int64_t i = 0; i < n; i++) {
        *out = -*in0;
        in0 = (int8_t*)((char*)in0 + stride[0]);
        out = (int8_t*)((char*)out + stride[1]);
    }
}

void i16_kernel_negate(char **ptrs, const int64_t *stride, int64_t n) {
    int16_t *in0 = (int16_t*)ptrs[0];
    int16_t *out = (int16_t*)ptrs[1];
    for (int64_t i = 0; i < n; i++) {
        *out = -*in0;
        in0 = (int16_t*)((char*)in0 + stride[0]);
        out = (int16_t*)((char*)out + stride[1]);
    }
}

void i32_kernel_negate(char **ptrs, const int64_t *stride, int64_t n) {
    int32_t *in0 = (int32_t*)ptrs[0];
    int32_t *out = (int32_t*)ptrs[1];
    for (int64_t i = 0; i < n; i++) {
        *out = -*in0;
        in0 = (int32_t*)((char*)in0 + stride[0]);
        out = (int32_t*)((char*)out + stride[1]);
    }
}

void i64_kernel_negate(char **ptrs, const int64_t *stride, int64_t n) {
    int64_t *in0 = (int64_t*)ptrs[0];
    int64_t *out = (int64_t*)ptrs[1];
    for (int64_t i = 0; i < n; i++) {
        *out = -*in0;
        in0 = (int64_t*)((char*)in0 + stride[0]);
        out = (int64_t*)((char*)out + stride[1]);
    }
}

void f32_kernel_negate(char **ptrs, const int64_t *stride, int64_t n) {
    float *in0 = (float*)ptrs[0];
    float *out = (float*)ptrs[1];
    for (int64_t i = 0; i < n; i++) {
        *out = -*in0;
        in0 = (float*)((char*)in0 + stride[0]);
        out = (float*)((char*)out + stride[1]);
    }
}

void f64_kernel_negate(char **ptrs, const int64_t *stride, int64_t n) {
    double *in0 = (double*)ptrs[0];
    double *out = (double*)ptrs[1];
    for (int64_t i = 0; i < n; i++) {
        *out = -*in0;
        in0 = (double*)((char*)in0 + stride[0]);
        out = (double*)((char*)out + stride[1]);
    }
}

void f32_kernel_power(char **ptrs, const int64_t *stride, int64_t n) {
    float *in0 = (float*)ptrs[0];
    float *in1 = (float*)ptrs[1];
    float *out = (float*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = powf(*in0, *in1);
        in0 = (float*)((char*)in0 + stride[0]);
        in1 = (float*)((char*)in1 + stride[1]);
        out = (float*)((char*)out + stride[2]);
    }
}

void f64_kernel_power(char **ptrs, const int64_t *stride, int64_t n) {
    double *in0 = (double*)ptrs[0];
    double *in1 = (double*)ptrs[1];
    double *out = (double*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = pow(*in0, *in1);
        in0 = (double*)((char*)in0 + stride[0]);
        in1 = (double*)((char*)in1 + stride[1]);
        out = (double*)((char*)out + stride[2]);
    }
}

void i8_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n) {
    int8_t *in0 = (int8_t*)ptrs[0];
    int8_t *in1 = (int8_t*)ptrs[1];
    int8_t *out = (int8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = max_8(*in0, *in1);
        in0 = (int8_t*)((char*)in0 + stride[0]);
        in1 = (int8_t*)((char*)in1 + stride[1]);
        out = (int8_t*)((char*)out + stride[2]);
    }
}

void i16_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n) {
    int16_t *in0 = (int16_t*)ptrs[0];
    int16_t *in1 = (int16_t*)ptrs[1];
    int16_t *out = (int16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = max_16(*in0, *in1);
        in0 = (int16_t*)((char*)in0 + stride[0]);
        in1 = (int16_t*)((char*)in1 + stride[1]);
        out = (int16_t*)((char*)out + stride[2]);
    }
}

void i32_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n) {
    int32_t *in0 = (int32_t*)ptrs[0];
    int32_t *in1 = (int32_t*)ptrs[1];
    int32_t *out = (int32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = max_32(*in0, *in1);
        in0 = (int32_t*)((char*)in0 + stride[0]);
        in1 = (int32_t*)((char*)in1 + stride[1]);
        out = (int32_t*)((char*)out + stride[2]);
    }
}

void i64_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n) {
    int64_t *in0 = (int64_t*)ptrs[0];
    int64_t *in1 = (int64_t*)ptrs[1];
    int64_t *out = (int64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = max_64(*in0, *in1);
        in0 = (int64_t*)((char*)in0 + stride[0]);
        in1 = (int64_t*)((char*)in1 + stride[1]);
        out = (int64_t*)((char*)out + stride[2]);
    }
}

void u8_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n) {
    uint8_t *in0 = (uint8_t*)ptrs[0];
    uint8_t *in1 = (uint8_t*)ptrs[1];
    uint8_t *out = (uint8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = max_u8(*in0, *in1);
        in0 = (uint8_t*)((char*)in0 + stride[0]);
        in1 = (uint8_t*)((char*)in1 + stride[1]);
        out = (uint8_t*)((char*)out + stride[2]);
    }
}

void u16_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n) {
    uint16_t *in0 = (uint16_t*)ptrs[0];
    uint16_t *in1 = (uint16_t*)ptrs[1];
    uint16_t *out = (uint16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = max_u16(*in0, *in1);
        in0 = (uint16_t*)((char*)in0 + stride[0]);
        in1 = (uint16_t*)((char*)in1 + stride[1]);
        out = (uint16_t*)((char*)out + stride[2]);
    }
}

void u32_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n) {
    uint32_t *in0 = (uint32_t*)ptrs[0];
    uint32_t *in1 = (uint32_t*)ptrs[1];
    uint32_t *out = (uint32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = max_u32(*in0, *in1);
        in0 = (uint32_t*)((char*)in0 + stride[0]);
        in1 = (uint32_t*)((char*)in1 + stride[1]);
        out = (uint32_t*)((char*)out + stride[2]);
    }
}

void u64_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n) {
    uint64_t *in0 = (uint64_t*)ptrs[0];
    uint64_t *in1 = (uint64_t*)ptrs[1];
    uint64_t *out = (uint64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = max_u64(*in0, *in1);
        in0 = (uint64_t*)((char*)in0 + stride[0]);
        in1 = (uint64_t*)((char*)in1 + stride[1]);
        out = (uint64_t*)((char*)out + stride[2]);
    }
}

void f32_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n) {
    float *in0 = (float*)ptrs[0];
    float *in1 = (float*)ptrs[1];
    float *out = (float*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = fmaxf(*in0, *in1);
        in0 = (float*)((char*)in0 + stride[0]);
        in1 = (float*)((char*)in1 + stride[1]);
        out = (float*)((char*)out + stride[2]);
    }
}

void f64_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n) {
    double *in0 = (double*)ptrs[0];
    double *in1 = (double*)ptrs[1];
    double *out = (double*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = fmax(*in0, *in1);
        in0 = (double*)((char*)in0 + stride[0]);
        in1 = (double*)((char*)in1 + stride[1]);
        out = (double*)((char*)out + stride[2]);
    }
}

void bool_kernel_maximum(char **ptrs, const int64_t *stride, int64_t n) {
    bool *in0 = (bool*)ptrs[0];
    bool *in1 = (bool*)ptrs[1];
    bool *out = (bool*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = max_bool(*in0, *in1);
        in0 = (bool*)((char*)in0 + stride[0]);
        in1 = (bool*)((char*)in1 + stride[1]);
        out = (bool*)((char*)out + stride[2]);
    }
}

void i8_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n) {
    int8_t *in0 = (int8_t*)ptrs[0];
    int8_t *in1 = (int8_t*)ptrs[1];
    int8_t *out = (int8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = min_8(*in0, *in1);
        in0 = (int8_t*)((char*)in0 + stride[0]);
        in1 = (int8_t*)((char*)in1 + stride[1]);
        out = (int8_t*)((char*)out + stride[2]);
    }
}

void i16_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n) {
    int16_t *in0 = (int16_t*)ptrs[0];
    int16_t *in1 = (int16_t*)ptrs[1];
    int16_t *out = (int16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = min_16(*in0, *in1);
        in0 = (int16_t*)((char*)in0 + stride[0]);
        in1 = (int16_t*)((char*)in1 + stride[1]);
        out = (int16_t*)((char*)out + stride[2]);
    }
}

void i32_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n) {
    int32_t *in0 = (int32_t*)ptrs[0];
    int32_t *in1 = (int32_t*)ptrs[1];
    int32_t *out = (int32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = min_32(*in0, *in1);
        in0 = (int32_t*)((char*)in0 + stride[0]);
        in1 = (int32_t*)((char*)in1 + stride[1]);
        out = (int32_t*)((char*)out + stride[2]);
    }
}

void i64_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n) {
    int64_t *in0 = (int64_t*)ptrs[0];
    int64_t *in1 = (int64_t*)ptrs[1];
    int64_t *out = (int64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = min_64(*in0, *in1);
        in0 = (int64_t*)((char*)in0 + stride[0]);
        in1 = (int64_t*)((char*)in1 + stride[1]);
        out = (int64_t*)((char*)out + stride[2]);
    }
}

void u8_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n) {
    uint8_t *in0 = (uint8_t*)ptrs[0];
    uint8_t *in1 = (uint8_t*)ptrs[1];
    uint8_t *out = (uint8_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = min_u8(*in0, *in1);
        in0 = (uint8_t*)((char*)in0 + stride[0]);
        in1 = (uint8_t*)((char*)in1 + stride[1]);
        out = (uint8_t*)((char*)out + stride[2]);
    }
}

void u16_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n) {
    uint16_t *in0 = (uint16_t*)ptrs[0];
    uint16_t *in1 = (uint16_t*)ptrs[1];
    uint16_t *out = (uint16_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = min_u16(*in0, *in1);
        in0 = (uint16_t*)((char*)in0 + stride[0]);
        in1 = (uint16_t*)((char*)in1 + stride[1]);
        out = (uint16_t*)((char*)out + stride[2]);
    }
}

void u32_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n) {
    uint32_t *in0 = (uint32_t*)ptrs[0];
    uint32_t *in1 = (uint32_t*)ptrs[1];
    uint32_t *out = (uint32_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = min_u32(*in0, *in1);
        in0 = (uint32_t*)((char*)in0 + stride[0]);
        in1 = (uint32_t*)((char*)in1 + stride[1]);
        out = (uint32_t*)((char*)out + stride[2]);
    }
}

void u64_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n) {
    uint64_t *in0 = (uint64_t*)ptrs[0];
    uint64_t *in1 = (uint64_t*)ptrs[1];
    uint64_t *out = (uint64_t*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = min_u64(*in0, *in1);
        in0 = (uint64_t*)((char*)in0 + stride[0]);
        in1 = (uint64_t*)((char*)in1 + stride[1]);
        out = (uint64_t*)((char*)out + stride[2]);
    }
}

void f32_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n) {
    float *in0 = (float*)ptrs[0];
    float *in1 = (float*)ptrs[1];
    float *out = (float*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = fminf(*in0, *in1);
        in0 = (float*)((char*)in0 + stride[0]);
        in1 = (float*)((char*)in1 + stride[1]);
        out = (float*)((char*)out + stride[2]);
    }
}

void f64_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n) {
    double *in0 = (double*)ptrs[0];
    double *in1 = (double*)ptrs[1];
    double *out = (double*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = fmin(*in0, *in1);
        in0 = (double*)((char*)in0 + stride[0]);
        in1 = (double*)((char*)in1 + stride[1]);
        out = (double*)((char*)out + stride[2]);
    }
}

void bool_kernel_minimum(char **ptrs, const int64_t *stride, int64_t n) {
    bool *in0 = (bool*)ptrs[0];
    bool *in1 = (bool*)ptrs[1];
    bool *out = (bool*)ptrs[2];
    for (int64_t i = 0; i < n; i++) {
        *out = min_bool(*in0, *in1);
        in0 = (bool*)((char*)in0 + stride[0]);
        in1 = (bool*)((char*)in1 + stride[1]);
        out = (bool*)((char*)out + stride[2]);
    }
}

void i8_kernel_where(char **ptrs, const int64_t *stride, int64_t n) {
    bool *in0 = (bool*)ptrs[0];
    int8_t *in1 = (int8_t*)ptrs[1];
    int8_t *in2 = (int8_t*)ptrs[2];
    int8_t *out = (int8_t*)ptrs[3];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 ? *in1 : *in2;
        in0 = (bool*)((char*)in0 + stride[0]);
        in1 = (int8_t*)((char*)in1 + stride[1]);
        in2 = (int8_t*)((char*)in2 + stride[2]);
        out = (int8_t*)((char*)out + stride[3]);
    }
}

void i16_kernel_where(char **ptrs, const int64_t *stride, int64_t n) {
    bool *in0 = (bool*)ptrs[0];
    int16_t *in1 = (int16_t*)ptrs[1];
    int16_t *in2 = (int16_t*)ptrs[2];
    int16_t *out = (int16_t*)ptrs[3];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 ? *in1 : *in2;
        in0 = (bool*)((char*)in0 + stride[0]);
        in1 = (int16_t*)((char*)in1 + stride[1]);
        in2 = (int16_t*)((char*)in2 + stride[2]);
        out = (int16_t*)((char*)out + stride[3]);
    }
}

void i32_kernel_where(char **ptrs, const int64_t *stride, int64_t n) {
    bool *in0 = (bool*)ptrs[0];
    int32_t *in1 = (int32_t*)ptrs[1];
    int32_t *in2 = (int32_t*)ptrs[2];
    int32_t *out = (int32_t*)ptrs[3];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 ? *in1 : *in2;
        in0 = (bool*)((char*)in0 + stride[0]);
        in1 = (int32_t*)((char*)in1 + stride[1]);
        in2 = (int32_t*)((char*)in2 + stride[2]);
        out = (int32_t*)((char*)out + stride[3]);
    }
}

void i64_kernel_where(char **ptrs, const int64_t *stride, int64_t n) {
    bool *in0 = (bool*)ptrs[0];
    int64_t *in1 = (int64_t*)ptrs[1];
    int64_t *in2 = (int64_t*)ptrs[2];
    int64_t *out = (int64_t*)ptrs[3];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 ? *in1 : *in2;
        in0 = (bool*)((char*)in0 + stride[0]);
        in1 = (int64_t*)((char*)in1 + stride[1]);
        in2 = (int64_t*)((char*)in2 + stride[2]);
        out = (int64_t*)((char*)out + stride[3]);
    }
}

void u8_kernel_where(char **ptrs, const int64_t *stride, int64_t n) {
    bool *in0 = (bool*)ptrs[0];
    uint8_t *in1 = (uint8_t*)ptrs[1];
    uint8_t *in2 = (uint8_t*)ptrs[2];
    uint8_t *out = (uint8_t*)ptrs[3];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 ? *in1 : *in2;
        in0 = (bool*)((char*)in0 + stride[0]);
        in1 = (uint8_t*)((char*)in1 + stride[1]);
        in2 = (uint8_t*)((char*)in2 + stride[2]);
        out = (uint8_t*)((char*)out + stride[3]);
    }
}

void u16_kernel_where(char **ptrs, const int64_t *stride, int64_t n) {
    bool *in0 = (bool*)ptrs[0];
    uint16_t *in1 = (uint16_t*)ptrs[1];
    uint16_t *in2 = (uint16_t*)ptrs[2];
    uint16_t *out = (uint16_t*)ptrs[3];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 ? *in1 : *in2;
        in0 = (bool*)((char*)in0 + stride[0]);
        in1 = (uint16_t*)((char*)in1 + stride[1]);
        in2 = (uint16_t*)((char*)in2 + stride[2]);
        out = (uint16_t*)((char*)out + stride[3]);
    }
}

void u32_kernel_where(char **ptrs, const int64_t *stride, int64_t n) {
    bool *in0 = (bool*)ptrs[0];
    uint32_t *in1 = (uint32_t*)ptrs[1];
    uint32_t *in2 = (uint32_t*)ptrs[2];
    uint32_t *out = (uint32_t*)ptrs[3];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 ? *in1 : *in2;
        in0 = (bool*)((char*)in0 + stride[0]);
        in1 = (uint32_t*)((char*)in1 + stride[1]);
        in2 = (uint32_t*)((char*)in2 + stride[2]);
        out = (uint32_t*)((char*)out + stride[3]);
    }
}

void u64_kernel_where(char **ptrs, const int64_t *stride, int64_t n) {
    bool *in0 = (bool*)ptrs[0];
    uint64_t *in1 = (uint64_t*)ptrs[1];
    uint64_t *in2 = (uint64_t*)ptrs[2];
    uint64_t *out = (uint64_t*)ptrs[3];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 ? *in1 : *in2;
        in0 = (bool*)((char*)in0 + stride[0]);
        in1 = (uint64_t*)((char*)in1 + stride[1]);
        in2 = (uint64_t*)((char*)in2 + stride[2]);
        out = (uint64_t*)((char*)out + stride[3]);
    }
}

void f32_kernel_where(char **ptrs, const int64_t *stride, int64_t n) {
    bool *in0 = (bool*)ptrs[0];
    float *in1 = (float*)ptrs[1];
    float *in2 = (float*)ptrs[2];
    float *out = (float*)ptrs[3];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 ? *in1 : *in2;
        in0 = (bool*)((char*)in0 + stride[0]);
        in1 = (float*)((char*)in1 + stride[1]);
        in2 = (float*)((char*)in2 + stride[2]);
        out = (float*)((char*)out + stride[3]);
    }
}

void f64_kernel_where(char **ptrs, const int64_t *stride, int64_t n) {
    bool *in0 = (bool*)ptrs[0];
    double *in1 = (double*)ptrs[1];
    double *in2 = (double*)ptrs[2];
    double *out = (double*)ptrs[3];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 ? *in1 : *in2;
        in0 = (bool*)((char*)in0 + stride[0]);
        in1 = (double*)((char*)in1 + stride[1]);
        in2 = (double*)((char*)in2 + stride[2]);
        out = (double*)((char*)out + stride[3]);
    }
}

void bool_kernel_where(char **ptrs, const int64_t *stride, int64_t n) {
    bool *in0 = (bool*)ptrs[0];
    bool *in1 = (bool*)ptrs[1];
    bool *in2 = (bool*)ptrs[2];
    bool *out = (bool*)ptrs[3];
    for (int64_t i = 0; i < n; i++) {
        *out = *in0 ? *in1 : *in2;
        in0 = (bool*)((char*)in0 + stride[0]);
        in1 = (bool*)((char*)in1 + stride[1]);
        in2 = (bool*)((char*)in2 + stride[2]);
        out = (bool*)((char*)out + stride[3]);
    }
}

void i8_kernel_fill(char **ptrs, const int64_t *stride, int64_t n) {
    int8_t *in0 = (int8_t*)ptrs[0];
    int8_t *out = (int8_t*)ptrs[1];
    int8_t fill_value = *in0;
    for (int64_t i = 0; i < n; i++) {
       *out = fill_value;
        out = (int8_t*)((char*)out + stride[1]);
    }
}

void i16_kernel_fill(char **ptrs, const int64_t *stride, int64_t n) {
    int16_t *in0 = (int16_t*)ptrs[0];
    int16_t *out = (int16_t*)ptrs[1];
    int16_t fill_value = *in0;
    for (int64_t i = 0; i < n; i++) {
       *out = fill_value;
        out = (int16_t*)((char*)out + stride[1]);
    }
}

void i32_kernel_fill(char **ptrs, const int64_t *stride, int64_t n) {
    int32_t *in0 = (int32_t*)ptrs[0];
    int32_t *out = (int32_t*)ptrs[1];
    int32_t fill_value = *in0;
    for (int64_t i = 0; i < n; i++) {
       *out = fill_value;
        out = (int32_t*)((char*)out + stride[1]);
    }
}

void i64_kernel_fill(char **ptrs, const int64_t *stride, int64_t n) {
    int64_t *in0 = (int64_t*)ptrs[0];
    int64_t *out = (int64_t*)ptrs[1];
    int64_t fill_value = *in0;
    for (int64_t i = 0; i < n; i++) {
       *out = fill_value;
        out = (int64_t*)((char*)out + stride[1]);
    }
}

void u8_kernel_fill(char **ptrs, const int64_t *stride, int64_t n) {
    uint8_t *in0 = (uint8_t*)ptrs[0];
    uint8_t *out = (uint8_t*)ptrs[1];
    uint8_t fill_value = *in0;
    for (int64_t i = 0; i < n; i++) {
       *out = fill_value;
        out = (uint8_t*)((char*)out + stride[1]);
    }
}

void u16_kernel_fill(char **ptrs, const int64_t *stride, int64_t n) {
    uint16_t *in0 = (uint16_t*)ptrs[0];
    uint16_t *out = (uint16_t*)ptrs[1];
    uint16_t fill_value = *in0;
    for (int64_t i = 0; i < n; i++) {
       *out = fill_value;
        out = (uint16_t*)((char*)out + stride[1]);
    }
}

void u32_kernel_fill(char **ptrs, const int64_t *stride, int64_t n) {
    uint32_t *in0 = (uint32_t*)ptrs[0];
    uint32_t *out = (uint32_t*)ptrs[1];
    uint32_t fill_value = *in0;
    for (int64_t i = 0; i < n; i++) {
       *out = fill_value;
        out = (uint32_t*)((char*)out + stride[1]);
    }
}

void u64_kernel_fill(char **ptrs, const int64_t *stride, int64_t n) {
    uint64_t *in0 = (uint64_t*)ptrs[0];
    uint64_t *out = (uint64_t*)ptrs[1];
    uint64_t fill_value = *in0;
    for (int64_t i = 0; i < n; i++) {
       *out = fill_value;
        out = (uint64_t*)((char*)out + stride[1]);
    }
}

void f32_kernel_fill(char **ptrs, const int64_t *stride, int64_t n) {
    float *in0 = (float*)ptrs[0];
    float *out = (float*)ptrs[1];
    float fill_value = *in0;
    for (int64_t i = 0; i < n; i++) {
       *out = fill_value;
        out = (float*)((char*)out + stride[1]);
    }
}

void f64_kernel_fill(char **ptrs, const int64_t *stride, int64_t n) {
    double *in0 = (double*)ptrs[0];
    double *out = (double*)ptrs[1];
    double fill_value = *in0;
    for (int64_t i = 0; i < n; i++) {
       *out = fill_value;
        out = (double*)((char*)out + stride[1]);
    }
}

void bool_kernel_fill(char **ptrs, const int64_t *stride, int64_t n) {
    bool *in0 = (bool*)ptrs[0];
    bool *out = (bool*)ptrs[1];
    bool fill_value = *in0;
    for (int64_t i = 0; i < n; i++) {
       *out = fill_value;
        out = (bool*)((char*)out + stride[1]);
    }
}

void i8_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    int8_t *in0 = (int8_t*)ptrs[idx_in];
    int8_t *out = (int8_t*)ptrs[idx_out];
    int8_t *values = (int8_t*)input_values;
    int8_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = out[output_indexes[i]] + current_value;
    }
}

void i16_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    int16_t *in0 = (int16_t*)ptrs[idx_in];
    int16_t *out = (int16_t*)ptrs[idx_out];
    int16_t *values = (int16_t*)input_values;
    int16_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = out[output_indexes[i]] + current_value;
    }
}

void i32_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    int32_t *in0 = (int32_t*)ptrs[idx_in];
    int32_t *out = (int32_t*)ptrs[idx_out];
    int32_t *values = (int32_t*)input_values;
    int32_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = out[output_indexes[i]] + current_value;
    }
}

void i64_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    int64_t *in0 = (int64_t*)ptrs[idx_in];
    int64_t *out = (int64_t*)ptrs[idx_out];
    int64_t *values = (int64_t*)input_values;
    int64_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = out[output_indexes[i]] + current_value;
    }
}

void u8_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    uint8_t *in0 = (uint8_t*)ptrs[idx_in];
    uint8_t *out = (uint8_t*)ptrs[idx_out];
    uint8_t *values = (uint8_t*)input_values;
    uint8_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = out[output_indexes[i]] + current_value;
    }
}

void u16_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    uint16_t *in0 = (uint16_t*)ptrs[idx_in];
    uint16_t *out = (uint16_t*)ptrs[idx_out];
    uint16_t *values = (uint16_t*)input_values;
    uint16_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = out[output_indexes[i]] + current_value;
    }
}

void u32_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    uint32_t *in0 = (uint32_t*)ptrs[idx_in];
    uint32_t *out = (uint32_t*)ptrs[idx_out];
    uint32_t *values = (uint32_t*)input_values;
    uint32_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = out[output_indexes[i]] + current_value;
    }
}

void u64_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    uint64_t *in0 = (uint64_t*)ptrs[idx_in];
    uint64_t *out = (uint64_t*)ptrs[idx_out];
    uint64_t *values = (uint64_t*)input_values;
    uint64_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = out[output_indexes[i]] + current_value;
    }
}

void f32_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    float *in0 = (float*)ptrs[idx_in];
    float *out = (float*)ptrs[idx_out];
    float *values = (float*)input_values;
    float current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = out[output_indexes[i]] + current_value;
    }
}

void f64_kernel_sum(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    double *in0 = (double*)ptrs[idx_in];
    double *out = (double*)ptrs[idx_out];
    double *values = (double*)input_values;
    double current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = out[output_indexes[i]] + current_value;
    }
}

void i8_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    int8_t *in0 = (int8_t*)ptrs[idx_in];
    int8_t *out = (int8_t*)ptrs[idx_out];
    int8_t *values = (int8_t*)input_values;
    int8_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value > out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void i16_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    int16_t *in0 = (int16_t*)ptrs[idx_in];
    int16_t *out = (int16_t*)ptrs[idx_out];
    int16_t *values = (int16_t*)input_values;
    int16_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value > out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void i32_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    int32_t *in0 = (int32_t*)ptrs[idx_in];
    int32_t *out = (int32_t*)ptrs[idx_out];
    int32_t *values = (int32_t*)input_values;
    int32_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value > out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void i64_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    int64_t *in0 = (int64_t*)ptrs[idx_in];
    int64_t *out = (int64_t*)ptrs[idx_out];
    int64_t *values = (int64_t*)input_values;
    int64_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value > out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void u8_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    uint8_t *in0 = (uint8_t*)ptrs[idx_in];
    uint8_t *out = (uint8_t*)ptrs[idx_out];
    uint8_t *values = (uint8_t*)input_values;
    uint8_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value > out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void u16_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    uint16_t *in0 = (uint16_t*)ptrs[idx_in];
    uint16_t *out = (uint16_t*)ptrs[idx_out];
    uint16_t *values = (uint16_t*)input_values;
    uint16_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value > out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void u32_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    uint32_t *in0 = (uint32_t*)ptrs[idx_in];
    uint32_t *out = (uint32_t*)ptrs[idx_out];
    uint32_t *values = (uint32_t*)input_values;
    uint32_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value > out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void u64_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    uint64_t *in0 = (uint64_t*)ptrs[idx_in];
    uint64_t *out = (uint64_t*)ptrs[idx_out];
    uint64_t *values = (uint64_t*)input_values;
    uint64_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value > out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void f32_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    float *in0 = (float*)ptrs[idx_in];
    float *out = (float*)ptrs[idx_out];
    float *values = (float*)input_values;
    float current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value > out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void f64_kernel_max(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    double *in0 = (double*)ptrs[idx_in];
    double *out = (double*)ptrs[idx_out];
    double *values = (double*)input_values;
    double current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value > out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void i8_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    int8_t *in0 = (int8_t*)ptrs[idx_in];
    int8_t *out = (int8_t*)ptrs[idx_out];
    int8_t *values = (int8_t*)input_values;
    int8_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value < out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void i16_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    int16_t *in0 = (int16_t*)ptrs[idx_in];
    int16_t *out = (int16_t*)ptrs[idx_out];
    int16_t *values = (int16_t*)input_values;
    int16_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value < out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void i32_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    int32_t *in0 = (int32_t*)ptrs[idx_in];
    int32_t *out = (int32_t*)ptrs[idx_out];
    int32_t *values = (int32_t*)input_values;
    int32_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value < out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void i64_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    int64_t *in0 = (int64_t*)ptrs[idx_in];
    int64_t *out = (int64_t*)ptrs[idx_out];
    int64_t *values = (int64_t*)input_values;
    int64_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value < out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void u8_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    uint8_t *in0 = (uint8_t*)ptrs[idx_in];
    uint8_t *out = (uint8_t*)ptrs[idx_out];
    uint8_t *values = (uint8_t*)input_values;
    uint8_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value < out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void u16_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    uint16_t *in0 = (uint16_t*)ptrs[idx_in];
    uint16_t *out = (uint16_t*)ptrs[idx_out];
    uint16_t *values = (uint16_t*)input_values;
    uint16_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value < out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void u32_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    uint32_t *in0 = (uint32_t*)ptrs[idx_in];
    uint32_t *out = (uint32_t*)ptrs[idx_out];
    uint32_t *values = (uint32_t*)input_values;
    uint32_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value < out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void u64_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    uint64_t *in0 = (uint64_t*)ptrs[idx_in];
    uint64_t *out = (uint64_t*)ptrs[idx_out];
    uint64_t *values = (uint64_t*)input_values;
    uint64_t current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value < out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void f32_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    float *in0 = (float*)ptrs[idx_in];
    float *out = (float*)ptrs[idx_out];
    float *values = (float*)input_values;
    float current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value < out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

void f64_kernel_min(char **ptrs, void *input_values, int64_t *output_indexes, int length, int idx_in, int idx_out) {
    double *in0 = (double*)ptrs[idx_in];
    double *out = (double*)ptrs[idx_out];
    double *values = (double*)input_values;
    double current_value = 0;
    for (int64_t i = 0; i < length; i++) {
         current_value = values[i];
         out[output_indexes[i]] = current_value < out[output_indexes[i]] ? current_value : out[output_indexes[i]];
    }
}

