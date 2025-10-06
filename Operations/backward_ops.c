#include "C:\Programming\Revamp2\Operations\backward_ops.h"


#define DEFINE_BACKWARD_BINARY_KERNEL(name, type, grad_p0_expr, grad_p1_expr)\
void name##_##type(char **ptrs, const int64_t *stride, int64_t n) { \
    type *p0_data = (type*)ptrs[0];      /* First parent tensor data */ \
    type *p1_data = (type*)ptrs[1];      /* Second parent tensor data */ \
    type *output_data = (type*)ptrs[2];  /* Output tensor data */ \
    type *p0_grad = (type*)ptrs[3];      /* First parent gradient accumulator */ \
    type *p1_grad = (type*)ptrs[4];      /* Second parent gradient accumulator */ \
    type *output_grad = (type*)ptrs[5];  /* Output gradient (incoming gradient) */ \
    for (int64_t i = 0; i < n; i++) { \
        *p0_grad += (grad_p0_expr); \
        *p1_grad += (grad_p1_expr); \
        p0_data = (type*)((char*)p0_data + stride[0]); \
        p1_data = (type*)((char*)p1_data + stride[1]); \
        output_data = (type*)((char*)output_data + stride[2]); \
        p0_grad = (type*)((char*)p0_grad + stride[3]); \
        p1_grad = (type*)((char*)p1_grad + stride[4]); \
        output_grad = (type*)((char*)output_grad + stride[5]); \
    } \
}


Iterator *return_backward_Iterator(Node *node) {

    Tensor_Config *cfg = tensor_config_make();

    if (!cfg) { printf("Failed to make config in function: binary_step_2_make_iterator\n"); return NULL; }

    // size_t elem_size = get_dtype_size(result_type);


    for (int i = 0; i < node->inputs->numel; i++) {
        tensor_config_add_operand(cfg, node->inputs->container[i]->data, get_tensor_ndim(node->inputs->container[i]), get_tensor_stride(node->inputs->container[i]), get_tensor_shape(node->inputs->container[i]), get_dtype_size(get_tensor_dtype(node->inputs->container[i])), TENSOR_INPUT);
        tensor_config_add_operand(cfg, node->inputs->container[i]->grad, get_tensor_ndim(node->inputs->container[i]), get_tensor_stride(node->inputs->container[i]), get_tensor_shape(node->inputs->container[i]), get_dtype_size(get_tensor_dtype(node->inputs->container[i])), TENSOR_INPUT);

    }
    
    tensor_config_add_operand(cfg, node->output->data, get_tensor_ndim(node->output),  get_tensor_stride(node->output), get_tensor_shape(node->output), get_dtype_size(get_tensor_dtype(node->output)), TENSOR_OUTPUT);
    tensor_config_add_operand(cfg, node->output->grad, get_tensor_ndim(node->output),  get_tensor_stride(node->output), get_tensor_shape(node->output), get_dtype_size(get_tensor_dtype(node->output)), TENSOR_OUTPUT);



    Iterator *iter = tensor_iterator_make();
    if (!iter) {
        printf("Failed to allocate iterator in function: return_backward_terator\n");
        // Prehaps free the tensors and config?
        return NULL;
    }

    if (!tensor_iterator_build(cfg, iter)) {
        printf("Failed to build tensor iterator in function: return_backward_Iterator\n");
        tensor_iterator_free(iter);
        return NULL;
    }

    return iter;


}


void backward_add_kernel(char **ptrs, const int64_t *stride, int64_t loop_len) {
    int64_t *p0_data = (int64_t*)ptrs[0];      /* First parent tensor data */ 
    int64_t *p1_data = (int64_t*)ptrs[1];      /* Second parent tensor data */ 
    int64_t *output_data = (int64_t*)ptrs[2];  /* Output tensor data */ 
    int64_t *p0_grad = (int64_t*)ptrs[3];      /* First parent gradient accumulator */ 
    int64_t *p1_grad = (int64_t*)ptrs[4];      /* Second parent gradient accumulator */ 
    int64_t *output_grad = (int64_t*)ptrs[5];  /* Output gradient (incoming gradient) */ 
    for (int64_t i = 0; i < loop_len; i++) { 
        *p0_grad += *output_grad; 
        *p1_grad += *output_grad; 
        p0_data = (int64_t*)((char*)p0_data + stride[0]); 
        p1_data = (int64_t*)((char*)p1_data + stride[1]); 
        output_data = (int64_t*)((char*)output_data + stride[2]); 
        p0_grad = (int64_t*)((char*)p0_grad + stride[3]); 
        p1_grad = (int64_t*)((char*)p1_grad + stride[4]); 
        output_grad = (int64_t*)((char*)output_grad + stride[5]); 
    } 

}

// void backward_add(Node *node) {


//     Tensor *p0 = (Tensor *) node->inputs->container[0];

//     Tensor *p1 = (Tensor *) node->inputs->container[1];

//     Tensor *out = (Tensor *) node->output;

//     for (int i = 0; i < out->size; i++) {

//         ((int64_t *)p0->grad)[i] += ((int64_t *)out->grad)[i];
//         ((int64_t *)p1->grad)[i] += ((int64_t *)out->grad)[i]; 

//     }


// }


// void backward_mul(Node *node) {

//     Tensor *p0 = (Tensor *) node->inputs->container[0];

//     Tensor *p1 = (Tensor *) node->inputs->container[1];

//     Tensor *out = (Tensor *) node->output;

//     for (int i = 0; i < out->size; i++) {

//         ((int64_t *)p0->grad)[i] += ((int64_t *)out->grad)[i] * ((int64_t *)p1->grad)[i];
//         ((int64_t *)p1->grad)[i] += ((int64_t *)out->grad)[i] * ((int64_t *)p0->grad)[i]; 

//     }

// }



void backward_add(Node *node) {


    Tensor *p0 = (Tensor *) node->inputs->container[0];

    Tensor *p1 = (Tensor *) node->inputs->container[1];

    Tensor *out = (Tensor *) node->output;

    for (int i = 0; i < out->size; i++) {

        ((float *)p0->grad)[i] += ((float *)out->grad)[i];
        ((float *)p1->grad)[i] += ((float *)out->grad)[i]; 

    }


}


void backward_mul(Node *node) {

    Tensor *p0 = (Tensor *) node->inputs->container[0];

    Tensor *p1 = (Tensor *) node->inputs->container[1];

    Tensor *out = (Tensor *) node->output;

    for (int i = 0; i < out->size; i++) {

        ((float *)p0->grad)[i] += ((float *)out->grad)[i] * ((float *)p1->grad)[i];
        ((float *)p1->grad)[i] += ((float *)out->grad)[i] * ((float *)p0->grad)[i]; 

    }

}

