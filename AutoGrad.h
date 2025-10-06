#ifndef AUTOGRAD_H
#define AUTOGRAD_H
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "C:\Programming\Revamp2\Tensor\Tensor.h"
#include "C:\Programming\Revamp2\Utils\utills.h"


typedef struct Tensor Tensor;

typedef struct Node Node;

typedef void (*backward_function)(Node *);

typedef struct TensorList {
    Tensor **container;
    int numel;
    int capacity;
}TensorList;

typedef struct NodeList {
    Node **container;
    int numel;
    int capacity;
}NodeList;


typedef struct Node {
    TensorList *inputs;
    Tensor *output;
    backward_function b_fn;
    char *node_name;
    char *b_fn_name;    
    bool visited;
    bool requires_grad;
    bool is_leaf;
} Node;


/* Engine Support */

TensorList *list_create();

void list_resize_container(TensorList *list);
void list_tensor_to_container(Tensor *from, TensorList *to);
void list_free(TensorList *list);


Node *node_create(Tensor *output, backward_function b_fn, char *node_name, char *b_fn_name);
NodeList *Node_list_create();
void Node_list_resize_container(NodeList *list);
void Node_list_Node_to_container(Node *from, NodeList *to);
void Node_list_free(NodeList *list);
void node_tensor_to_inputs(Tensor *from, Node *to);
void node_free(Node *node);
void create_backward_graph(NodeList *list, Node *node);


/* Gradient Computation */
bool is_leaf(Tensor *input);
void retain_grad(Tensor *input);
void zero_grad(Tensor *input);
void seed_grad(Tensor *input);
void gradient(Tensor *input);
void requires_grad(Tensor *input, bool requires_grad);

#endif // AUTOGRAD_H
