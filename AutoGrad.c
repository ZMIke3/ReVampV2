#include "C:\Programming\Revamp2\AutoGrad.h"


TensorList *list_create() {
    TensorList *list = (TensorList *) malloc(sizeof(TensorList));
    list->capacity = 10;
    list->numel = 0;
    list->container = (Tensor **) malloc(sizeof(Tensor *) * list->capacity);
    return list;
}

void list_resize_container(TensorList *list) {
    if (!list || !list->container) {
        printf("List is not initialized\n");
        return;
    }

    if (list->numel >= list->capacity) {
        list->capacity *=2;
        list->container = (Tensor **) realloc(list->container, sizeof(Tensor *) * list->capacity);
    }
}

void list_tensor_to_container(Tensor *from, TensorList *to) {

    list_resize_container(to);

    to->container[to->numel++] = from;

}

void list_free(TensorList *list) {
    
    if (!list) {
        printf("List is not initialized\n");
    }

    for (int i = 0; i < list->numel; i++) {
        free(list->container[i]);
    }

    free(list);
}


Node *node_create(Tensor *output, backward_function b_fn, char *node_name, char *b_fn_name) {
    Node *node = (Node *) malloc(sizeof(Node));
    node->inputs = list_create();
    node->output = output;
    node->node_name = strdup(node_name);
    node->b_fn_name = strdup(b_fn_name);
    node->visited = false;
    node->b_fn = b_fn;
    return node;

}

NodeList *Node_list_create() {
    NodeList *list = (NodeList *) malloc(sizeof(NodeList));
    list->capacity = 10;
    list->numel = 0;
    list->container = (Node **) malloc(sizeof(Node *) * list->capacity);
    return list;
}

void Node_list_resize_container(NodeList *list) {
    if (!list || !list->container) {
        printf("List is not initialized\n");
        return;
    }

    if (list->numel >= list->capacity) {
        list->capacity *=2;
        list->container = (Node **) realloc(list->container, sizeof(Node *) * list->capacity);
    }
}

void Node_list_Node_to_container(Node *from, NodeList *to) {

    Node_list_resize_container(to);

    to->container[to->numel++] = from;
}

void Node_list_free(NodeList *list) {
    
    if (!list) {
        printf("List is not initialized\n");
    }

    for (int i = 0; i < list->numel; i++) {
        free(list->container[i]);
    }

    free(list);
}


void node_tensor_to_inputs(Tensor *from, Node *to) {

    if (!to || !from) {
        printf("One of the inputs are not initialized\n");
        return;
    }

    list_tensor_to_container(from, to->inputs);

}

void node_free(Node *node) {
    if (!node) {
        printf("Node is not initialized\n");
    }

    list_free(node->inputs);
    free(node->output);
    free(node->node_name);
    free(node->b_fn_name);
    free(node);
}


void zero_grad(Tensor *input) {

    dtype tensor_dtype = get_tensor_dtype(input);
    input->grad = calloc(input->size, get_dtype_size(tensor_dtype));

    for (int i = 0; i < input->size; i++) {

        SET_TENSOR_GRAD_VALUE(tensor_dtype, input, i, 0);

    }
    

}

void seed_grad(Tensor *input) {

    dtype tensor_dtype = get_tensor_dtype(input);

    input->grad = calloc(input->size, get_dtype_size(tensor_dtype));

    for (int i = 0; i < input->size; i++) {

        SET_TENSOR_GRAD_VALUE(tensor_dtype, input, i, 1);

    }
    

}

void create_backward_graph(NodeList *list, Node *node) {
    if (!node) {
        printf("create_backward_graph: node is NULL\n");
        return;
    }
    if (node->visited) return;
    node->visited = true;

    if (!node->inputs || node->inputs->numel == 0) {
        // Leaf node: no parents/inputs, nothing more to recurse
        Node_list_Node_to_container(node, list);
        return;
    }
    for (int i = 0; i < node->inputs->numel; i++) {
        if (!node->inputs->container[i]) {
            printf("create_backward_graph: node->inputs->container[%d] is NULL\n", i);
            continue;
        }
        if (!node->inputs->container[i]->node || node->inputs->container[i]->is_leaf == false) {
            // It's common for leaves (inputs) to have no node; skip
            continue;
        }
        create_backward_graph(list, node->inputs->container[i]->node);
    }
    Node_list_Node_to_container(node, list);
}


void gradient(Tensor *input) {

    seed_grad(input);

    NodeList *list =  Node_list_create();

//    if (create_backward_graph(list, input->node) == NULL) {printf("There is NULL\n");}


    
    printf("Start_g\n");
    create_backward_graph(list, input->node);
    printf("End_g\n");

    printf("Start_gg\n");
    for (int i = 0; i < list->numel; i++) {
        
        if (list->container[i] == NULL) {
            printf("True it's NULL\n");
        }
    }
    printf("End_gg\n");

    for (int i = list->numel - 1; i >= 0; i--) {

        if (list->container[i]->inputs == NULL) {
            printf("True\n");
            continue;
        }

        list->container[i]->b_fn(list->container[i]);
    }

    for (int i = 0; i < list->numel; i++) {
        list->container[i]->visited = false;
    }

    
}


// void gradient(Tensor *input) {
//     seed_grad(input);
//     NodeList *list = Node_list_create();
//     create_backward_graph(list, input->node); 
//     for (int i = list->numel - 1; i >= 0; i--) {
//         if (list->container[i]->b_fn) { 
//             list->container[i]->b_fn(list->container[i]);
//         }
//     }
// }

