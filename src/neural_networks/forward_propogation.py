import numpy as np
from random import seed
#initialize network
def init_network(num_inputs, num_hidden_layers, num_hidden_nodes_arr, num_output):
    num_nodes_previous = num_inputs
    num_nodes = 0
    network = {}

    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer_num in range(num_hidden_layers+1):

        #this means we have gotten to the out output layer
        if(layer_num == num_hidden_layers):
            layer_name = 'output'
            num_nodes = num_output
        else:
            layer_name = 'hidden_layer{}'.format(layer_num+1)
            num_nodes = num_hidden_nodes_arr[layer_num]
        
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2)
            }

        num_nodes_previous = num_nodes
    
    return network #return the network

#create a network with function defined
small_network = init_network(2, 3, [3,3,3], 1)

print(small_network)

#compute weighted sum at each node
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)

print('\nThe inputs to the network are {}'.format(inputs))

def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))

def forward_propogate(network, inputs):
    layer_inputs = list(inputs)

    for layer in network:
        layer_data = network[layer]

        layer_outputs = []
        for layer_node in layer_data:

            node_data = layer_data[layer_node]

            #compute the wieghted sum and the node activation 
            layer_outputs
