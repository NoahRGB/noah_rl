import torch

def detect_activation(activation: str):
    if activation == "RELU":
        return torch.nn.ReLU()
    elif activation == "TANH":
        return torch.nn.Tanh()
    elif activation == "NONE":
        return None
    else:
        raise ValueError(f"Activation {activation} not supported")


def build_network(architecture: list, input_shape: tuple = None, output_shape: tuple= None):
    layers = []
    last_output = None

    for layer in architecture:

        if layer["type"] == "FLATTEN":
            layers.append(torch.nn.Flatten())
            continue

        layer_in = layer["in"]
        layer_out = layer["out"]

        if layer_in == "IN": layer_in = input_shape[0]
        if layer_out == "OUT": layer_out = output_shape[0]

        last_output = layer_out

        activation = None
        if "activation" in layer:
            activation = detect_activation(layer["activation"])
        
        if layer["type"] == "LINEAR":
            layers.append(torch.nn.Linear(layer_in, layer_out))
        elif layer["type"] == "CONV":
            kernel_size, stride = layer["kernel"], layer["stride"]
            layers.append(torch.nn.Conv2d(layer_in, layer_out, kernel_size, stride))

        if activation:
            layers.append(activation)

    return torch.nn.Sequential(*layers), last_output



