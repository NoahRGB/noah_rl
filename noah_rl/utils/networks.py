# NoahRGB 09/2026
#
# a file containing various helper functions/classes for 
# parsing PyTorch network architectures that are specified
# in the yaml files in noah_rl/params/
#
# this helps to abstract the repeated process of building
# networks out of the individual algorithm files
#

import gymnasium as gym
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

def build_network(architecture: list, input_shape: tuple|None = None, output_shape: tuple|None = None):
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

class Encoder(torch.nn.Module):

    def __init__(self, architecture: list, input_shape: tuple|None = None, output_shape: tuple|None = None):
        super(Encoder, self).__init__()

        self.body, self.encoder_output = build_network(architecture=architecture, input_shape=input_shape, output_shape=output_shape)

    def forward(self, inp):
        return self.body(inp)

def detect_head(action_space: gym.spaces.Discrete|gym.spaces.Box, input_size: int):

    if isinstance(action_space, gym.spaces.Discrete):
        return CategoricalHead(input_size=input_size, output_size=action_space.n)
    
    elif isinstance(action_space, gym.spaces.Box):
        return GaussianHead(input_size=input_size, output_size=action_space.shape[0])

class CategoricalHead(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int):
        super(CategoricalHead, self).__init__()

        self.head = torch.nn.Linear(input_size, output_size)

    def forward(self, inp):
        logits = self.head(inp)
        return torch.distributions.Categorical(logits=logits)

class GaussianHead(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int):
        super(GaussianHead, self).__init__()

        self.mean_head = torch.nn.Linear(input_size, output_size)
        self.log_std_head = torch.nn.Linear(input_size, output_size)

    def forward(self, inp):
        mean = self.mean_head(inp)
        stdev = self.log_std_head(inp).exp()
        distribution = torch.distributions.Independent(torch.distributions.Normal(mean, stdev), 1)
        return distribution 
