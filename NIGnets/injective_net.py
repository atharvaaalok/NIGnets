import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class NIGnet(nn.Module):

    available_intersection_modes = ['possible', 'impossible']

    def __init__(self, layer_count, act_fn, intersection = 'possible'):
        super(NIGnet, self).__init__()

        if intersection not in self.available_intersection_modes:
            raise ValueError(f'Invalid intersection mode. ' \
                             f'Choose from {self.available_intersection_modes}')
        
        self.intersection = intersection

        # Define the transformation from t on the [0, 1] interval to unit circle for closed shapes
        self.closed_transform = lambda t: torch.hstack([
            torch.cos(2 * torch.pi * t),
            torch.sin(2 * torch.pi * t)
        ])

        self.Linear_class = nn.Linear if intersection == 'possible' else ExpLinear
        
        layers = []
        for i in range(layer_count):
            layers.append(self.Linear_class(2, 2))
            layers.append(act_fn())
        layers.append(self.Linear_class(2, 2))
        
        self.linear_act_stack = nn.Sequential(*layers)
    

    def forward(self, t):
        X = self.closed_transform(t)
        X = self.linear_act_stack(X)
        return X



class ExpLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = True):
        super().__init__()

        assert in_features == out_features, 'ExpLinear requires in_features == out_features'

        self.in_features = in_features
        self.out_features = out_features
        
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a = math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    

    def forward(self, x):
        exp_weight = torch.matrix_exp(self.W)
        return F.linear(x, exp_weight.t(), self.bias)