import torch
import torch.nn as nn


class NIGnet(nn.Module):

    def __init__(self, layer_count, act_fn):
        super(NIGnet, self).__init__()

        # Define the transformation from t on the [0, 1] interval to unit circle for closed shapes
        self.closed_transform = lambda t: torch.hstack([
            torch.cos(2 * torch.pi * t),
            torch.sin(2 * torch.pi * t)
        ])

        layers = []
        for i in range(layer_count):
            layers.append(nn.Linear(2, 2))
            layers.append(act_fn())
        layers.append(nn.Linear(2, 2))
        
        self.linear_act_stack = nn.Sequential(*layers)
    

    def forward(self, t):
        X = self.closed_transform(t)
        X = self.linear_act_stack(X)
        return X