import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Callable
import copy
import matplotlib.pyplot as plt


class NIGnet(nn.Module):
    """
    Neural Injective Geometry network (NIGnet) for mapping a parameter `t` in [0, 1] onto a simple
    closed curve.

    Attributes
    ----------
    available_intersection_modes : list[str]
        List of valid intersection modes. Valid values are ['possible', 'impossible']
    layer_count : int
        The number of (linear + activation) layers in the network. There is one extra linear layer
        at the end.
    intersection : str
        The chosen intersection mode that determines the linear layer to be `nn.Linear` or
        `ExpLinear`.
    closed_transform : Callable[[torch.Tensor], torch.Tensor]
        A function that maps `t` (in [0, 1]) onto the unit circle.
    linear_act_stack : nn.Sequential
        A sequence of alternating linear layers (or ExpLinear layers) and activation functions.
    skip_connections : bool
        Adds skip connections similar to those used in ResNet architecture.
    
    Parameters
    ----------
    layer_count : int
        The number of (linear + activation) layers in the network. There is one extra linear layer
        at the end.
    act_fn : Callable[[], torch.nn.Module]
        Activation function constructor (e.g. `nn.ReLU`)
    intersection : str, optional
        Intersection mode determining layer type, by default 'possible'.
        Must be one of ['possible', 'impossible'].
        If set to 'possible', standard nn.Linear is used;
        If set to 'impossible', ExpLinear is used.
    skip_connections : bool, optional
        Adds skip connections similar to those used in ResNet architecture.
    """

    available_intersection_modes = ['possible', 'impossible']

    def __init__(
        self,
        layer_count: int,
        act_fn: Callable[[], nn.Module] = None,
        monotonic_net: nn.Module = None,
        preaux_net = None,
        intersection: str = 'possible',
        skip_connections: bool = True,
        geometry_dim: int = 2
    ) -> None:
        """
        Initialize NIGnet with specified architecture and intersection mode.
        
        Raises
        ------
        ValueError
            If `intersection` is not one of the supported modes.
        """

        super(NIGnet, self).__init__()

        self.layer_count = layer_count

        if (act_fn is None) and (monotonic_net is None):
            raise ValueError(
                'Either an activation function or a monotonic network must be specified.'
            )
        if (act_fn is not None) and (monotonic_net is not None):
            raise ValueError('Only one of "act_fn" or "monotonic_net" can be specified.')
        self.act_fn = act_fn
        self.monotonic_net = monotonic_net

        self.preaux_net = nn.ModuleList()
        self.preaux_net = copy.deepcopy(preaux_net)

        if intersection not in self.available_intersection_modes:
            raise ValueError(f'Invalid intersection mode. ' \
                             f'Choose from {self.available_intersection_modes}')
        self.intersection = intersection

        self.skip_connections = skip_connections

        self.geometry_dim = geometry_dim
        # Define the transformation from t on the [0, 1] interval to unit circle for closed shapes
        if preaux_net is not None:
            self.closed_transform = self.preaux_net
        else:
            if geometry_dim == 2:
                self.closed_transform = lambda t: torch.hstack([
                    torch.cos(2 * torch.pi * t),
                    torch.sin(2 * torch.pi * t)
                ])
            elif geometry_dim == 3:
                self.closed_transform = lambda t, s: torch.hstack([
                torch.sin(torch.pi * s) * torch.cos(2 * torch.pi * t),
                torch.sin(torch.pi * s) * torch.sin(2 * torch.pi * t),
                torch.cos(torch.pi * s)
                ])

        Linear_class = nn.Linear if intersection == 'possible' else ExpLinear
        
        self.linear_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()

        # Initialize parameter list to hold skip connection scaling factor 'alpha'
        # In the forward pass alpha**2 will be used to ensure positive scaling
        self.alphas = nn.ParameterList()

        for i in range(layer_count):
            self.linear_layers.append(Linear_class(geometry_dim, geometry_dim))
            if act_fn is not None:
                self.act_layers.append(act_fn())
            else:
                self.act_layers.append(copy.deepcopy(monotonic_net))
            
            self.alphas.append(nn.Parameter(torch.tensor(1.0)))
        
        self.final_linear = Linear_class(geometry_dim, geometry_dim)


    def forward(self, T: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the NIGnet.

        Parameters
        ----------
        t : torch.Tensor
            A tensor of shape (N, 1) with values in [0, 1] representing the parameter for the simple
            closed curve.
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, 2) containing 2D coordinates of points on the simple closed
            curve represented by the network.
        """

        if self.geometry_dim == 2:
            t = T
            X = self.closed_transform(t)
        elif self.geometry_dim == 3:
            t, s = T[:, 0:1], T[:, 1:2]
            X = self.closed_transform(t, s)

        for i, (linear_layer, act_layer) in enumerate(zip(self.linear_layers, self.act_layers)):
            # Apply linear transformation
            X = linear_layer(X)

            if self.skip_connections:
                residual = X
            
            if self.act_fn is not None:
                X = act_layer(X)
            else:
                # Apply activation function or monotonic network to each component of x separately
                x1, x2 = X[:, 0:1], X[:, 1:2]
                X = torch.stack([act_layer(x1), act_layer(x2)], dim = -1)
            
            if self.skip_connections:
                alpha_sq = self.alphas[i] ** 2
                X = (X + alpha_sq * residual) / 2.0
        
        X = self.final_linear(X)

        return X


    def generate_noisy_shapes(
        self,
        noise_amount: float,
        num_generations: int,
        num_pts: int = 1000
    ) -> None:
        """
        Generate and plot noisy variations of the original NIGnet shape using four different
        visualizations.

        Parameters
        ----------
        noise_amount : float
            Standard deviation of the Gaussian noise added to the weight matrices.
        num_generations : int
            Number of noisy versions to generate.
        num_pts : int, optional
            Number of points sampled along the curve, default is 1000.
        """
        
        t = torch.linspace(0, 1, num_pts).reshape(-1, 1)

        # Compute original shape
        original_shape = self(t).detach().cpu()

        # Store all noisy shapes
        noisy_shapes = torch.zeros((num_generations, num_pts, 2))

        for i in range(num_generations):
            # Create a noisy copy of the network
            noisy_net = copy.deepcopy(self)
            for param in noisy_net.parameters():
                param.data += torch.randn_like(param) * noise_amount
            
            # Generate the noisy shape and store its x and y components separately
            noisy_shapes[i] = noisy_net(t).detach().cpu()

        # Compute the mean and standard deviation across noisy shapes
        mean_shape = torch.mean(noisy_shapes, axis = 0)
        std_shape = torch.std(noisy_shapes, axis = 0)

        # Create a 4x1 grid of subplots
        fig, axes = plt.subplots(4, 1, figsize = (6, 24))

        # Common plot settings
        plot_kwargs = {
            'original': {'color': 'k', 'lw': 3, 'label': 'Original Shape'},
            'mean': {'color': 'r', 'lw': 1, 'ls': '--', 'label': 'Mean Shape'},
            'fill': {'color': 'grey', 'alpha': 0.1},
        }

        # Subplot 1: Noisy shapes
        ax = axes[0]
        for shape in noisy_shapes:
            ax.plot(shape[:, 0], shape[:, 1], alpha = 0.3, linewidth = 0.8)
        ax.plot(original_shape[:, 0], original_shape[:, 1], **plot_kwargs['original'])
        ax.plot(mean_shape[:, 0], mean_shape[:, 1], **plot_kwargs['mean'])
        ax.set_title('Noisy Shapes')

        # Subplot 2: Mean ± Std Variation using fill_betweenx
        ax = axes[1]
        ax.fill_betweenx(
            mean_shape[:, 1],
            mean_shape[:, 0] - std_shape[:, 0],
            mean_shape[:, 0] + std_shape[:, 0],
            color = 'blue', alpha = 0.3, label = 'X Variation (Mean ± Std)'
        )
        ax.fill_between(
            mean_shape[:, 0],
            mean_shape[:, 1] - std_shape[:, 1],
            mean_shape[:, 1] + std_shape[:, 1],
            color = 'green', alpha = 0.3, label = 'Y Variation (Mean ± Std)')
        ax.plot(original_shape[:, 0], original_shape[:, 1], **plot_kwargs['original'])
        ax.plot(mean_shape[:, 0], mean_shape[:, 1], **plot_kwargs['mean'])
        ax.set_title('Mean ± Std Variation')

        # Subplot 3: Individual Filled Noisy Shapes
        ax = axes[2]
        for i in range(num_generations):
            ax.fill(noisy_shapes[i, :, 0], noisy_shapes[i, :, 1], **plot_kwargs['fill'])
        ax.plot(original_shape[:, 0], original_shape[:, 1], **plot_kwargs['original'])
        ax.plot(mean_shape[:, 0], mean_shape[:, 1], **plot_kwargs['mean'])
        ax.set_title('Noisy Shapes Filled')

        # Subplot 4: Plot envelope
        ax = axes[3]
        for i in range(num_generations):
            ax.fill(noisy_shapes[i, :, 0], noisy_shapes[i, :, 1], color = 'skyblue')
        ax.plot(original_shape[:, 0], original_shape[:, 1], **plot_kwargs['original'])
        ax.plot(mean_shape[:, 0], mean_shape[:, 1], **plot_kwargs['mean'])
        ax.set_title('Noisy Shapes Envelope')

        for i in range(4):
            ax = axes[i]
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.axis('equal')
            ax.grid(True, alpha = 0.5)
            ax.legend()

        plt.tight_layout()
        plt.show()


class ExpLinear(nn.Module):
    """Linear layer with matrix-exponentiated weight transformation.

    Performs linear transformation using W = exp(weight_matrix) instead of direct weight matrix
    multiplication. This ensures injectivity of the transformation.

    Attributes
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    W : torch.nn.Parameter
        The learnable weight matrix, which is exponentiated in the forward pass.
    bias : torch.nn.Parameter or None
        Optional learnable bias term for each output feature.

    Parameters
    ----------
    in_features : int
        Number of features in the input.
    out_features : int
        Number of features in the output.
    bias : bool, optional
        If True, a learnable bias is included, by default True.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """
        Initialize the ExpLinear module.
        
        Raises
        ------
        AssertionError
            If in_features != out_features.
        """
        
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
    

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the layer using Kaiming uniform initialization for the weight
        matrix, and a uniform distribution for the bias.
        """

        nn.init.kaiming_uniform_(self.W, a = math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of ExpLinear.

        The weight matrix `self.W` is exponentiated using `torch.matrix_exp` before being applied to
        the input `x`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_features).
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_features)
        """
        
        exp_weight = torch.matrix_exp(self.W)
        return F.linear(x, exp_weight.t(), self.bias)