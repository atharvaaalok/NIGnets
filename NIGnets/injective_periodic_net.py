from typing import Any, Callable, Optional
import torch
from torch import Tensor
from torch.nn import Module, ReLU, Linear

from .monotonic_nets import MonoNet

__all__ = ["Injective2DNet"]

class Injective2DNet(Module):
    def __init__(
        self,
        *,
        n_hiden_layers: int,
        hidden_width: int, 
        act_fn: Optional[Callable[..., Any]] = None,
        base_fn: Optional[Callable[..., Any]] = None
    ) -> None:
        """A 2D injective network that maps the unit interval to an injective function.

        Args:
            n_hiden_layers (int): Number of hidden layers.
            hidden_width (int): Width of the hidden layers.
            act_fn (Optional[Callable[..., Any]], optional): Activation function. If None (default), `torch.nn.ReLU()` will be used.
            base_fn (Optional[Callable[..., Any]], optional): Base function for the periodic part. If None (default), `torch.sin` will be used.
        """
        super().__init__()

        act_fn = act_fn or ReLU()
        base_fn = base_fn or torch.sin

        # We need 4 shifts to make the function injective
        # because we do not know if a function if monotonically
        # increasing or decreasing for each dimension.
        # The simplest way to do it is just to use 4 shifts since
        # functions in pairs of them will be just negations of each other.
        shifts = [0.25*i for i in range(4)]

        # Define the transformation from t on the [0, 1] interval to unit circle
        def closed_transform(t: Tensor) -> list[Tensor]:
            xs = [base_fn(2 * torch.pi * (t-shift)) for shift in shifts]
            return torch.hstack(xs)

        self.closed_transform = closed_transform

        # defines a 2D function that is monotonically increasing with respect to all input dimensions
        self.mono_mixed_transforms = MonoNet(
            in_features=len(shifts),
            out_features=2,
            hidden_widths=[hidden_width]*(n_hiden_layers),
            act_fn=act_fn
        )
    
    def forward(self, t):
        h = t
        h = self.closed_transform(h)
        y = self.mono_mixed_transforms(h)
        return y
