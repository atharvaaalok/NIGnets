from typing import Any, Callable, Literal, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.nn import functional as F, Linear, Module, Sequential

__all__ = ["ConstrainedLinear", "MonoActivation", "MonoDense", "MonoNet"]

class ConstrainedLinear(Linear):
    r"""Applies an affine linear transformation to the incoming data, but using absolute values of weights: :math:`y = x|A|^T + b`.

    Examples::

        >>> m = MonoLinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device=None,
        dtype: torch.dtype=None,
    ) -> None:
        """Applies an affine linear transformation to the incoming data, but using absolute values of weights: :math:`y = x|A|^T + b`.

        Args:
            in_features (int): size of each input sample
            out_features (int): size of each output sample
            bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: `True`
            device (torch.device, optional): the desired device of the weights and bias. Default: None
            dtype (torch.dtype, optional): the desired data type of the weights and bias. Default: None
        """
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, torch.abs(self.weight), self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class MonoActivation(Module):
    """Applies an activation function to the incoming data, but with a specified curvature.

    Args:
        act_fn (Optional[Callable[..., Any]]): Activation function. If None, the identity function will be used.
        curvature (Optional[Literal["convex", "concave"]]): Curvature of the activation function. If None, no curvature will be enforced (both convex and concave parts will be used).
    
    """
    def __init__(
        self,
        act_fn: Optional[Callable[..., Any]]=None,
        *,
        curvature: Optional[Literal["convex", "concave"]] = None
    ) -> None:
        
        super().__init__()
        self.act_fn = act_fn
        self.curvature = curvature

    def forward(self, input: Tensor) -> Tensor:
        if self.act_fn is None:
            return input
        
        if self.curvature == "convex":
            return self.act_fn(input)
        
        if self.curvature == "concave":
            return -self.act_fn(-input)
        
        k = (input.shape[1]+1) // 2
        x, y = input[:, :k], input[:, k:]
        return torch.hstack([self.act_fn(x), -(self.act_fn(-y))])
        
    def extra_repr(self) -> str:
        return f"{self.act_fn=}, {self.curvature=}"
        
class MonoDense(Module):
    """A constrained linear layer followed by a monotonic activation function.

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: `True`
        device (torch.device, optional): the desired device of the weights and bias. Default: None
        dtype (torch.dtype, optional): the desired data type of the weights and bias. Default: None
        act_fn (Optional[Callable[..., Any]]): Activation function. If None, the identity function will be used.
        curvature (Optional[Literal["convex", "concave"]]): Curvature of the activation function. If None, no curvature will be enforced (both convex and concave parts will be used).
    
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device=None,
        dtype: torch.dtype=None,
        *,
        act_fn: Optional[Callable[..., Any]],
        curvature: Optional[Literal["convex", "concave"]] = None,
    ) -> None:
        super().__init__()
        self.linear = ConstrainedLinear(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
        self.activation = MonoActivation(act_fn=act_fn, curvature=curvature)

    def forward(self, input: Tensor) -> Tensor:
        return self.activation(self.linear(input))


class MonoNet(Sequential):
    """A sequence of constrained linear layers followed by monotonic activation functions.

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        hidden_widths (Sequence[int]): Widths of the hidden layers.
        act_fn (Union[Callable[..., Any], Sequence[Callable[..., Any]]]): Activation function. If a single function is provided, it will be used for all hidden layers.
        device (torch.device, optional): the desired device of the weights and bias. Default: None
        dtype (torch.dtype, optional): the desired data type of the weights and bias. Default: None
    
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_widths: Sequence[int],
        act_fn: Union[Callable[..., Any], Sequence[Callable[..., Any]]],
        device: torch.device=None,
        dtype: torch.dtype=None,
    ):
        if len(hidden_widths) < 1:
            raise ValueError(f"Number of hidden layers must be greater than 0, got {len(hidden_widths)=}")
        
        if isinstance(act_fn, Callable):
            act_fn = tuple(act_fn for _ in range(len(hidden_widths)))

        if len(hidden_widths) != len(act_fn):
            raise ValueError(f"The number of activation functions and the number of hidden layer width must be the same, got {len(hidden_widths)=}, {len(act_fn)=}")
        
        hidden_layers = (
            MonoDense(in_features=in_features, out_features=hidden_widths[0], act_fn=act_fn[0], device=device, dtype=dtype),
            *(MonoDense(in_features=hidden_widths[i], out_features=hidden_widths[i+1], act_fn=act_fn[i+1]) for i in range(len(hidden_widths)-1)),
        )
        output_layer = MonoDense(in_features=hidden_widths[-1], out_features=out_features, act_fn=None, device=device, dtype=dtype)

        super().__init__(*hidden_layers, output_layer)
