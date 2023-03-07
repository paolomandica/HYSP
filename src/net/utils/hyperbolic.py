import geoopt
import geoopt.manifolds.stereographic.math as gmath
import numpy as np
import torch.nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch


class HyperMapper(object):
    """A class to map between euclidean and hyperbolic space and compute distances."""

    def __init__(self, c=1.) -> None:
        """Initialize the hyperbolic mapper.

        Args:
            c (float, optional): Hyperbolic curvature. Defaults to 1.0
        """
        self.c = c
        self.K = torch.tensor(-self.c, dtype=float)

    def expmap(self, x):
        """Exponential mapping from Euclidean to hyperbolic space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (..., d)
        """
        return gmath.project(gmath.expmap0(x.double(), k=self.K), k=self.K)

    def logmap(self, x):
        """Logarithmic mapping from hyperbolic to Euclidean space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (..., d)
        """
        return gmath.project(gmath.logmap0(x.double(), k=self.K), k=self.K)

    def poincare_distance(self, x, y):
        """Poincare distance between two points in hyperbolic space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)
            y (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (...)
        """
        return gmath.dist(x, y, k=self.K)

    def cosine_distance(self, x, y):
        """Cosine distance between two points.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)
            y (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (...)
        """
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)



"""
Network definitions from https://github.com/ferrine/hyrnn
"""

def mobius_linear(
        input,
        weight,
        bias=None,
        hyperbolic_input=True,
        hyperbolic_bias=True,
        nonlin=None,
        k=-1.0,
):
    if not isinstance(k, torch.Tensor):
        k = torch.tensor(k)
    if hyperbolic_input:
        output = mobius_matvec(weight.double(), input.double(), k=k)
    else:
        output = F.linear(input.double(), weight.double())
        output = gmath.expmap0(output, k=k)
    if bias is not None:
        if not hyperbolic_bias:
            bias = gmath.expmap0(bias, k=k)
        output = gmath.mobius_add(output, bias.unsqueeze(0).expand_as(output), k=k)
    if nonlin is not None:
        output = gmath.mobius_fn_apply(nonlin, output, k=k)
    output = gmath.project(output, k=k)
    return output


def mobius_matvec(m: torch.Tensor, x: torch.Tensor, *, k: torch.Tensor, dim=-1):
    return _mobius_matvec(m, x, k, dim=dim)


def _mobius_matvec(m: torch.Tensor, x: torch.Tensor, k: torch.Tensor, dim: int = -1):
    if m.dim() > 2 and dim != -1:
        raise RuntimeError(
            "broadcasted Möbius matvec is supported for the last dim only"
        )
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    if dim != -1 or m.dim() == 2:
        # mx = torch.tensordot(x, m, [dim], [1])
        mx = torch.matmul(m, x.transpose(1, 0)).transpose(1, 0)
    else:
        mx = torch.matmul(m, x.unsqueeze(-1)).squeeze(-1)
    mx_norm = mx.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    res_c = gmath.tan_k(mx_norm / x_norm * gmath.artan_k(x_norm, k), k) * (mx / mx_norm)
    cond = (mx == 0).prod(dim=dim, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return res


class MobiusLinear(torch.nn.Linear):
    def __init__(
            self,
            *args,
            hyperbolic_input=True,
            hyperbolic_bias=True,
            nonlin=None,
            k=-1.0,
            fp64_hyper=True,
            **kwargs
    ):
        k = torch.tensor(k)
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            if hyperbolic_bias:
                self.ball = manifold = geoopt.PoincareBall(c=k.abs())
                self.bias = geoopt.ManifoldParameter(self.bias, manifold=manifold)
                with torch.no_grad():
                    # self.bias.set_(gmath.expmap0(self.bias.normal_() / 4, k=k))
                    self.bias.set_(gmath.expmap0(self.bias.normal_() / 400, k=k))
        with torch.no_grad():
            # 1e-2 was the original value in the code. The updated one is from HNN++
            std = 1 / np.sqrt(2 * self.weight.shape[0] * self.weight.shape[1])
            # Actually, we divide that by 100 so that it starts really small and far from the border
            std = std / 100
            self.weight.normal_(std=std)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin
        self.k = k
        self.fp64_hyper = fp64_hyper

    def forward(self, input):
        if self.fp64_hyper:
            input = input.double()
        else:
            input = input.float()
        with autocast(enabled=False):  # Do not use fp16
            return mobius_linear(
                input,
                weight=self.weight,
                bias=self.bias,
                hyperbolic_input=self.hyperbolic_input,
                nonlin=self.nonlin,
                hyperbolic_bias=self.hyperbolic_bias,
                k=self.k,
            )

    def extra_repr(self):
        info = super().extra_repr()
        info += "c={}, hyperbolic_input={}".format(self.ball.c, self.hyperbolic_input)
        if self.bias is not None:
            info = ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        return info
