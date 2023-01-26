"""
Network definitions from https://github.com/ferrine/hyrnn
"""

import geoopt
import geoopt.manifolds.stereographic.math as gmath
import numpy as np
import torch.nn
import torch.nn.functional
from torch.cuda.amp import autocast
import torch


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
        output = torch.nn.functional.linear(input.double(), weight.double())
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


class MobiusDist2Hyperplane(torch.nn.Module):
    def __init__(self, in_features, out_features, k=-1.0, fp64_hyper=True):
        k = torch.tensor(k)
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = ball = geoopt.PoincareBall(c=k.abs())
        self.sphere = sphere = geoopt.manifolds.Sphere()
        self.scale = torch.nn.Parameter(torch.zeros(out_features))
        point = torch.randn(out_features, in_features) / 4
        point = gmath.expmap0(point, k=k)
        tangent = torch.randn(out_features, in_features)
        self.point = geoopt.ManifoldParameter(point, manifold=ball)
        self.fp64_hyper = fp64_hyper
        with torch.no_grad():
            self.tangent = geoopt.ManifoldParameter(tangent, manifold=sphere).proj_()

    def forward(self, input):
        if self.fp64_hyper:
            input = input.double()
        else:
            input = input.float()
        with autocast(enabled=False):  # Do not use fp16
            input = input.unsqueeze(-2)
            distance = gmath.dist2plane(
                x=input, p=self.point, a=self.tangent, k=self.ball.c, signed=True
            )
            return distance * self.scale.exp()

    def extra_repr(self):
        return (
            "in_features={in_features}, out_features={out_features}"
            #             "c={ball.c}".format(
            #                 **self.__dict__
            #             )
        )


def square_norm(x):
    """
    Helper function returning square of the euclidean norm.
    Also here we clamp it since it really likes to die to zero.
    """
    norm = torch.norm(x, dim=-1, p=2) ** 2
    return torch.clamp(norm, min=1e-5)


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)8
    return torch.clamp(dist, 1e-7, np.inf)


######### FRECHET MEAN #########

class FrechtMean(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def arcosh(self, x):
        x = torch.clamp(x, min=1+1e-6)
        z = torch.sqrt(x * x - 1)
        return torch.log(x + z)

    # def arcosh(self, x, eps=1e-5):  # pragma: no cover
    #     x = x.clamp(-1 + eps, 1 - eps)
    #     return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))

    def l_prime(self, y):
        # pdb.set_trace()
        y = torch.clamp(y, min=1e-6)  # Aggiunto da Luca per evitare divisione per zero

        cond = y < 1e-6
        val = 4 * torch.ones_like(y)
        numerat = self.arcosh(1 + 2 * y)
        denomin = (y.pow(2) + y).sqrt()
        frac = 2 * numerat/denomin
        ret = torch.where(cond, val, frac)
        return ret

    def forward(self, X, w=None, K=-1.0, max_iter=1000, rtol=1e-6, atol=1e-6):
        """
        Args
        ----
            X (tensor): point of shape [..., points, dim]
            w (tensor): weights of shape [..., points]
            K (float): curvature (must be negative)
        Returns
        -------
            frechet mean (tensor): shape [..., dim]
        """

        self.w = w
        self.K = K
        self.max_iter = max_iter
        self.rtol = rtol
        self.atol = atol

        mu = X[..., 0, :].clone()

        x_ss = X.pow(2).sum(dim=-1)
        # pdb.set_trace()

        if self.w is None:
            w = torch.ones(X.shape[0]).to(X.device)

        mu_prev = mu
        iters = 0
        for _ in range(self.max_iter):
            mu_ss = mu.pow(2).sum(dim=-1)
            xmu_ss = (X - mu.unsqueeze(-2)).pow(2).sum(dim=-1)

            # print("mu_ss", mu_ss)
            # print("xmu_ss", xmu_ss)

            alphas = self.l_prime(-self.K * xmu_ss / ((1 + self.K * x_ss) *
                                  (1 + self.K * mu_ss.unsqueeze(-1)))) / (1 + self.K * x_ss)
            # print("Cancheros", -self.K * xmu_ss / ((1 + self.K * x_ss) * (1 + self.K * x_ss)))
            # print("alphas", alphas)

            alphas = alphas * w

            c = (alphas * x_ss).sum(dim=-1)
            b = (alphas.unsqueeze(-1) * X).sum(dim=-2)
            a = alphas.sum(dim=-1)
            # print("c", c)
            # print("b", b)
            # print("a", a)

            b_ss = b.pow(2).sum(dim=-1)

            eta = (a - self.K * c - ((a - self.K * c).pow(2) + 4 *
                   self.K * b_ss).sqrt()) / (2 * (-self.K) * b_ss)
            # print("eta", eta)

            mu = eta.unsqueeze(-1) * b

            dist = (mu - mu_prev).norm(dim=-1)
            # print("dist", dist)
            prev_dist = mu_prev.norm(dim=-1)
            if (dist < self.atol).all() or (dist / prev_dist < self.rtol).all():
                break

            mu_prev = mu
            iters += 1

        return mu

    def darcosh(self, x):
        cond = (x < 1 + 1e-7)
        x = torch.where(cond, 2 * torch.ones_like(x), x)
        x = torch.where(~cond, 2 * self.arcosh(x) / torch.sqrt(x**2 - 1), x)
        return x

    def d2arcosh(self, x):
        cond = (x < 1 + 1e-7)
        x = torch.where(cond, -2/3 * torch.ones_like(x), x)
        x = torch.where(~cond, 2 / (x**2 - 1) - 2 * x * self.arcosh(x) / ((x**2 - 1)**(3/2)), x)
        return x

    def grad_var(self, X, y, w, K):
        """
        Args
        ----
            X (tensor): point of shape [..., points, dim]
            y (tensor): mean point of shape [..., dim]
            w (tensor): weight tensor of shape [..., points]
            K (float): curvature (must be negative)

        Returns
        -------
            grad (tensor): gradient of variance [..., dim]
        """
        yl = y.unsqueeze(-2)
        xnorm = 1 + K * X.norm(dim=-1).pow(2)
        ynorm = 1 + K * yl.norm(dim=-1).pow(2)
        xynorm = (X - yl).norm(dim=-1).pow(2)

        D = xnorm * ynorm
        v = 1 - 2 * K * xynorm / D

        Dl = D.unsqueeze(-1)
        vl = v.unsqueeze(-1)

        first_term = (X - yl) / Dl
        sec_term = K / Dl.pow(2) * yl * xynorm.unsqueeze(-1) * xnorm.unsqueeze(-1)
        return -(4 * self.darcosh(vl) * w.unsqueeze(-1) * (first_term + sec_term)).sum(dim=-2)

    def inverse_hessian(self, X, y, w, K):
        """
        Args
        ----
            X (tensor): point of shape [..., points, dim]
            y (tensor): mean point of shape [..., dim]
            w (tensor): weight tensor of shape [..., points]
            K (float): curvature (must be negative)

        Returns
        -------
            inv_hess (tensor): inverse hessian of [..., points, dim, dim]
        """
        yl = y.unsqueeze(-2)
        xnorm = 1 + K * X.norm(dim=-1).pow(2)
        ynorm = 1 + K * yl.norm(dim=-1).pow(2)
        xynorm = (X - yl).norm(dim=-1).pow(2)

        D = xnorm * ynorm
        v = 1 - 2 * K * xynorm / D

        Dl = D.unsqueeze(-1)
        vl = v.unsqueeze(-1)
        vll = vl.unsqueeze(-1)

        """
        \partial T/ \partial y
        """
        first_const = -8 * (K ** 2) * xnorm / D.pow(2)
        matrix_val = (first_const.unsqueeze(-1) * yl).unsqueeze(-1) * (X - yl).unsqueeze(-2)
        first_term = matrix_val + matrix_val.transpose(-1, -2)

        sec_const = -16 * (K ** 3) * xnorm.pow(2) / D.pow(3) * xynorm
        sec_term = (sec_const.unsqueeze(-1) * yl).unsqueeze(-1) * yl.unsqueeze(-2)

        third_const = -4 * K / D + 4 * (K ** 2) * xnorm / D.pow(2) * xynorm
        third_term = third_const.reshape(*third_const.shape, 1, 1) * torch.eye(y.shape[-1]).to(
            X).reshape((1, ) * len(third_const.shape) + (y.shape[-1], y.shape[-1]))

        Ty = first_term + sec_term + third_term

        """
        T
        """

        first_term = K / Dl * (X - yl)
        sec_term = K.pow(2) / Dl.pow(2) * yl * xynorm.unsqueeze(-1) * xnorm.unsqueeze(-1)
        T = 4 * (first_term + sec_term)

        """
        inverse of shape [..., points, dim, dim]
        """
        first_term = self.d2arcosh(vll) * T.unsqueeze(-1) * T.unsqueeze(-2)
        sec_term = self.darcosh(vll) * Ty
        hessian = ((first_term + sec_term) * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=-3) / -K
        inv_hess = torch.inverse(hessian)
        return inv_hess

    def backward(self, X, y, grad, w, K):
        """
        Args
        ----
            X (tensor): point of shape [..., points, dim]
            y (tensor): mean point of shape [..., dim]
            grad (tensor): gradient
            K (float): curvature (must be negative)

        Returns
        -------
            gradients (tensor, tensor, tensor): 
                gradient of X [..., points, dim], weights [..., dim], curvature []
        """
        if not torch.is_tensor(K):
            K = torch.tensor(K).to(X)

        with torch.no_grad():
            inv_hess = self.inverse_hessian(X, y, w=w, K=K)

        with torch.enable_grad():
            # clone variables
            X = torch.nn.Parameter(X.detach())
            y = y.detach()
            w = torch.nn.Parameter(w.detach())
            K = torch.nn.Parameter(K)

            grad = (inv_hess @ grad.unsqueeze(-1)).squeeze()
            gradf = self.grad_var(X, y, w, K)
            dx, dw, dK = torch.autograd.grad(-gradf, (X, w, K), grad)

        return dx, dw, dK
