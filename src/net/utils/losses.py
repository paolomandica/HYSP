import torch
import torch.nn as nn
import torch.nn.functional as F

from src.net.utils.tools import HyperMapper


class PositiveOnlyLoss():
    """A class to compute the positive only loss."""

    def __init__(self, hyper=True, c=1.) -> None:
        """Initialize the positive only loss.

        Args:
            hyper (bool, optional): Whether to use hyperbolic loss. Defaults to True.
            c (float, optional): Hyperbolic curvature. Defaults to 1.0.
        """
        self.hyper = hyper
        self.mapper = HyperMapper(c=c)

    def compute(self, x, y, lambda_hyper=1.):
        if self.hyper:
            # expmap of embeddings into poincare ball
            x = self.mapper.expmap(x)
            y = self.mapper.expmap(y)

            # compute euclidean and hyperbolic loss
            if lambda_hyper <= 1e-7:
                loss_euc = self.mapper.cosine_distance(x, y)
                loss_hyp = torch.zeros_like(loss_euc)
            elif lambda_hyper >= (1 - 1e-7):
                loss_hyp = self.mapper.poincare_distance(x, y)
                loss_euc = torch.zeros_like(loss_hyp)
            else:
                loss_euc = self.mapper.cosine_distance(x, y)
                loss_hyp = self.mapper.poincare_distance(x, y)

            return loss_euc, loss_hyp

        else:
            # compute euclidean loss
            return self.mapper.cosine_distance(x, y)


class HyperCrossEntropyLoss():
    """A class to compute the cross entropy loss in hyperbolic space."""

    def __init__(self, c=1., temperature=0.07) -> None:
        """Initialize the hyperbolic cross entropy loss.

        Args:
            c (float, optional): Hyperbolic curvature. Defaults to 1.0.
            temperature (float, optional): Temperature for the softmax. Defaults to 0.07.
        """
        self.c = c
        self.temperature = temperature
        self.mapper = HyperMapper(c=c)
        self.criterion = nn.CrossEntropyLoss()

    def _get_hyper_logits(self, q, k, queue):
        queue = queue.permute(1, 0)

        # project embeddings into poincare ball
        q = self.mapper.expmap(q)
        k = self.mapper.expmap(k)
        queue = self.mapper.expmap(queue)

        # compute loss
        queue = queue.unsqueeze(0).repeat(k.shape[0], 1, 1)
        k = k.unsqueeze(1)
        y = torch.cat([k, queue], dim=1)
        q = q.unsqueeze(1).repeat(1, y.shape[1], 1)
        # logits = self.distances.poincare_distance(q, y) / self.temperature
        sqdist = torch.sum((q - y)**2, dim=-1)
        squnorm = torch.sum(q**2, dim=-1)
        sqvnorm = torch.sum(y**2, dim=-1)
        x_temp = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1e-7
        logits = torch.acosh(x_temp)
        logits /= self.temperature

        return -logits

    def _get_euclidean_logits(self, q, k, queue):
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, queue])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        return logits

    def _get_cross_entropy_loss(self, q, k, queue, hyper=True):

        if not hyper:
            # Normalize the feature
            q = F.normalize(q, dim=1)
            queue = F.normalize(queue, dim=0)

            with torch.no_grad():
                k = F.normalize(k, dim=1)

            logits = self._get_euclidean_logits(q, k, queue)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        else:
            logits = self._get_hyper_logits(q, k, queue)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = self.criterion(logits, labels)
        return loss

    def compute(self, q, k, queue, lambda_hyper=1.):
        if lambda_hyper <= 1e-7:
            loss_euc = self._get_cross_entropy_loss(q, k, queue, hyper=False)
            loss_hyp = torch.zeros_like(loss_euc)
        elif lambda_hyper >= (1 - 1e-7):
            loss_hyp = self._get_cross_entropy_loss(q, k, queue, hyper=True)
            loss_euc = torch.zeros_like(loss_hyp)
        else:
            loss_euc = self._get_cross_entropy_loss(q, k, queue, hyper=False)
            loss_hyp = self._get_cross_entropy_loss(q, k, queue, hyper=True)

        return loss_euc, loss_hyp
