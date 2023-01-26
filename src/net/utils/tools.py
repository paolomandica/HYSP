from src.net.hyperbolic import MobiusLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt.manifolds.stereographic.math as gmath
import numpy as np
import torch.nn.functional
from collections import OrderedDict

from src.net.utils.hyper_math import dist as poincare_dist


class Distances():
    def __init__(self, hyper_c=1.) -> None:
        self.hyper_c = hyper_c

    def cosine_distance(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def cosine_similarity(self, x, y):
        criterion = nn.CosineSimilarity(dim=-1)
        return criterion(x, y).mean()

    def poincare_distance_no_clip(self, x, y):
        """ Old poincare distance with hyper_c = 1 """
        sqdist = torch.sum((x - y)**2, dim=-1)
        squnorm = torch.sum(x**2, dim=-1)
        sqvnorm = torch.sum(y**2, dim=-1)
        x_temp = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1e-7
        return torch.acosh(x_temp)

    def poincare_distance(self, x, y):
        # return poincare_dist(x, y, c=self.hyper_c)
        return self.poincare_distance_no_clip(x, y)

    def project(self, x):
        # expmap of embedding into poincare ball
        K = torch.tensor(-self.hyper_c, dtype=float)
        return gmath.project(gmath.expmap0(x.double(), k=K), k=K)

    def inverse_project(self, x):
        # expmap of embedding into poincare ball
        K = torch.tensor(-self.hyper_c, dtype=float)
        return gmath.project(gmath.logmap0(x.double(), k=K), k=K)


class PositiveOnlyLoss():
    def __init__(self, hyper=True, hyper_c=1.) -> None:
        self.hyper = hyper
        self.hyper_c = hyper_c
        self.distances = Distances(hyper_c=hyper_c)

    def compute(self, x, y, lambda_hyper=1.):
        if self.hyper:
            # expmap of embeddings into poincare ball
            x = self.distances.project(x)
            y = self.distances.project(y)

            # compute euclidean and hyperbolic loss
            if lambda_hyper <= 1e-7:
                loss_euc = self.distances.cosine_distance(x, y)
                loss_hyp = torch.zeros_like(loss_euc)
            elif lambda_hyper >= (1 - 1e-7):
                loss_hyp = self.distances.poincare_distance(x, y)
                loss_euc = torch.zeros_like(loss_hyp)
            else:
                loss_euc = self.distances.cosine_distance(x, y)
                loss_hyp = self.distances.poincare_distance(x, y)

            return loss_euc, loss_hyp

        else:
            # compute euclidean loss
            return self.distances.cosine_distance(x, y)


class HyperCrossEntropyLoss():
    def __init__(self, hyper_c=1., temperature=0.07) -> None:
        self.hyper_c = hyper_c
        self.temperature = temperature
        self.distances = Distances(hyper_c=hyper_c)
        self.criterion = nn.CrossEntropyLoss()

    def _get_hyper_logits(self, q, k, queue):
        queue = queue.permute(1, 0)

        # project embeddings into poincare ball
        q = self.distances.project(q)
        k = self.distances.project(k)
        queue = self.distances.project(queue)

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


class EMA():
    """Exponential moving average."""

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def update_moving_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)


class MLP(nn.Module):
    """MLP class for projector and predictor."""

    def __init__(self, input_size, hidden_size, hyper=False, bias=True):
        super().__init__()

        n_layers = len(hidden_size)
        layer_list = []

        if not hyper:
            for idx, next_dim in enumerate(hidden_size):

                if idx == n_layers-1:
                    layer_list.append(nn.Linear(input_size, next_dim, bias=bias))
                    # if bias is False:
                    #     layer_list.append(nn.BatchNorm1d(next_dim, affine=False))
                else:
                    layer_list.append(nn.Linear(input_size, next_dim, bias=bias))
                    layer_list.append(nn.BatchNorm1d(next_dim))
                    layer_list.append(nn.ReLU(inplace=True))
                    input_size = next_dim

        else:
            for idx, next_dim in enumerate(hidden_size):
                layer_list.append(MobiusLinear(input_size, next_dim))
                input_size = next_dim

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)


def compute_metrics(x, y, hyper_c=1.):
    distances = Distances(hyper_c=hyper_c)

    euc_dist = F.mse_loss(x, y)

    euc_norm1 = torch.linalg.norm(x, dim=-1)
    euc_norm2 = torch.linalg.norm(y, dim=-1)
    cosine_dist = distances.cosine_distance(x, y)

    x = distances.project(x)
    y = distances.project(y)

    radius_x = torch.linalg.norm(x, dim=-1)
    radius_y = torch.linalg.norm(y, dim=-1)

    x_norm_e = x / torch.linalg.norm(x, dim=-1).reshape(-1, 1)
    y_norm_e = y / torch.linalg.norm(y, dim=-1).reshape(-1, 1)
    ang_e = torch.acos((x_norm_e * y_norm_e).sum(dim=-1)) * 180/np.pi

    poincare_dist = distances.poincare_distance(x, y)

    return euc_norm1, euc_norm2, radius_x, radius_y, ang_e, cosine_dist, poincare_dist, euc_dist


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def load_weights(model, weights_path, ignore_weights=None, protocol=None):
    # model.apply(weights_init)

    if weights_path != 'None':
        print("Loading weights from {}".format(weights_path))
        weights = torch.load(weights_path, map_location='cpu')['state_dict']
        weights = OrderedDict([[k.split('model.')[-1],
                                v.cpu()] for k, v in weights.items()])

        # filter weights
        if ignore_weights is not None:
            for i in ignore_weights:
                ignore_name = list()
                for w in weights:
                    if w.find(i) == 0:
                        ignore_name.append(w)
                for n in ignore_name:
                    weights.pop(n)
                    print('Filter [{}] remove weights [{}].'.format(i, n))

        try:
            model.load_state_dict(weights)
        except (KeyError, RuntimeError):
            state = model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            for d in diff:
                print('Can not find weights [{}].'.format(d))
            state.update(weights)
            model.load_state_dict(state)

    if protocol is not None:
        assert protocol in ['linear', 'semi', 'supervised',
                            'unsupervised'], "Choose a valid evaluation protocol"

        if protocol in ['linear', 'unsupervised']:
            try:
                for param in model.online_encoder.parameters():
                    param.requires_grad = False
            except:
                for param in model.encoder.parameters():
                    param.requires_grad = False
