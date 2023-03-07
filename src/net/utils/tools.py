from src.net.utils.hyperbolic import MobiusLinear, HyperMapper
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.functional
from collections import OrderedDict


class EMA():
    """Exponential moving average."""

    def __init__(self, beta):
        """Initialize the EMA.

        Args:
            beta (float): Decay rate.
        """
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


class HyperMetrics(object):
    """Compute metrics for embeddings in euclidean and hyperbolic space.

    Args:
        c (float, optional): Hyperbolic curvature. Defaults to 1.0
    """
    def __init__(self, c=1.) -> None:
        self.c = c
        self.mapper = HyperMapper(c=self.c)

    def compute(self, x, y):
        metrics = {}

        # MSE and Cosine distance
        metrics['mse'] = F.mse_loss(x, y)
        metrics['cosine_dist'] = self.mapper.cosine_distance(x, y)

        # Project embeddings to hyperbolic space
        x_h = self.mapper.expmap(x)
        y_h = self.mapper.expmap(y)

        # Radii in hyperbolic space
        radius_x = torch.linalg.norm(x_h, dim=-1)
        radius_y = torch.linalg.norm(y_h, dim=-1)
        metrics['radius_x'] = radius_x
        metrics['radius_y'] = radius_y

        # Angle between embeddings
        x_norm_e = x_h / radius_x.reshape(-1, 1)
        y_norm_e = y_h / radius_y.reshape(-1, 1)
        ang_e = torch.acos((x_norm_e * y_norm_e).sum(dim=-1)) * 180/np.pi
        metrics['ang_e'] = ang_e

        # Poincare distance between embeddings
        metrics['poincare_dist'] = self.mapper.poincare_distance(x_h, y_h)

        return metrics


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
