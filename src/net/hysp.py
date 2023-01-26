import torch
import torch.nn as nn
import copy

from src.net.st_gcn import Model as STGCN
from src.net.utils.tools import Distances, PositiveOnlyLoss, EMA, MLP


class HYSP(nn.Module):
    def __init__(
            self, pretrain=True, encoder='stgcn', in_channels=3, hidden_channels=64, out_channels=1024,
            projection_hidden_size=[1024, 1024], predictor_hidden_size=[1024, 1024],
            num_classes=60, moving_average_decay=0.999,
            dropout=0.5, graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
            edge_importance_weighting=True, hyper=True, hyper_c=1., **kwargs):
        super().__init__()

        self.pretrain = pretrain
        self.hyper = hyper
        print("Hyperbolic setting:", hyper)
        self.loss = PositiveOnlyLoss(hyper, hyper_c)

        if encoder == 'stgcn':
            self.online_encoder = STGCN(in_channels=in_channels, hidden_channels=hidden_channels,
                                        out_channels=out_channels, dropout=dropout, graph_args=graph_args,
                                        edge_importance_weighting=edge_importance_weighting)
        else:
            raise NotImplementedError(f'encoder {encoder} not implemented')

        if self.pretrain:
            # ONLINE branch
            self.online_projector = MLP(out_channels, projection_hidden_size)
            self.online_predictor = MLP(projection_hidden_size[-1], predictor_hidden_size)

            # TARGET branch
            self.target_encoder = self._no_grad_copy(self.online_encoder)
            self.target_projector = self._no_grad_copy(self.online_projector)
            self.ema_updater = EMA(moving_average_decay)

        else:
            self.online_projector = MLP(out_channels, [num_classes], self.hyper)
            self.distances = Distances()

    @torch.no_grad()
    def _no_grad_copy(self, net):
        new_net = copy.deepcopy(net)
        for p in new_net.parameters():
            p.requires_grad = False
        return new_net

    @torch.no_grad()
    def update_moving_average(self):
        self.ema_updater.update_moving_average(self.target_encoder, self.online_encoder)
        self.ema_updater.update_moving_average(self.target_projector, self.online_projector)

    def forward(self, x, y=None, lambda_hyper=0.):
        assert not (
            self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if not self.pretrain:
            if not self.hyper:
                emb = self.online_encoder(x)
                return self.online_projector(emb)
            else:
                emb = self.online_encoder(x)
                emb = self.distances.project(emb)
                return self.online_projector(emb)

        # compute query embeddings
        q_one = self.online_encoder(x)
        q_two = self.online_encoder(y)

        q_one = self.online_predictor(self.online_projector(q_one))
        q_two = self.online_predictor(self.online_projector(q_two))

        # compute target embeddings
        with torch.no_grad():
            k_one = self.target_projector(self.target_encoder(x))
            k_two = self.target_projector(self.target_encoder(y))

        if not self.hyper:
            # compute loss for euclidean pretraining
            loss_one = self.loss.compute(q_one, k_two.detach())
            loss_two = self.loss.compute(q_two, k_one.detach())
            loss = (loss_one + loss_two)/2
            return loss, q_one.detach(), k_two.detach()

        else:
            # Compute loss for hyperbolic pretraining
            loss_euc_one, loss_hyp_one = self.loss.compute(q_one, k_two, lambda_hyper)
            loss_euc_two, loss_hyp_two = self.loss.compute(q_two, k_one, lambda_hyper)

            loss_euc = (loss_euc_one + loss_euc_two)/2
            loss_hyp = (loss_hyp_one + loss_hyp_two)/2

            loss = (1 - lambda_hyper) * loss_euc + (lambda_hyper) * loss_hyp
            return loss, q_one, k_two, loss_euc, loss_hyp
