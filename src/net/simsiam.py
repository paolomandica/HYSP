import torch.nn as nn

from src.net.st_gcn import Model as STGCN
from src.net.utils.tools import Distances, PositiveOnlyLoss, MLP


class SimSiam(nn.Module):
    def __init__(
            self, pretrain=True, in_channels=3, hidden_channels=64, out_channels=1024,
            projection_hidden_size=[1024, 1024], predictor_hidden_size=[1024, 1024],
            num_classes=60, dropout=0.5, graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
            edge_importance_weighting=True, hyper=True, hyper_c=1., **kwargs):
        super().__init__()

        self.pretrain = pretrain
        self.hyper = hyper
        print("Hyperbolic setting:", hyper)
        self.loss = PositiveOnlyLoss(hyper, hyper_c)

        self.online_encoder = STGCN(in_channels=in_channels, hidden_channels=hidden_channels,
                                    out_channels=out_channels, dropout=dropout, graph_args=graph_args,
                                    edge_importance_weighting=edge_importance_weighting)

        if self.pretrain:
            # ONLINE branch
            self.online_projector = MLP(out_channels, projection_hidden_size, bias=True)        # bias=False
            self.online_predictor = MLP(projection_hidden_size[-1], predictor_hidden_size)

        else:
            self.online_projector = MLP(out_channels, [num_classes], self.hyper)
            self.distances = Distances()

    def forward(self, x, y=None, lambda_hyper=0.):
        assert not (
            self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if not self.pretrain:
            if not self.hyper:
                return self.online_projector(self.online_encoder(x))
            else:
                emb = self.online_encoder(x)
                emb = self.distances.project(emb)
                return self.online_projector(emb)

        # compute query embeddings
        z1 = self.online_projector(self.online_encoder(x))
        z2 = self.online_projector(self.online_encoder(y))

        p1 = self.online_predictor(z1)
        p2 = self.online_predictor(z2)

        if not self.hyper:
            # compute loss for euclidean pretraining
            loss_one = self.loss.compute(p1, z2.detach())
            loss_two = self.loss.compute(p2, z1.detach())
            loss = (loss_one + loss_two)/2
            return loss, p1.detach(), z2.detach()

        else:
            # Compute loss for hyperbolic pretraining
            loss_euc_one, loss_hyp_one = self.loss.compute(p1, z2.detach(), lambda_hyper)
            loss_euc_two, loss_hyp_two = self.loss.compute(p2, z1.detach(), lambda_hyper)

            loss_euc = (loss_euc_one + loss_euc_two)/2
            loss_hyp = (loss_hyp_one + loss_hyp_two)/2

            loss = (1 - lambda_hyper) * loss_euc + (lambda_hyper) * loss_hyp
            return loss, p1.detach(), z2.detach(), loss_euc, loss_hyp
