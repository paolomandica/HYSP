import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.net.st_gcn import Model as STGCN
from src.net.utils.tools import weights_init, EMA, MLP, HyperCrossEntropyLoss


class SkeletonCLR(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, pretrain=True, queue_size=32768,
                 in_channels=3, hidden_channels=64, out_channels=1024,
                 projection_hidden_size=[1024, 1024],
                 moving_average_decay=0.999,
                 dropout=0.5, graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        self.pretrain = pretrain
        self.criterion = nn.CrossEntropyLoss()

        # ONLINE branch
        self.online_encoder = STGCN(in_channels=in_channels, hidden_channels=hidden_channels,
                                    out_channels=out_channels, dropout=dropout, graph_args=graph_args,
                                    edge_importance_weighting=edge_importance_weighting)
        self.online_projector = MLP(out_channels, projection_hidden_size)
        self.online_encoder.apply(weights_init)
        self.online_projector.apply(weights_init)

        if self.pretrain:
            self.K = queue_size
            self.T = 0.07

            # TARGET branch
            self.target_encoder = self._no_grad_copy(self.online_encoder)
            self.target_projector = self._no_grad_copy(self.online_projector)
            self.ema_updater = EMA(moving_average_decay)

            # create the queue
            self.K = queue_size
            self.register_buffer("queue", torch.randn(projection_hidden_size[-1], queue_size))
            self.queue = F.normalize(self.queue, dim=0)

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

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        self.queue = torch.cat((self.queue[:, batch_size:], keys.T), dim=1)

    def forward(self, image_one, image_two=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """

        if not self.pretrain:
            return self.online_projector(self.online_encoder(image_one))

        # compute query features
        q = self.online_projector(self.online_encoder(image_one))
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self.update_moving_average()

            k = self.target_projector(self.target_encoder(image_two))
            k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return self.criterion(logits, labels)


class HyperSkeletonCLR(nn.Module):
    def __init__(self, pretrain=True, queue_size=32768,
                 in_channels=3, hidden_channels=64, out_channels=1024,
                 projection_hidden_size=[1024, 1024],
                 moving_average_decay=0.999,
                 dropout=0.5, graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, temperature=0.07, ** kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        self.pretrain = pretrain

        # ONLINE branch
        self.online_encoder = STGCN(in_channels=in_channels, hidden_channels=hidden_channels,
                                    out_channels=out_channels, dropout=dropout, graph_args=graph_args,
                                    edge_importance_weighting=edge_importance_weighting)
        self.online_projector = MLP(out_channels, projection_hidden_size)
        # self.online_encoder.apply(weights_init)
        # self.online_projector.apply(weights_init)

        if self.pretrain:
            self.loss = HyperCrossEntropyLoss(hyper_c=1., temperature=temperature)

            # TARGET branch
            self.target_encoder = self._no_grad_copy(self.online_encoder)
            self.target_projector = self._no_grad_copy(self.online_projector)
            self.ema_updater = EMA(moving_average_decay)

            # create the queue
            self.K = queue_size
            self.register_buffer("queue", torch.randn(projection_hidden_size[-1], queue_size))
            # self.queue = F.normalize(self.queue, dim=0)

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

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        self.queue = torch.cat((self.queue[:, batch_size:], keys.T), dim=1)

    def forward(self, image_one, image_two=None, lambda_hyper=0.):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """

        if not self.pretrain:
            return self.online_projector(self.online_encoder(image_one))

        # compute query embeddings
        q_one = self.online_projector(self.online_encoder(image_one))

        # compute target embeddings
        with torch.no_grad():
            k_two = self.target_projector(self.target_encoder(image_two))

        loss_euc, loss_hyp = self.loss.compute(q_one, k_two, self.queue, lambda_hyper)
        loss = (1 - lambda_hyper) * loss_euc + (lambda_hyper) * loss_hyp

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_two)

        return loss, q_one, k_two, loss_euc, loss_hyp
