import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.feeder.tools import process_stream
from geoopt.optim import RiemannianSGD
from torchmetrics import Accuracy

from src.net.hysp import HYSP
from src.net.skeletonclr import SkeletonCLR
from src.net.utils.tools import weights_init, load_weights
from src.net.hysp import HYSP
from src.net.simsiam import SimSiam
from src.net.skeletonclr import SkeletonCLR, HyperSkeletonCLR
from src.net.utils.tools import load_weights


class EvalLearner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.model == 'hysp':
            self.model = HYSP(**vars(cfg.model_args), hyper=cfg.hyper)
        elif cfg.model == 'skeletonclr':
            if not cfg.hyper:
                self.model = SkeletonCLR(**vars(cfg.model_args))
            else:
                self.model = HyperSkeletonCLR(**vars(cfg.model_args))
        elif cfg.model == 'simsiam':
            self.model = SimSiam(**vars(cfg.model_args), hyper=cfg.hyper)
        else:
            raise NotImplementedError

        load_weights(self.model, cfg.weights, cfg.ignore_weights, cfg.protocol)

        self.loss = nn.CrossEntropyLoss()
        self.top1_accuracy = Accuracy(task='multiclass', num_classes=self.cfg.model_args.num_classes, top_k=1)
        self.top5_accuracy = Accuracy(task='multiclass', num_classes=self.cfg.model_args.num_classes, top_k=5)
        self.best_top1_acc = 0.

        self.save_hyperparameters()

    def forward(self, batch):
        data, label = batch[0], batch[1].long()
        data = process_stream(data, self.cfg.stream)
        output = self.model(data)
        return output, label

    def training_step(self, batch, _):
        output, label = self.forward(batch)
        loss = self.loss(output, label)
        self.log('loss', loss, on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, _):
        output, label = self.forward(batch)
        loss = self.loss(output, label)
        self.log('val_loss', loss)
        return {'loss': loss, 'pred': output, 'target': label}

    def validation_epoch_end(self, outputs):
        top1_acc = 0.
        top5_acc = 0.
        loss_epoch = 0.

        for output in outputs:
            loss = output['loss']
            acc1 = self.top1_accuracy(output['pred'], output['target'])
            acc5 = self.top5_accuracy(output['pred'], output['target'])
            loss_epoch += loss
            top1_acc += acc1
            top5_acc += acc5

        loss_epoch = round(loss_epoch.item() / len(outputs), 4)
        top1_acc = round((top1_acc.item() / len(outputs)) * 100, 2)
        top5_acc = round((top5_acc.item() / len(outputs)) * 100, 2)

        if top1_acc > self.best_top1_acc:
            self.best_top1_acc = top1_acc

        log_dict = {
            'loss_epoch': loss_epoch,
            'top1_acc': top1_acc,
            'top5_acc': top5_acc,
            'best_top1_acc': self.best_top1_acc
        }

        self.log_dict(log_dict, sync_dist=True)
        self.print("\n----- Results epoch {} -----".format(self.current_epoch))
        for k, v in log_dict.items():
            self.print(k, '\t=\t', v)
        self.print()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD([{'params': self.model.online_encoder.parameters(), 'lr': self.cfg.encoder_lr},
                                     {'params': self.model.online_projector.parameters()}],
                                    lr=self.cfg.base_lr, momentum=0.9,
                                    nesterov=self.cfg.nesterov, weight_decay=float(self.cfg.weight_decay))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg.step, gamma=0.1, verbose=False)
        return [optimizer], [lr_scheduler]


class HyperEvalLearner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.model == "hysp":
            self.model = HYSP(**vars(cfg.model_args), hyper=True)
        else:
            raise NotImplementedError("Only hysp is supported for hyperbolic evaluation")

        load_weights(self.model, cfg.weights, cfg.ignore_weights, cfg.protocol)

        self.loss = nn.CrossEntropyLoss()
        self.top1_accuracy = Accuracy(task='multiclass', num_classes=self.cfg.model_args.num_classes, top_k=1)
        self.top5_accuracy = Accuracy(task='multiclass', num_classes=self.cfg.model_args.num_classes, top_k=5)
        self.best_top1_acc = 0.

        self.save_hyperparameters()

    def forward(self, batch):
        data, label = batch[0], batch[1].long()
        data = process_stream(data, self.cfg.stream)
        output = self.model(data)
        return output, label

    def training_step(self, batch, _):
        output, label = self.forward(batch)
        loss = self.loss(output, label)
        self.log('loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, _):
        output, label = self.forward(batch)
        loss = self.loss(output, label)
        loss = self.loss(output, label)
        self.log('val_loss', loss, sync_dist=True)
        return {'loss': loss, 'pred': output, 'target': label}

    def validation_epoch_end(self, outputs):
        top1_acc = 0.
        top5_acc = 0.
        loss_epoch = 0.

        for output in outputs:
            loss = output['loss']
            acc1 = self.top1_accuracy(output['pred'], output['target'])
            acc5 = self.top5_accuracy(output['pred'], output['target'])
            loss_epoch += loss
            top1_acc += acc1
            top5_acc += acc5

        loss_epoch = round(loss_epoch.item() / len(outputs), 4)
        top1_acc = round((top1_acc.item() / len(outputs)) * 100, 2)
        top5_acc = round((top5_acc.item() / len(outputs)) * 100, 2)

        if top1_acc > self.best_top1_acc:
            self.best_top1_acc = top1_acc
            label_list = []
            score_list = []
            for output in outputs:
                score_list.append(output['pred'])
                label_list.append(output['target'])
            np.save(str(self.cfg.work_dir.joinpath("score_list.npy")),    # (f"score_list_top1_acc={str(top1_acc)}.npy")),
                    torch.cat(score_list).cpu().numpy())
            np.save(str(self.cfg.work_dir.joinpath("label_list.npy")),        # (f"label_list_top1_acc={str(top1_acc)}.npy")),
                    torch.cat(label_list).cpu().numpy())

        log_dict = {
            'loss_epoch': loss_epoch,
            'top1_acc': top1_acc,
            'top5_acc': top5_acc,
            'best_top1_acc': self.best_top1_acc
        }

        self.log_dict(log_dict, sync_dist=True)
        self.print("\n----- Results epoch {} -----".format(self.current_epoch))
        for k, v in log_dict.items():
            self.print(k, '\t=\t', v)
        self.print()

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=LR)
        if self.cfg.optimizer == 'RiemannianSGD':
            optimizer = RiemannianSGD([{'params': self.model.encoder.parameters(), 'lr': self.cfg.encoder_lr},
                                       {'params': self.model.projector.parameters()}],
                                      lr=self.cfg.base_lr, momentum=0.9,
                                      nesterov=self.cfg.nesterov, weight_decay=float(self.cfg.weight_decay))
        elif self.cfg.optimizer == 'SGD':
            optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': self.cfg.encoder_lr},
                                         {'params': self.model.projector.parameters()}],
                                        lr=self.cfg.base_lr, momentum=0.9,
                                        nesterov=self.cfg.nesterov, weight_decay=float(self.cfg.weight_decay))
        else:
            raise ValueError("Choose a valid optimizer")

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg.step, gamma=0.1, verbose=False)
        return [optimizer], [lr_scheduler]
