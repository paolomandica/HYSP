from src.net.utils.tools import load_weights, weights_init
import pytorch_lightning as pl
import torch
from src.feeder.tools import process_stream
from geoopt.optim import RiemannianAdam, RiemannianSGD
from flash.core.optimizers import LARS

from src.net.hysp import HYSP
from src.net.simsiam import SimSiam
from src.net.skeletonclr import HyperSkeletonCLR, SkeletonCLR
from src.net.utils.tools import compute_metrics


class BaseLearner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.lr
        self.lr_min = cfg.lr_min

    def configure_optimizers(self):
        if self.cfg.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9,
                                        nesterov=self.cfg.nesterov, weight_decay=float(self.cfg.weight_decay))
        elif self.cfg.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=float(self.cfg.weight_decay))
        elif self.cfg.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.cfg.optimizer == 'RiemannianSGD':
            optimizer = RiemannianSGD(self.parameters(), lr=self.lr, momentum=0.9,
                                      nesterov=self.cfg.nesterov, weight_decay=float(self.cfg.weight_decay))
        elif self.cfg.optimizer == 'RiemannianAdam':
            optimizer = RiemannianAdam(self.parameters(), lr=self.lr,
                                       weight_decay=float(self.cfg.weight_decay), stabilize=10)
        elif self.cfg.optimizer == 'LARS':
            optimizer = LARS(self.parameters(), lr=self.lr, momentum=0.9,
                             weight_decay=float(self.cfg.weight_decay))
        else:
            raise ValueError("Invalid optimizer {}".format(self.cfg.optimizer))

        if self.cfg.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.cfg.step, gamma=0.1)

        elif self.cfg.scheduler == 'cosine':
            schedulers = []
            if not self.cfg.curriculum:
                schedulers.append(torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=self.cfg.warmup_step, verbose=True))
                schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.cfg.num_epoch-self.cfg.warmup_step, verbose=True, eta_min=self.lr_min))
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=schedulers, milestones=[self.cfg.warmup_step], verbose=True)
            else:
                schedulers.append(torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=1, total_iters=self.cfg.final_hyper_epoch))
                schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.cfg.num_epoch-self.cfg.final_hyper_epoch, eta_min=self.lr*0.1, verbose=True))
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=schedulers, milestones=[self.cfg.final_hyper_epoch], verbose=True)

        else:
            raise ValueError("Invalid scheduler {}".format(self.cfg.scheduler))

        return [optimizer], [scheduler]

    def on_before_zero_grad(self, _):
        if self.cfg.model == 'hysp':
            self.model.update_moving_average()

    def log_metrics(self, q, k, prefix='train', on_epoch=True, sync_dist=True):
        euc_norm_x, euc_norm_y, radius_x, radius_y, ang_e, cosine_dist, poincare_dist, euc_dist = compute_metrics(
            q, k, self.cfg.model_args.hyper_c)

        self.log(f'{prefix}/euc_norm_online', euc_norm_x.mean(), on_step=False, on_epoch=on_epoch, sync_dist=True)
        self.log(f'{prefix}/euc_norm_target', euc_norm_y.mean(), on_step=False, on_epoch=on_epoch, sync_dist=True)
        self.log(f'{prefix}/radius_online', radius_x.mean(), on_step=False, on_epoch=on_epoch, sync_dist=True)
        self.log(f'{prefix}/radius_target', radius_y.mean(), on_step=False, on_epoch=on_epoch, sync_dist=True)
        self.log(f'{prefix}/ang_e', ang_e.mean(), on_step=False, on_epoch=on_epoch, sync_dist=True)
        self.log(f'{prefix}/cosine_dist', cosine_dist.mean(), on_step=False, on_epoch=on_epoch, sync_dist=True)
        self.log(f'{prefix}/poincare_dist', poincare_dist.mean(), on_step=False, on_epoch=on_epoch, sync_dist=True)
        self.log(f'{prefix}/euc_dist', euc_dist.mean(), on_step=False, on_epoch=on_epoch, sync_dist=True)


class TrainLearner(BaseLearner):
    def __init__(self, cfg):
        super().__init__(cfg)

        if cfg.model == 'hysp':
            self.model = HYSP(**vars(cfg.model_args), hyper=False)
        elif cfg.model == 'skeletonclr':
            self.model = SkeletonCLR(**vars(cfg.model_args))
        elif cfg.model == 'simsiam':
            self.model = SimSiam(**vars(cfg.model_args), hyper=False)
        else:
            raise ValueError("Choose a valid learner")

        # self.model.apply(weights_init)
        self.save_hyperparameters()

    def forward(self, batch):
        [data1, data2], _ = batch
        data1 = process_stream(data1.float(), self.cfg.stream)
        data2 = process_stream(data2.float(), self.cfg.stream)
        output = self.model(data1, data2)
        return output

    def training_step(self, batch, _):
        loss, q, k = self.forward(batch)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/loss', loss.mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log_metrics(q, k, 'train')

        return {'loss': loss.mean()}

    def validation_step(self, batch, _):
        loss, q, k = self.forward(batch)

        self.log('val/loss', loss.mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log_metrics(q, k, 'val')

        return {'loss': loss.mean()}


class HyperTrainLearner(BaseLearner):
    def __init__(self, cfg):
        super().__init__(cfg)

        if cfg.model == 'hysp':
            self.model = HYSP(**vars(cfg.model_args), hyper=True)
        elif cfg.model == 'skeletonclr':
            self.model = HyperSkeletonCLR(**vars(cfg.model_args))
        elif cfg.model == 'simsiam':
            self.model = SimSiam(**vars(cfg.model_args), hyper=True)
        else:
            raise ValueError("Choose a valid learner")

        # self.model.apply(weights_init)

        if cfg.weights != 'None':
            load_weights(self.model, cfg.weights)

        self.best_val_loss = 1e10
        self.lambda_hyper = 1.

        self.save_hyperparameters()

    def forward(self, batch):
        [data1, data2], _ = batch
        data1 = process_stream(data1.float(), self.cfg.stream)
        data2 = process_stream(data2.float(), self.cfg.stream)

        if self.cfg.curriculum:
            self.lambda_hyper = (self.current_epoch - (self.cfg.initial_hyper_epoch-1)) / \
                ((self.cfg.final_hyper_epoch-1) - (self.cfg.initial_hyper_epoch-1))
            self.lambda_hyper = torch.max(torch.tensor([0, self.lambda_hyper])).item()
            self.lambda_hyper = torch.min(torch.tensor([1, self.lambda_hyper])).item()
        else:
            self.lambda_hyper = 1.

        loss, q, k, loss_euc, loss_hyp = self.model(data1, data2, self.lambda_hyper)
        return loss, q, k, loss_euc, loss_hyp

    def training_step(self, batch, _):

        loss, q, k, loss_euc, loss_hyp = self.forward(batch)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_step=False, on_epoch=True, sync_dist=True)

        self.log('train/loss', loss.mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/loss_euc', loss_euc.mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/loss_hyp', loss_hyp.mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('lambda_hyper', self.lambda_hyper, on_step=False, on_epoch=True, sync_dist=True)

        self.log_metrics(q, k, 'train')

        return {'loss': loss.mean()}

    def validation_step(self, batch, _):

        [data1, data2], label = batch
        data1 = process_stream(data1.float(), self.cfg.stream)
        data2 = process_stream(data2.float(), self.cfg.stream)
        label = label.long()

        loss, q, k, loss_euc, loss_hyp = self.model(data1, data2, self.lambda_hyper)

        self.log('val/loss', loss.mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/loss_euc', loss_euc.mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/loss_hyp', loss_hyp.mean(), on_step=False, on_epoch=True, sync_dist=True)

        self.log_metrics(q, k, 'val')

        return {'loss': loss.mean()}
