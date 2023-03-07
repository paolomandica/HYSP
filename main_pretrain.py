import multiprocessing as mp
import torch
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.feeder.ntu_feeder import Feeder_double
from tools import load_config, PeriodicCheckpoint
from src.train_learners import TrainLearner, HyperTrainLearner

pl.seed_everything(123)
# torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == '__main__':

    cfg = load_config(name='Training')

    if isinstance(cfg.num_workers, int):
        num_workers = cfg.num_workers
    else:
        num_workers = mp.cpu_count()

    try:
        print(f"Creating directory {cfg.work_dir}...")
        Path(cfg.work_dir).mkdir(parents=True)
    except FileExistsError:
        print(f"Directory {cfg.work_dir} already exists")

    # data loading
    num_devices = len(cfg.device) if isinstance(cfg.device, list) else int(cfg.device)
    cfg.lr = cfg.base_lr

    # initialize data feeder

    train_feeder = Feeder_double(cfg.train_feeder_args.data_path,
                                 cfg.train_feeder_args.label_path,
                                 cfg.train_feeder_args.shear_amplitude,
                                 cfg.train_feeder_args.temperal_padding_ratio)
    val_feeder = Feeder_double(cfg.val_feeder_args.data_path,
                               cfg.val_feeder_args.label_path,
                               cfg.val_feeder_args.shear_amplitude,
                               cfg.val_feeder_args.temperal_padding_ratio)

    # create dataloaders

    train_loader = DataLoader(
        dataset=train_feeder,
        batch_size=cfg.batch_size // num_devices,
        shuffle=True,
        pin_memory=True,    # set True when memory is abundant
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=True)

    val_loader = DataLoader(
        dataset=val_feeder,
        batch_size=cfg.val_batch_size // num_devices,
        shuffle=False,
        pin_memory=True,    # set True when memory is abundant
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=True)

    # init wandb logger
    wandb_logger = None
    if cfg.wandb.enable:
        wandb_logger = WandbLogger(project=cfg.wandb.project, group=cfg.wandb.group,
                                   name=cfg.wandb.name, entity=cfg.wandb.entity,
                                   save_dir=cfg.wandb.save_dir)

    # init self-supervised learner
    if not cfg.hyper:
        learner = TrainLearner(cfg)
    else:
        learner = HyperTrainLearner(cfg)

    if wandb_logger is not None:
        wandb_logger.watch(learner, log_freq=cfg.log_interval)

    checkpoint_callback = PeriodicCheckpoint(dirpath=cfg.work_dir, every=cfg.save_interval)

    # init trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.device,
        max_epochs=cfg.num_epoch,
        log_every_n_steps=cfg.log_interval,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        strategy="ddp_find_unused_parameters_false",
        num_nodes=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=cfg.eval_interval,
        precision=cfg.precision,
        detect_anomaly=True)

    # start training
    ckpt_path = cfg.resume_from if cfg.resume_from != 'None' else None
    trainer.fit(learner, train_loader, val_loader, ckpt_path=ckpt_path)
