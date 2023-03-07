import multiprocessing as mp
import wandb
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.feeder.ntu_feeder import Feeder_single
from tools import load_config
from src.eval_learners import EvalLearner

pl.seed_everything(123)


if __name__ == '__main__':

    cfg = load_config(name='Linear Evaluation')

    if isinstance(cfg.num_workers, int):
        num_workers = cfg.num_workers
    else:
        num_workers = mp.cpu_count()

    # train data loading
    partial_sample = True if cfg.protocol == 'semi' else False

    cfg.work_dir = Path(cfg.work_dir).joinpath(cfg.protocol)
    try:
        print(f"Creating directory {str(cfg.work_dir)}...")
        cfg.work_dir.mkdir(parents=True)
    except FileExistsError:
        print(f"Directory {str(cfg.work_dir)} already exists")

    num_devices = len(cfg.device) if isinstance(cfg.device, list) else int(cfg.device)

    # initialize data feeder
    train_feeder = Feeder_single(cfg.train_feeder_args.data_path,
                                 cfg.train_feeder_args.label_path,
                                 cfg.train_feeder_args.shear_amplitude,
                                 cfg.train_feeder_args.temperal_padding_ratio,
                                 partial_sample=partial_sample)
    test_feeder = Feeder_single(cfg.test_feeder_args.data_path,
                                cfg.test_feeder_args.label_path,
                                cfg.test_feeder_args.shear_amplitude,
                                cfg.test_feeder_args.temperal_padding_ratio)

    # create dataloaders

    train_loader = DataLoader(
        dataset=train_feeder,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,    # set True when memory is abundant
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=True)

    test_loader = DataLoader(
        dataset=test_feeder,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        pin_memory=True,    # set True when memory is abundant
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=False)

    # init wandb logger
    wandb_logger = None
    if cfg.wandb.enable:
        if not hasattr(cfg.wandb, 'id'):
            cfg.wandb.id = wandb.util.generate_id()
        wandb_logger = WandbLogger(project=cfg.wandb.project, group=cfg.wandb.group,
                                   name=cfg.wandb.name, id=cfg.wandb.id, entity=cfg.wandb.entity,
                                   save_dir=cfg.wandb.save_dir)

    # init self-supervised learner
    learner = EvalLearner(cfg)

    if wandb_logger is not None:
        wandb_logger.watch(learner, log_freq=10)

    checkpoint_callback = ModelCheckpoint(dirpath=str(cfg.work_dir), save_top_k=1,
                                          verbose=True, monitor='top1_acc', mode='max',
                                          filename='{epoch}-{top1_acc:.2f}')

    # init trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.device,
        max_epochs=cfg.num_epoch,
        accumulate_grad_batches=num_devices,
        sync_batchnorm=True,
        strategy="ddp_find_unused_parameters_false",
        num_nodes=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=cfg.eval_interval,
        precision=cfg.precision,
        detect_anomaly=True
    )

    # start training
    trainer.fit(learner, train_loader, test_loader)
