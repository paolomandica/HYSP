from . import utils

import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path


def load_config(name):
    parser = argparse.ArgumentParser(description=name)
    parser.add_argument('-c', '--config', type=str, required=True,
                        default='./config/ntu60/pretext/prova.yaml')
    arg = parser.parse_args()

    # load YAML config file as dict
    with open(arg.config, 'r') as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)

    # load new parser with default arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.set_defaults(**default_arg)
    try:
        cfg = parser.parse_args('')
    except Exception as e:
        print(e)

    # build sub-parsers
    for k, value in cfg._get_kwargs():
        if isinstance(value, dict):
            new_parser = argparse.ArgumentParser(add_help=False)
            new_parser.set_defaults(**value)
            cfg.__setattr__(k, new_parser.parse_args(''))

    return cfg


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath: str,  every: int):
        super().__init__()
        self.dirpath = dirpath
        self.every = every

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if (pl_module.current_epoch + 1) % self.every == 0:
            assert self.dirpath is not None
            current = Path(self.dirpath) / f"epoch-{pl_module.current_epoch + 1}.ckpt"
            trainer.save_checkpoint(current)
