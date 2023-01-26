import gc
import torch
import pickle
import numpy as np
from tqdm import tqdm
from numpy.lib.format import open_memmap

from src.feeder.ntu_feeder import Feeder_single
from src.net.hysp import HYSP
from src.net.utils.tools import load_weights, Distances

DATA_PATH = "data/ntu60_frame50/xview/train_position.npy"
LABEL_PATH = "data/ntu60_frame50/xview/train_label.pkl"

EPOCH = 150
CHECKPOINT_PATH = f"checkpoints/hysp/ntu60_xview_joint_2/epoch-{EPOCH}.ckpt"
NORMS_OUTPATH = f"checkpoints/hysp/norm_dicts/norms_epoch-{EPOCH}.pkl"
EMBEDS_Z_H_OUTPATH = f"checkpoints/hysp/norm_dicts/embeds_z_h_epoch-{EPOCH}.pt"
EMBEDS_Z_H_HAT_OUTPATH = f"checkpoints/hysp/norm_dicts/embeds_z_h_hat_epoch-{EPOCH}.pt"


def get_dataloader():
    feeder = Feeder_single(DATA_PATH, LABEL_PATH, no_aug=True)
    dataloader = torch.utils.data.DataLoader(feeder, batch_size=1, shuffle=False,
                                             num_workers=4, persistent_workers=True)
    return dataloader


def load_model():
    model = HYSP(pretrain=True, hyper=False).cuda().eval()
    ignore_keys = ['queue']
    load_weights(model, CHECKPOINT_PATH, ignore_keys, 'linear')
    return model


def compute_norms():
    print("Computing radii...")
    dataloader = get_dataloader()
    model = load_model()
    projector = Distances()

    id_to_norms = {}
    embeds_z_h = torch.tensor([]).cuda()
    embeds_z_h_hat = torch.tensor([]).cuda()

    for _, (data, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        z = model.online_projector(model.online_encoder(data.cuda()))
        z_hat = model.target_projector(model.target_encoder(data.cuda()))
        z_h = projector.project(z)
        z_hat_h = projector.project(z_hat)

        embeds_z_h = torch.cat((embeds_z_h, z_h), dim=0)
        embeds_z_h_hat = torch.cat((embeds_z_h_hat, z_hat_h), dim=0)

    torch.save(embeds_z_h, EMBEDS_Z_H_OUTPATH)
    torch.save(embeds_z_h_hat, EMBEDS_Z_H_HAT_OUTPATH)


def load_norms():
    with open(NORMS_OUTPATH, 'rb') as handle:
        id_to_uncertainty = pickle.load(handle)
    return id_to_uncertainty


def main():
    compute_norms()


if __name__ == '__main__':
    main()
