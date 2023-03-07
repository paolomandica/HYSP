import gc
import torch
import pickle
import numpy as np
from tqdm import tqdm
from numpy.lib.format import open_memmap

from src.feeder.ntu_feeder import Feeder_single
from src.net.hysp import HYSP
from src.net.utils.tools import load_weights, HyperMapper

DATA_PATH = "data/ntu60_frame50/xview/train_position.npy"
LABEL_PATH = "data/ntu60_frame50/xview/train_label.pkl"
CHECKPOINT_PATH = "checkpoints/hysp/ntu60_xview_joint/epoch-300.ckpt"
OUTPUT_PATH = "checkpoints/hysp/ntu60_xview_joint/linear/id_to_uncertainty_2.pkl"


def get_dataloader():
    feeder = Feeder_single(DATA_PATH, LABEL_PATH, no_aug=True)
    dataloader = torch.utils.data.DataLoader(feeder, batch_size=1, shuffle=False,
                                             num_workers=4, persistent_workers=True)
    return dataloader


def load_model():
    hparams = torch.load(CHECKPOINT_PATH)['hyper_parameters']['cfg']
    model = HYSP(pretrain=False, hyper=False).cuda().eval()
    ignore_keys = ['online_projector', 'online_predictor', 'target_encoder', 'target_projector', 'queue']
    load_weights(model, CHECKPOINT_PATH, ignore_keys, 'linear')
    return model


def compute_radii():
    print("Computing radii...")
    dataloader = get_dataloader()
    model = load_model()
    mapper = HyperMapper()

    id_to_uncertainty = {}

    for i, (data, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        emb = model.online_encoder(data.cuda())
        emb_h = mapper.expmap(emb)
        uncertainty = 1 - torch.norm(emb_h, dim=-1)
        id_to_uncertainty[i] = uncertainty.item()

    with open(OUTPUT_PATH, 'wb') as handle:
        pickle.dump(id_to_uncertainty, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_radii():
    with open(OUTPUT_PATH, 'rb') as handle:
        id_to_uncertainty = pickle.load(handle)
    return id_to_uncertainty


def get_mixed_radii(id_to_uncertainty):
    sorted_ids = sorted(id_to_uncertainty, key=id_to_uncertainty.get)
    sorted_ids = sorted_ids[::2]
    new_ids = {i: True for i in sorted_ids}
    for key in id_to_uncertainty.keys():
        if key not in new_ids:
            new_ids[key] = False
    return new_ids


def analyze_radii():
    id_to_uncertainty = load_radii()
    print("Mean uncertainty: ", np.mean(list(id_to_uncertainty.values())))
    print("Median uncertainty: ", np.median(list(id_to_uncertainty.values())))
    print("Max uncertainty: ", np.max(list(id_to_uncertainty.values())))
    print("Min uncertainty: ", np.min(list(id_to_uncertainty.values())))

    # how many under and over median using numpy
    print("\nNumber of samples under median: ", np.sum(
        np.array(list(id_to_uncertainty.values())) < np.median(list(id_to_uncertainty.values()))))
    print("Number of samples over median: ", np.sum(
        np.array(list(id_to_uncertainty.values())) >= np.median(list(id_to_uncertainty.values()))))
    print()


def process_subdataset(subset="low"):
    print("Processing {} uncertainty subdataset...".format(subset))
    dataloader = get_dataloader()
    id_to_uncertainty = load_radii()
    median = np.median(list(id_to_uncertainty.values()))

    dataset = open_memmap(
        'data/ntu60_frame50/xview/train_position_{}_uncertainty.npy'.format(subset),
        dtype='float32',
        mode='w+',
        shape=(len(dataloader)//2, 3, 50, 25, 2))
    labels = np.array([])

    if subset == "low":
        idx = 0
        for i, (data, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            if id_to_uncertainty[i] < median:
                dataset[idx] = data
                labels = np.append(labels, label)
                idx += 1

    elif subset == "high":
        idx = 0
        for i, (data, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            if id_to_uncertainty[i] >= median:
                dataset[idx] = data
                labels = np.append(labels, label)
                idx += 1

    elif subset == "mixed":
        new_ids = get_mixed_radii(id_to_uncertainty)
        idx = 0
        for i, (data, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            if new_ids[i]:
                dataset[idx] = data
                labels = np.append(labels, label)
                idx += 1

    else:
        raise ValueError("Invalid subset name.")

    print("Saving {} uncertainty subdataset...".format(subset))
    # np.save("data/ntu60_frame50/xview/train_position_{}_uncertainty.npy".format(subset), data)
    with open("data/ntu60_frame50/xview/train_label_{}_uncertainty.pkl".format(subset), 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_subdatasets():
    print("Generating subdatasets...")
    process_subdataset("low")
    gc.collect()
    process_subdataset("high")
    gc.collect()
    # process_subdataset("mixed")
    # gc.collect()


def main():
    # compute_radii()
    analyze_radii()
    generate_subdatasets()


if __name__ == '__main__':
    main()
