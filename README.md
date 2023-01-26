# HYSP

This repository provides the Official PyTorch implementation of the paper "Hyperbolic Self-paced Learning for Self-supervised Skeleton-based Action Representations" (ICLR 2023).

Luca Franco, Paolo Mandica, Bharti Munjal, Fabio Galasso  
Sapienza University of Rome


## Requirements

  ![Python >=3.8](https://img.shields.io/badge/Python->=3.8-yellow.svg)    ![PyTorch >=1.10](https://img.shields.io/badge/PyTorch->=1.10-blue.svg)


## Environment Setup

```bash
# Install requirements using pip
pip install -r requirements.txt
```


## Data Preparation

- Download the raw data of [NTU RGB+D](https://github.com/shahroudy/NTURGB-D) and [PKU-MMD](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html).
- For NTU RGB+D dataset, preprocess data with `code/tools/ntu_gendata.py`. For PKU-MMD dataset, preprocess data with `code/tools/pku_part1_gendata.py`.
- Then downsample the data to 50 frames with `code/feeder/preprocess_ntu.py` and `code/feeder/preprocess_pku.py`.
- If you don't want to process the original data, download the file folder [action_dataset](https://drive.google.com/drive/folders/1VnD3CLcD7bT5fMGI3tDGPlcWZmBbXS0m?usp=sharing).


## Self-supervised Pre-Training

Example for self-supervised pre-training. You can change hyperparameters through `.yaml` files in `config/DATASET/pretext` folder.

```bash
# example of pre-training on NTU-60 xview dataset
python main_pretrain.py --config config/ntu60/pretext/pretext_xview.yaml
```

If you are using 2 or more gpus use the following launch script (substitute NUM_GPUS with the number of gpus):
```bash
torchrun --standalone --nproc_per_node=NUM_GPUS main_pretrain.py --config config/ntu60/pretext_xview.yaml
```

## Evaluation

Example for evaluation. You can change hyperparameters through `.yaml` files in `config/DATASET/eval` folder. For example, you can set the `protocol` to `linear`, `semi` or `supervised` depending on the type of evaluation you want to perform.

```bash
# example of pre-training on NTU-60 xview dataset
python main_eval.py --config config/ntu60/eval/eval_xview.yaml
```

## 3-stream Ensemble

Once a model has been pre-trained and evaluated on all 3 single streams (joint, motion, bone), you can compute the 3-stream ensemble performance by running the following script. Remember to substitute the correct paths inside the script.

```bash
python code/ensemble/ensemble_ntu.py
```

## Training Precision

For **linear evaluation** you can set `precision: 16` in the config file, while for **pre-training, semi and supervised evaluation** you should set `precision: 32` for higher stability.

## Acknowledgement

The framework of our code is extended from the following repositories. We sincerely thank the authors for releasing the codes.
- Some parts of our code are based on [AimCLR](https://github.com/levigty/aimclr).
- The encoder is based on [ST-GCN](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md).

## Licence

This project is licensed under the terms of the MIT license.
