# HYSP

This is the official PyTorch implementation of the ICLR 2023 paper [**Hyperbolic Self-paced Learning for Self-supervised Skeleton-based Action Representations**](https://arxiv.org/abs/2303.06242).

*Luca Franco <sup>&dagger; 1</sup>, Paolo Mandica <sup>&dagger; 1</sup>, Bharti Munjal <sup>1,2</sup>, Fabio Galasso<sup>1</sup>*  
*<sup>1</sup> Sapienza University of Rome, <sup>2</sup> Technical University of Munich*  
<sup>&dagger;</sup> Equal contribution

[[`arXiv`](https://arxiv.org/abs/2303.06242)][[`BibTeX`](#Citation)][[`OpenReview`](https://openreview.net/forum?id=3Bh6sRPKS3J)]

--- 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hyperbolic-self-paced-learning-for-self/unsupervised-skeleton-based-action-2)](https://paperswithcode.com/sota/unsupervised-skeleton-based-action-2?p=hyperbolic-self-paced-learning-for-self)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hyperbolic-self-paced-learning-for-self/skeleton-based-action-recognition-on-pku-mmd)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-pku-mmd?p=hyperbolic-self-paced-learning-for-self)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hyperbolic-self-paced-learning-for-self/unsupervised-skeleton-based-action)](https://paperswithcode.com/sota/unsupervised-skeleton-based-action?p=hyperbolic-self-paced-learning-for-self)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hyperbolic-self-paced-learning-for-self/unsupervised-skeleton-based-action-1)](https://paperswithcode.com/sota/unsupervised-skeleton-based-action-1?p=hyperbolic-self-paced-learning-for-self)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hyperbolic-self-paced-learning-for-self/skeleton-based-action-recognition-on-ntu-rgbd-1)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd-1?p=hyperbolic-self-paced-learning-for-self)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hyperbolic-self-paced-learning-for-self/skeleton-based-action-recognition-on-ntu-rgbd)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd?p=hyperbolic-self-paced-learning-for-self)


<p align="center">
<img src=".github/model.png" width=100% height=100% 
class="center">
</p>


## Requirements

  ![Python >=3.8](https://img.shields.io/badge/Python->=3.8-yellow.svg)    ![PyTorch >=1.10](https://img.shields.io/badge/PyTorch->=1.10-blue.svg)


## Environment Setup

1. Create conda environment and activate it
```bash
conda create -n hysp python=3.9
conda activate hysp
```

2. Install requirements using pip inside the conda env
```bash
pip install -r requirements.txt
```


## Data Preparation

- Download the raw data of [NTU RGB+D](https://github.com/shahroudy/NTURGB-D) and [PKU-MMD](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html).
- For NTU RGB+D dataset, preprocess data with `code/tools/ntu_gendata.py`. For PKU-MMD dataset, preprocess data with `code/tools/pku_part1_gendata.py`.
- Then downsample the data to 50 frames with `code/feeder/preprocess_ntu.py` and `code/feeder/preprocess_pku.py`.
- If you don't want to process the original data, download the file folder [action_dataset](https://drive.google.com/drive/folders/1VnD3CLcD7bT5fMGI3tDGPlcWZmBbXS0m?usp=sharing).


## Self-supervised Pre-Training

Example of self-supervised pre-training on NTU-60 xview. You can change the hyperparameters by modifying the `.yaml` files in the `config/DATASET/pretext` folder.

```bash
python main_pretrain.py --config config/ntu60/pretext/pretext_xview.yaml
```

If you are using 2 or more gpus use the following launch script (substitute NUM_GPUS with the number of gpus):
```bash
torchrun --standalone --nproc_per_node=NUM_GPUS main_pretrain.py --config config/ntu60/pretext_xview.yaml
```

## Evaluation

Example of evaluation of a model pre-trained on NTU-60 xview. You can change hyperparameters through `.yaml` files in `config/DATASET/eval` folder. For example, you can set the `protocol` to `linear`, `semi` or `supervised` depending on the type of evaluation you want to perform.

```bash
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

This project is based on the following open-source projects: [AimCLR](https://github.com/levigty/aimclr), [ST-GCN](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md). We sincerely thank the authors for making the source code publicly available.

## Licence

This project is licensed under the terms of the MIT license.

## <a name="Citation"></a>Citation

If you find this repository useful, please consider giving a star :star: and citation:

```latex
@inproceedings{
  franco2023hyperbolic,
  title={Hyperbolic Self-paced Learning for Self-supervised Skeleton-based Action Representations},
  author={Luca Franco and Paolo Mandica and Bharti Munjal and Fabio Galasso},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=3Bh6sRPKS3J}
}
```
