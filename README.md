# OCSampler

This repo is the implementation of [OCSampler: Compressing Videos to One Clip with Single-step Sampling](https://arxiv.org/abs/2201.04388). (CVPR 2022)

# Dependencies

- GPU: TITAN Xp
- GCC: 5.4.0
- Python: 3.6.13
- PyTorch: 1.5.1+cu102
- TorchVision: 0.6.1+cu102
- MMCV: 1.5.3
- MMAction2: 0.12.0

# Installation:

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.6.13 -y
conda activate open-mmlab
```

b. Install PyTorch and TorchVision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

c. Install MMCV.

```shell
pip install mmcv
```

d. Clone the OCSampler repository.

```shell
git clone https://github.com/MCG-NJU/OCSampler
```

e. Install build requirements and then install MMAction2.

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

## Data Preparation:
Please refer to the default [MMAction2 dataset setup](https://github.com/open-mmlab/mmaction2/blob/master/docs/data_preparation.md) to set datasets correctly.

Specially, for ActivityNet dataset, we adopt the training annotation file with one label, 
since there are only 6 out of 10024 videos with more than one labels and these labels are similar.
Owing to the different label mapping between [MMAction2](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/activitynet/label_map.txt) 
and [FrameExit](https://raw.githubusercontent.com/antran89/ActivityNet/master/Crawler/classes.txt) in ActivityNet, we provide two kinds of annotation files.
You can check it in `data/ActivityNet/` and `configs/activitynet_*.py`.

For Mini-Kinetics, please download [Kinetics 400](https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz) 
and use the train/val splits file from [AR-Net](https://github.com/mengyuest/AR-Net#dataset-preparation)

## Pretrained Models:

The pretrained models are provided in [Google Drive](https://drive.google.com/drive/folders/1JQprqRhWH7hvy5HHctzrVSDCK3xKXvMu?usp=sharing)


## Training

Here we take training the OCSampler in ActivityNet dataset for example.

```shell
# bash tools/dist_train.sh {CONFIG_FILE} {GPUS} {--validate}
bash tools/dist_train.sh configs/activitynet_10to6_resnet50.py 8 --validate
```

Note that we directly port the weights of classification models provided from [FrameExit](https://github.com/Qualcomm-AI-research/FrameExit).

## Inference

Here we take evaluating the OCSampler in ActivityNet dataset for example.

```shell
# bash tools/dist_test.sh {CONFIG_FILE} {CHECKPOINT} {GPUS} {--eval mean_average_precision / top_k_accuracy}
bash tools/dist_test.sh configs/activitynet_10to6_resnet50.py modelzoo/anet_10to6_checkpoint.pth 8 --eval mean_average_precision
```

If you want to directly evaluating the OCSampler on other classifier, you can add `again_load` param in config file like [this](configs/activitynet_slowonly_inference_with_ocsampler.py).

```shell
bash tools/dist_test.sh configs/activitynet_slowonly_inference_with_ocsampler.py modelzoo/anet_10to6_checkpoint.pth 8 --eval mean_average_precision
```

# Citation

If you find OCSampler useful in your research, please cite us using the following entry:

```shell
@inproceedings{lin2022ocsampler,
  title={OCSampler: Compressing Videos to One Clip with Single-step Sampling},
  author={Lin, Jintao and Duan, Haodong and Chen, Kai and Lin, Dahua and Wang, Limin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13894--13903},
  year={2022}
}
```

## Acknowledge

In addition to the MMAction2 codebase, this repo contains modified codes from:

- [FrameExit](https://github.com/Qualcomm-AI-research/FrameExit): for implementation of its classifier.
