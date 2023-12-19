# High Dynamic Range Image Reconstruction via Deep Explicit Polynomial Curve Estimation (EPCE)


This repository is adapted from [the code](https://github.com/jqtangust/epce-hdr) linked to the paper [High Dynamic Range Image Reconstruction via Deep Explicit Polynomial Curve Estimation](https://arxiv.org/abs/2307.16426) authored by  Jiaqi Tang, Xiaogang Xu, Sixing Hu and Ying-Cong Chen presented at ECAI 2023. <br>

The authors of the current repository are:

- Claire Alexandra Friedrich 
- Céline Kalbermatten
- Adam Zinebi

The repository was created within the scope of a Machine Learning project during the course [CSS-433 Machine Learning](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) at [EPFL](https://www.epfl.ch/en/).


## Table of contents:

- [Introduction](#introduction)
- [Setup](#setup)
- [Files](#files)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgement](#acknowledgement)

## Introduction
Due to different physical imaging parameters, the tone-mapping functions between images and real radiance are highly diverse, which makes HDR reconstruction extremely challenging. Existing solutions can not explicitly clarify a corresponding relationship between the tone-mapping function and the generated HDR image, but this relationship is vital when guiding the reconstruction of HDR images. To address this problem, we propose a method to explicitly estimate the tone mapping function and its corresponding HDR image in one network.

## Setup

### Pre-requisites

- Python3
- [PyTorch](https://pytorch.org/)
- GPU, CUDA, cuDNN
- [OpenCV](https://opencv.org)
- [PIL](https://pypi.org/project/Pillow/)
- [Numpy](https://numpy.org/)
- [argparse](https://pypi.org/project/argparse/)
- [scikit-image](https://scikit-image.org/)
- [tqdm](https://pypi.org/project/tqdm/)
- [yalm](https://pypi.org/project/PyYAML/)
- [tensorboard](https://pypi.org/project/tensorboard/)
- [lmdb](https://pypi.org/project/lmdb/)
- [scipy](https://scipy.org/install/)
- [timm](https://pypi.org/project/timm/)
- [einops](https://pypi.org/project/einops/)

**`requirements.txt`** is provided to install the necessary Python dependencies

```sh
pip install -r requirements.txt
```

## Files

- `split_data.py`
- `reduce_tiled_dataset.py`
- `dataloader.py`
- `EPCE_model.py`
- `train.py`
- `test.py`
- `options.py`
- `vgg.py`
- `util.py`

### Description

The whole implementation of the project has been done in Python.

The file `split_data.py` creates a dataset in the structure needed for the training and testing of the model. More information can be found in the part about the [dataset](#dataset).

The file `reduce_tiled_dataset.py` reduces the created dataset to a certain percentage of it. More information can be found in the part about the [dataset](#dataset).

The file `dataloader.py` defines a custom HDR class that loads LDR and HDR images. It provides methods to transform the images into tensors and organize the into a dictionary. 

>**_Note:_** The path to the dataset has to be set in line 22 and 24. 

The file `EPCE_model.py` implements a neural network architecture for HDR image processing and enhancement. It includes modules for curve estimation using polynomial functions, pixel-wise learning for image refinement and a Pyramid-Path Vision Transformer (PPViT) with transformer blocks, attention mechanisms and up- and downsampling layers, enabling advanced HDR image reconstruction. 

The file `train.py` is designed to train a model for HDR image reconstruction. It initializes the model, optimizes it using defined loss functions, does training and validation loops, saves intermediate results, and ultimately saves the trained model. Additionally, it plots the losses throughout the process.

The file `test.py`evaluates the trained HDR image model. It loads test data, applies the model to generate HDR images, saves the results, and computes evaluation metrics like PSNR and SSIM for the generated images compared to ground truth. The final results are printed.

The file `options.py` contains a class Options that defines and handles various settings and configurations used for training, debugging, and evaluation of the EPCE model. It uses the argparse module to define command-line arguments for different options like batch size, learning rate, number of epochs, GPU IDs, debugging flags, and testing options such as checkpoint paths and logging scores during evaluation. The parse() method parses these options and returns the parsed arguments.

The file `util.py` contains several utility functions including methods for checkpoint loading and saving, HDR image tonemapping, saving HDR and LDR images, updating learning rates and plotting losses. 

The file `vgg.py` implements a VGG19 network for perceptual loss computation during training of HDR image generation models, using pre-trained layers to extract features and compute the loss.


## Dataset

The dataset is expected to contain LDR (input) and HDR (ground truth) image pairs. The network is trained to learn the mapping from LDR images to their corresponding HDR ground truth counterparts.

The dataset should have the following folder structure: 

```
> dataset
    > train
        > HDR
            > hdr_image_1.hdr/exr
            > hdr_image_2.hdr/exr
            .
            .
        > LDR
            > ldr_image_1.jpg/png
            > ldr_image_2.jpg/png
            .
            .
    > test
        > HDR
            > hdr_image_1.hdr/exr
            > hdr_image_2.hdr/exr
            .
            .
        > LDR
            > ldr_image_1.jpg/png
            > ldr_image_2.jpg/png
            .
            .
```

- The train and test datasets of 1700 tiled coloured images can be downloaded [here](https://drive.google.com/drive/folders/1KyE1_YEZJeJ_O8cztDCHOH0f19J_vnnb?usp=sharing).
  The training set consists of the 1700 tiled images that have been created from the 34 full images, which together form the testing set.
- The train and test dataset of 34 full images in black and white can be dowloaded [here](https://drive.google.com/drive/folders/1qgAQajoZujeJ700HGq5aJ6Ym9kww8Zad?usp=sharing)


### Create your own coloured tiled dataset

If you want to generate a dataset from your own coloured tiled images, order your LDR and HDR images according to the following folder structure:

```
> NAME_OF_THE_FOLDER_WITH_THE_DATA (put as data path)

    > raw_ref_video_hdr
        - contains the test HDR and LDR images in .hdr format

    > raw_video_png
        - contains the test LDR images in .png format

    > tile_ref_video_hdr
        - contains the training HDR and LDR images in .hdr format

    > tile_ref_video_png
        - contains the training HDR and LDR images in .png format
```

When your data is structured in the above ways, split your data by using the provided script: 
```sh
python3 split_data.py data_path
```

**Note:** `data_path` is the path (str) to the dataset on your local computer

### Reduce the size of the coloured tiled dataset

Since the EPCE model is very large and training on a lot of coloured tiled images takes a lot of time, the dataset can be reduced to a certain percentage of it. By doing so, the precision of the model will be less high but the training time is decreased significantly. 
In order to reduce the dataset, the file `reduce_tiled_dataset.py` can be executed. A new directory containing the reduced dataset is created and the model can then be trained on this reduced dataset. The reduced dataset consists of tiled images for training as well as for testing. No full images are included. 


## Training


After the dataset has been prepared, the model can be trained using:

```sh
python3 train.py
```
- Training results (LDR input, HDR prediction and HDR ground truth) are stored in the **`train_results`** directory. The number of stored results can be varied by varying the `opt.save_results_after` parameter. If no images should be stored, the corresponding part can be commented in the code.
- Training on the full black and white dataset takes about 10 seconds per epoch. This gives a total of 30 minutes for 200 epochs and 60 minutes for 400 epochs.
- Training on the full tiled colour dataset takes about 45 minutes per epoch. Training on only 25% or 10% of the whole dataset takes 500 or 200 seconds per epoch respectively. Training on 2% of the whole dataset takes about 45 seconds per epoch and therefore a total of about 2.5 hours for 200 epochs. 



### Pretrained models

Some pre-trained models can be downloaded from the following links. It always contains the latest checkpoint of the training process.

- [EPCE model trained on 34 full images in black and white for 200 epochs](https://drive.google.com/file/d/1AyuuPePtOPpfvnFMlIIWgz762Zalojl4/view?usp=sharing)
- [EPCE model trained on 34 full images in black and white for 400 epochs](https://drive.google.com/file/d/1DU3pNbqjESL-X_H2PXLTNSDH7avRaesN/view?usp=sharing)
- [EPCE model trained on 2% of the total of 1700 256x256 tiled coloured images with 200 epochs](https://drive.google.com/file/d/1uZTgso8eQrJTSvu0vqfmGSodXr5uhh1e/view?usp=sharing)


## Evaluation

The performance of the network can be evaluated using: 

```sh
python3 test.py --ckpt_path /path/to/checkpoint
```

- Test results (LDR input, HDR prediction and HDR ground truth) are stored in the **`test_results`** directory. If no images should be stored, the corresponding part can be commented in the code.
- HDR images can be visualised using [OpenHDRViewer](https://viewer.openhdr.org) or by installing [HDR + WCG Image Viewer](https://apps.microsoft.com/detail/9PGN3NWPBWL9?rtc=1&hl=fr-ch&gl=CH) on windows.
- If the checkpoint path is not specified, it defaults to `checkpoints/latest.ckpt` for evaluating the model.

### Test results with the pretrained models

The generated test HDR images for each of the above models can be found at the following links: 

- [Test results of the EPCE model trained on 34 full images in black and white for 200 epochs](https://drive.google.com/drive/folders/1ntQ-qjeVLK6FMeQcmUECL2cXCtQrDqDV?usp=sharing)
- [Test results of EPCE model trained on 34 full images in black and white for 400 epochs](https://drive.google.com/drive/folders/1hamkzUn0yuthCIhBA1jYqAG_QURVxUHZ?usp=sharing)
- [Test results of the EPCE model trained on 2% of the total of 1700 256x256 tiled coloured images with 200 epochs](https://drive.google.com/drive/folders/1YQpXP0tta-8KXKB51ivSpa-P926QEyHh?usp=sharing)

The three slurm output files of the tests containing the SSIM and PSNR loss metrics can be found in the folder `test slurm output files`.


## Acknowledgement

This project on HDR reconstruction was provided by the [Laboratory of Integrated Performance in Design (LIPID)](https://www.epfl.ch/labs/lipid/) at EPFL and supervised by Stephen Wasilewski and Cho Yunjoung. 

The code was adapted from the previously cited [repository](https://github.com/jqtangust/epce-hdr).



