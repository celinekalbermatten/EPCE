# High Dynamic Range Image Reconstruction via Deep Explicit Polynomial Curve Estimation (EPCE)


This repository is adapted from [the code](https://github.com/jqtangust/epce-hdr) linked to the paper [High Dynamic Range Image Reconstruction via Deep Explicit Polynomial Curve Estimation](https://arxiv.org/abs/2307.16426) authored by  Jiaqi Tang, Xiaogang Xu, Sixing Hu and Ying-Cong Chen presented at ECAI 2023. <br>

The authors of the current repository are:

- Claire Alexandra Friedrich 
- Céline Kalbermatten
- Adam Zinebi

The repository was created within the scope of a Machine Learning project during the course [CSS-433 Machine Learning](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) at [EPFL](https://www.epfl.ch/en/).


## Table of contents:

- [Introduction](#ibstract)
- [Setup](#setup)
- [Files](#files)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgement](#acknowledgement)

## Introcution
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
- `reduce_dataset.py`
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

The file `reduce_dataset.py` reduces the created dataset to a certain percentage of it. More information can be found in the part about the [dataset](#dataset).

The file `dataloader.py` defines a custom HDR class that loads LDR and HDR images. It provides methods to transform the images into tensors and organize the into a dictionary. MAYBE MENTION THE DECREASE DATA SIZE FUNCTION

The file `EPCE_model.py` implements a neural network architecture for HDR image processing and enhancement. It includes modules for curve estimation using polynomial functions, pixel-wise learning for image refinement and a Pyramid-Path Vision Transformer (PPViT) with transformer blocks, attention mechanisms and up- and downsampling layers, enabling advanced HDR image reconstruction. 

The file `train.py` is designed to train a model for HDR image reconstruction. It initializes the model, optimizes it using defined loss functions, does training and validation loops, saves intermediate results, and ultimately saves the trained model. Additionally, it plots the losses throughout the process.

The file `test.py`evaluates the trained HDR image model. It loads test data, applies the model to generate HDR images, saves the results, and computes evaluation metrics like PSNR and SSIM for the generated images compared to ground truth. The final results are printed.

The file `options.py` contains a class Options that defines and handles various settings and configurations used for training, debugging, and evaluation of the EPCE model. It uses the argparse module to define command-line arguments for different options like batch size, learning rate, number of epochs, GPU IDs, debugging flags, and testing options such as checkpoint paths and logging scores during evaluation. The parse() method parses these options and returns the parsed arguments.

The file `vgg.py` implements a VGG19 network for perceptual loss computation during training of HDR image generation models, using pre-trained layers to extract features and compute the loss.

The file `util.py` contains several utility functions including methods for checkpoint loading and saving, HDR image tonemapping, saving HDR and LDR images, updating learning rates and plotting losses. 


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

- The train and test datasets can be downloaded [here](https://drive.google.com/drive/folders/1KyE1_YEZJeJ_O8cztDCHOH0f19J_vnnb?usp=sharing)
- The pretrained model has been trained on 1700 256x256 LDR/HDR pairs generated by tiling 34 2560x1280 test LDR/HDR frames into 50 smaller frames (check in the report for details). 



### Create your own dataset

If you want to generate a dataset from your own images, order your LDR and HDR images according to the following folder structure:

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


## Training


After the dataset has been prepared, the model can be trained using:

```sh
python3 ACHTUNGACHTUNG.py
```
- Training results (LDR input, HDR prediction and HDR ground truth) are stored in the **`train_results`** directory.

The corresponding parameters/options for training have been specified in the **`options.py`** file and can be easily altered. They can be logged using -

```sh
python3 ACHTUNGACHTUNG.py --help
```
- **`--iter`** param is used to specify the number of feedback iterations for global and local feedback mechanisms (refer to paper/architecture diagram)
- Checkpoints of the model are saved in the **`checkpoints`** directory. (Saved after every 2 epochs by default)
- GPU is used for training. Specify GPU IDs using **`--gpu_ids`** param.
- The model takes around 5 hours to train on a dataset of 1700 images on a Tesla V100-PCIE-32GB GPU.

### Pretrained models

Three pre-trained models can be downloaded from the following links. 

These models have been trained with the default options, on 256x256 size images for 200 epochs. (MAYBE MODIFY)

- [2-Iterations model from paper](https://drive.google.com/open?id=13vTGH-GVIWVL79X8NJra0yiguoO1Ox4V)
- [FHDR model trained on 1700 256x256 images with 200 epochs](https://drive.google.com/file/d/1_Bp6kR56uttLXwW9IWdaiGZwmIoDIqlG/view?usp=drive_link)
- FHDR model trained on clear sky 256x256 images with 200 epoch [UP-COMING]


## Evaluation of the model

The performance of the network can be evaluated using: 

```sh
python3 ACHTUNGTEST.py --ckpt_path /path/to/checkpoint
```

- Test results (LDR input, HDR prediction and HDR ground truth) are stored in the **`test_results`** directory.
- HDR images can be viewed using [OpenHDRViewer](https://viewer.openhdr.org) or by installing [HDR + WCG Image Viewer](https://apps.microsoft.com/detail/9PGN3NWPBWL9?rtc=1&hl=fr-ch&gl=CH) on windows.
- If the checkpoint path is not specified, it defaults to `checkpoints/latest.ckpt` for evaluating the model.
- PSNR and SSIM scores can be logged for quantitative evaluation by:

```sh
python3 ACHTUNGTEST.py --log_scores
```

## Acknowledgement

This project on HDR reconstruction was provided by the [Laboratory of Integrated Performance in Design (LIPID)](https://www.epfl.ch/labs/lipid/) at EPFL and supervised by Stephen Wasilewski and Cho Yunjoung. 

The code was adapted from the previously cited [repository](https://github.com/jqtangust/epce-hdr).



