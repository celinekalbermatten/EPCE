# High Dynamic Range Image Reconstruction via Deep Explicit Polynomial Curve Estimation (EPCE)


This repository is adapted from [the code](https://github.com/jqtangust/epce-hdr) linked to the paper [High Dynamic Range Image Reconstruction via Deep Explicit Polynomial Curve Estimation](https://arxiv.org/abs/2307.16426) authored by  Jiaqi Tang, Xiaogang Xu, Sixing Hu and Ying-Cong Chen presented at ECAI 2023.


## Table of contents:

- [Introduction](#ibstract)
- [Setup](#setup)
- [Dataset](#dataset)
- [Training](#training)
- [Pretrained models](#pretrained-models)
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

## Dataset

The dataset is expected to contain LDR (input) and HDR (ground truth) image pairs. The network is trained to learn the mapping from LDR images to their corresponding HDR ground truth counterparts.

The dataset should have the following folder structure - 

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



