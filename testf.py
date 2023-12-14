import torch
from torch import nn
from torch.utils.data import DataLoader
import EPCE
import glob
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_loader import HDRDataset
from model import FHDR
import potions
from util import (
    load_checkpoint,
    make_required_directories,
    mu_tonemap,
    save_checkpoint,
    save_hdr_image,
    save_ldr_image,
    update_lr,

)
from sklearn.model_selection import train_test_split


from skimage.metrics import structural_similarity
import cv2

opt = potions.Options().parse()

dataset = HDRDataset(mode="test", opt=opt)

# split dataset into training and validation sets

# create separate data loaders for training and validation
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
print("samples: ", len(data_loader))


modelstat = 'pathtoyourmodel'

# Define the curve estimation model
model = EPCE.PPVisionTransformer()

model_state_dict = torch.load(modelstat)

# Load the state dictionary into your model
model.load_state_dict(model_state_dict)

# Set the model to evaluation mode if you are using it for inference

l1 = torch.nn.L1Loss()
perceptual_loss = EPCE.VGGLoss()
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# Set the device (CPU or GPU)
device = torch.device("cuda")
model.cuda()


model.eval()  # Set the model to evaluation mode
losses = []
ssimes = []
n = 0
with torch.no_grad():
    total_loss = 0.0
    for batch in tqdm(data_loader):
        n += 1


        inputs = batch['ldr_image']
        inputs = inputs.to(device)
        targets = batch['hdr_image']
        targets = targets.to(device)
        outputs = model(inputs)
        # Forward pass

        # Calculate loss
        l1_loss = 0
        vgg_loss = 0

        # tonemapping ground truth ->
        # computing loss for n generated outputs (from n-iterations) ->

        for i in range(len(outputs)):
            l1_loss += l1(outputs[i], targets[i])
            vgg_loss += perceptual_loss(outputs[i], targets[i])
        # averaged over n iterations
        l1_loss /= len(outputs)
        vgg_loss /= len(outputs)
       # averaged over batches
        l1_loss = torch.mean(l1_loss)
        vgg_loss = torch.mean(vgg_loss)

        # FHDR loss function
        loss = l1_loss + (vgg_loss * 10)
        total_loss += loss.item()
        losses.append(loss.item())

        save_hdr_image(
                img_tensor=outputs,
                batch=0,
                path="./test_results/generated_hdr_b_{}_{}.hdr".format(
                    i
                ),
        )
        save_hdr_image(
                img_tensor=targets,
                batch=0,
                path="./test_results/gt_hdr_b_{}_{}.hdr".format(
                    i
                ),
            )
        # Load images
        image1 = cv2.imread("./test_results/generated_hdr_b_{}_{}.hdr".format(
                    i
                ))
        image2 = cv2.imread("./test_results/gt_hdr_b_{}_{}.hdr".format(
                    i
                ))

        # Convert to grayscale
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Compute SSIM between the two images
        (score, diff) = structural_similarity(image1_gray, image2_gray, full=True)

        # The diff image conctains the actual image differences between the two images
        # and is represented as a floating point data type in the range [0,1]
        # so we must convert the array to 8-bit unsigned integers in the range
        # [0,255] image1 we can use it with OpenCV
        diff = (diff * 255).astype("uint8")
        print("Image Similarity: {:.4f}%".format(score * 100))
        ssimes.append(score)
    average_loss = total_loss / len(data_loader)
    print("Average loss: ", average_loss)
