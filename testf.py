import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import EPCE
import glob
from torchvision import transforms
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from skimage.measure import compare_ssim

import os
import time

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

# ======================================
# Initial training options 
# ======================================

# initialize options
opt = potions.Options().parse()

# define the batch size
batch_size = 1

# ======================================
# Load the data
# ======================================

dataset = HDRDataset(mode="test", opt=opt)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# print the number of testing images
print("Testing samples: ", len(dataset))

# ========================================
# Model initialisation, 
# loading & GPU configuration
# ========================================

# define the curve estimation model
model = EPCE.PPVisionTransformer()

# take the saved model
modelstat = 'pathtoyourmodel'
# load the saved model's state dictionary
model_state_dict = torch.load(modelstat)
# load the state dictionary into the model
model.load_state_dict(model_state_dict)

# ========================================
# GPU configuration
# ========================================

# set the device -> CPU or GPU 
device = torch.device("cuda")
# move the model to the device
model.cuda()
#model.to(device)

# ========================================
# Evaluation of the model, 
# computation of the evaluation metrics
# ========================================

# make the necessary directories for saving the test results
make_required_directories(mode="test")

# initialize the evaluation metrics
avg_psnr = 0
avg_ssim = 0
avg_mse = 0

# define the mean squared error loss
mse_loss = nn.MSELoss()

print("Starting evaluation. Results will be saved in '/test_results' directory")

with torch.no_grad():

    for batch, data in enumerate(tqdm(data_loader, desc="Testing %")):

        # get the LDR images
        input = data["ldr_image"]
        input = input.to(device)

        # get the HDR images
        ground_truth = data["hdr_image"]
        ground_truth = ground_truth.to(device)

        # generate the output from the model
        output = model(input)
        print('output:', output.size())
        output = output.squeeze(dim=0)
        print('output after squeeze:', output.size())

        # get the final output from the model
        output = output[-1]

        for batch_ind in range(len(output.data)):

            # save the generated images
            save_ldr_image(img_tensor=input, batch=batch_ind, path="./test_results/ldr_b_{}_{}.png".format(batch, batch_ind),)
            
            save_hdr_image(img_tensor=output, batch=batch_ind, path="./test_results/generated_hdr_b_{}_{}.hdr".format(batch, batch_ind),)
            
            save_hdr_image(img_tensor=ground_truth, batch=batch_ind, path="./test_results/gt_hdr_b_{}_{}.hdr".format(batch, batch_ind),)

            if opt.log_scores:
                # calculate the PSNR score
                mse = mse_loss(output.data[batch_ind], ground_truth.data[batch_ind])
                avg_mse += mse.item()
                psnr = 10 * np.log10(1 / mse.item())

                avg_psnr += psnr

                generated = (np.transpose(output.data[batch_ind].cpu().numpy(), (1, 2, 0)) + 1) / 2.0
                real = (np.transpose(ground_truth.data[batch_ind].cpu().numpy(), (1, 2, 0))+ 1) / 2.0

                # calculate the SSIM score
                ssim = compare_ssim(generated, real, multichannel=True)
                avg_ssim += ssim

# ========================================
# Printing the results
# ========================================

if opt.log_scores:
    print("===> Avg PSNR: {:.4f} dB".format(avg_psnr / len(dataset)))
    print("Avg SSIM -> " + str(avg_ssim / len(dataset)))
    print("Avg MSE -> " + str(avg_mse / len(dataset)))

print("Evaluation completed.")


"""l1 = torch.nn.L1Loss()
perceptual_loss = EPCE.VGGLoss()

# set the model to evaluation mode
model.eval()  
losses = []
ssimes = []
n = 0
with torch.no_grad():

    total_loss = 0.0

    for batch in tqdm(data_loader):
        n += 1

        # get the LDR images
        inputs = batch['ldr_image']
        inputs = inputs.to(device)

        # get the HDR images
        targets = batch['hdr_image']
        targets = targets.to(device)

        # generate the output from the model
        outputs = model(inputs)

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
    print("Average loss: ", average_loss)"""
