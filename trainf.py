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
import random



pathtocheckp = 'foldercheckp'
opt = potions.Options().parse()

dataset = HDRDataset(mode="train", opt=opt)

new_data = []  # Liste pour les nouveaux dictionnaires


for m in dataset:
    ldr_image = m['ldr_image']
    hdr_image = m['hdr_image']

    _, height, width = ldr_image.shape

    # Diviser la hauteur et la largeur en 4 segments égaux
    segment_height = height // 4
    segment_width = width // 4

    # Création des sous-images (16 au total) et ajout dans la nouvelle liste
    for i in range(4):
        for j in range(4):
            h_start = i * segment_height
            w_start = j * segment_width

            new_dict = {
                'ldr_image': ldr_image[ :, h_start:h_start + segment_height, w_start:w_start + segment_width],
                'hdr_image': hdr_image[ :, h_start:h_start + segment_height, w_start:w_start + segment_width]
            }
            new_data.append(new_dict)

random.shuffle(new_data)

# split dataset into training and validation sets
train_dataset, val_dataset = train_test_split(new_data, test_size=0.2, random_state=42)
opt.batch_size = 8
# create separate data loaders for training and validation
train_data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
print("Training samples: ", len(train_data_loader))
print("Validation samples: ", len(val_data_loader))



# Define the curve estimation model
model = EPCE.PPVisionTransformer()
# Define the loss function

l1 = torch.nn.L1Loss()
perceptual_loss = EPCE.VGGLoss()
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# Set the device (CPU or GPU)
device = torch.device("cuda")

# Move the model to the device
model.to(device)
# Set the model to training mode
model.train()
# Iterate over the dataset for multiple epochs
losses = []
losses_validation = []

opt.epochs = 2000

for epoch in range(2000):
    total_loss = 0.0
    if epoch > 100:
      update_lr(optimizer, epoch, opt)
    for batch in tqdm(train_data_loader):
        # Move the batch to the device
        optimizer.zero_grad()
        input = batch['ldr_image']
        input = input.to(device)
        output_true = batch['hdr_image']
        output_true = output_true.to(device)
        output = model(input)

        l1_loss = 0
        vgg_loss = 0

        # tonemapping ground truth ->
        # computing loss for n generated outputs (from n-iterations) ->

        for i in range(len(output)):
            l1_loss += l1(output[i], output_true[i])
            vgg_loss += perceptual_loss(output[i], output_true[i])
        # averaged over n iterations
        l1_loss /= len(output)
        vgg_loss /= len(output)
       # averaged over batches
        l1_loss = torch.mean(l1_loss)
        vgg_loss = torch.mean(vgg_loss)

        # FHDR loss function
        loss = l1_loss + (vgg_loss * 10)
        # backpropagate and step
        loss.backward()
        optimizer.step()
        # Accumulate the loss
        total_loss += loss.item()
    # Print the average loss for the epoch
    average_loss = total_loss / len(train_data_loader)
    losses.append(average_loss)

    # VALIDATION LOOP
    # set model to evaluation mode
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_batch, val_data in enumerate(val_data_loader):
            input_val = batch['ldr_image']
            input_val = input_val.to(device)
            ground_truth_val = batch['hdr_image']
            ground_truth_val = ground_truth_val.to(device)
            optimizer.zero_grad()

            output_val = model(input_val)

            # calculate validation loss
            l1_loss_val = 0
            vgg_loss_val = 0

            for i in range(len(output)):
              l1_loss_val += l1(output_val[i],ground_truth_val[i])
              vgg_loss_val += perceptual_loss(output_val[i], ground_truth_val[i])

        # averaged over n iterations
            l1_loss_val /= len(output_val)
            vgg_loss_val /= len(output_val)

        # averaged over batches
            l1_loss_val = torch.mean(l1_loss_val)
            vgg_loss_val = torch.mean(vgg_loss_val)

        # FHDR loss function
            val_loss = l1_loss_val + (vgg_loss_val * 10)
            val_losses.append(val_loss)

    # Calculate average validation loss for the entire validation dataset
    average_val_loss = sum(val_losses) / len(val_losses)
    print(f"Average validation Loss: {average_val_loss}")
    losses_validation.append(average_val_loss)
    print(f"Epoch [{epoch+1}/{opt.epochs}], Average Loss: {average_loss:.4f}")

    model.train()

    if epoch%100 == 0:
      model_save_path = pathtochekp + '/model_odt_{}.pth'.format(epoch)
      torch.save(model.state_dict(), model_save_path)
      print("Model saved to {}".format(model_save_path))
    

    # Calculate average validation loss for the entire validation dataset
    average_val_loss = sum(val_losses) / len(val_losses)
    print(f"Average validation Loss: {average_val_loss}")
    losses_validation.append(average_val_loss.item())
