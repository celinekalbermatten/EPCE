import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import EPCE_adam
import glob
from torchvision import transforms
import numpy as np
import torch.nn as nn

from tqdm import tqdm

import os
import time

from data_loader import HDRDataset, decrease_data_size
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
    plot_losses
)

from sklearn.model_selection import train_test_split

import random


# ======================================
# Information about GPUs -> DELETE later
# ======================================

import pynvml

def get_gpu_info():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    gpu_info = []
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        gpu_info.append({
            "index": i,
            "name": gpu_name,
            "memory_total": gpu_memory.total,
            "memory_used": gpu_memory.used,
            "memory_free": gpu_memory.free
        })
    
    pynvml.nvmlShutdown()
    return gpu_info

# Retrieve and print GPU information including free memory
gpu_info = get_gpu_info()
for gpu in gpu_info:
    print(f"GPU {gpu['index']} Name: {gpu['name']}")
    print(f"   Total Memory: {gpu['memory_total'] / 1024**2} MB")
    print(f"   Used Memory : {gpu['memory_used'] / 1024**2} MB")
    print(f"   Free Memory : {gpu['memory_free'] / 1024**2} MB")
    print("-" * 20)

# ======================================
# Initial training options 
# and folder creation
# ======================================

# create the name of the folder to save the model later
pathtocheckp = './foldercheckp'

# initalize the training options
opt = potions.Options().parse()

# ======================================
# Load the data
# ======================================

dataset = HDRDataset(mode="train", opt=opt)

# decrease the size of the data
#dataset = decrease_data_size(dataset)

# split dataset into training and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# set the batch size
opt.batch_size = 1
batch_size = 1

# create separate data loaders for training and validation
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# print the number of training and validation samples
print("Training samples: ", len(train_data_loader))
print("Validation samples: ", len(val_data_loader))

# ========================================
# Model initialization
# ========================================

# curve estimation model
model = EPCE_adam.PPVisionTransformer()
# decrease the size of the model from torch.32 to torch.16
model = model.half()

# ========================================
# Initialization of losses and optimizer
# ========================================

# define the loss function
l1 = torch.nn.L1Loss()
perceptual_loss = EPCE_adam.VGGLoss()

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# ========================================
# GPU configuration
# ========================================

# set the device -> CPU or GPU
device = torch.device("cuda")
# move the model to the device
model.to(device)

# ========================================
# Training
# ========================================

# set the model to training mode
model.train()

# define the number of epochs
opt.epochs = 200
num_epochs = 200
print(f"# of epochs: {num_epochs}")

# initalize the loss lists
losses_train = []
losses_validation = []

# TODO comment
opt.lr_decay_after = 100
lr_decay_after = 100

for epoch in range(num_epochs):
    print(f"-------------- Epoch # {epoch} --------------")
    epoch_start = time.time()

    total_loss = 0.0

    # check whether the learning rate needs to be updated
    if epoch > lr_decay_after:
      update_lr(optimizer, epoch, opt)

    losses_epoch = []

    for batch in tqdm(train_data_loader):
        # print the batch size
        print('batch size:', batch_size)

        # move the batch to the device
        optimizer.zero_grad()

        # get the LDR images
        input = batch['ldr_image']
        input = input.to(device)
        input = input.to(dtype=torch.half)

        # get the HDR images
        output_true = batch['hdr_image']
        output_true = output_true.to(device)
        output_true = output_true.to(dtype=torch.half)

        # forward pass through the model
        output = model(input)

        l1_loss = 0
        vgg_loss = 0

        for i in range(len(output)):
            l1_loss += l1(output[i], output_true[i])
            vgg_loss += perceptual_loss(output[i], output_true[i])

        # average over n iterations
        l1_loss /= len(output)
        vgg_loss /= len(output)

        # average over batches
        l1_loss = torch.mean(l1_loss)
        vgg_loss = torch.mean(vgg_loss)

        # FHDR loss function
        loss = l1_loss + (vgg_loss * 10)
        losses_epoch.append(loss.item())

        # backpropagate and step
        loss.backward()
        optimizer.step()

        # accumulate the loss
        total_loss += loss.item()

        """# output is the final reconstructed image so last in the array of outputs of n iterations
        output = output[-1]

        # save the results
        if (batch + 1) % opt.save_results_after == 0: 
            save_ldr_image(img_tensor=input, batch=0, path="./training_results/ldr_e_{}_b_{}.jpg".format(epoch, batch + 1),)
            
            save_hdr_image(img_tensor=output, batch=0, path="./training_results/generated_hdr_e_{}_b_{}.hdr".format(epoch, batch + 1),)
            
            save_hdr_image(img_tensor=output_true, batch=0, path="./training_results/gt_hdr_e_{}_b_{}.hdr".format(epoch, batch + 1),)
        
    
    print(f"Training loss: {losses_epoch[-1]}")
    losses_train.append(losses_epoch[-1])
    """

    # TODO comment this part to print
    # Print the average loss for the epoch
    average_loss = total_loss / len(train_data_loader)
    losses_train.append(average_loss)

# ========================================
# Validation
# ========================================

    # validation loop -> set the model mode to evaluation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_batch, val_data in enumerate(val_data_loader):

            # get the LDR images
            input_val = val_batch['ldr_image']
            #input_val = val_data['ldr_image']
            input_val = input_val.to(device)

            # get the HDR images
            ground_truth_val = val_batch['hdr_image']
            #ground_truth_val = val_data['hdr_image']
            ground_truth_val = ground_truth_val.to(device)

            # TODO remove this
            optimizer.zero_grad()

            # forward pass through the model
            output_val = model(input_val)

            # calculate the validation loss
            l1_loss_val = 0
            vgg_loss_val = 0

            for i in range(len(output_val)):
                l1_loss_val += l1(output_val[i],ground_truth_val[i])
                vgg_loss_val += perceptual_loss(output_val[i], ground_truth_val[i])

            # average over n iterations
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
    # TODO comment next line
    print(f"Epoch [{epoch+1}/{opt.epochs}], Average Loss: {average_loss:.4f}")

    # set model back to training mode
    model.train()

    # compute the time taken per epoch
    epoch_finish = time.time()
    time_taken = (epoch_finish - epoch_start)

    print("End of epoch {}. Time taken: {} s.".format(epoch, int(time_taken)))

# ========================================
# Save the model
# ========================================

    if epoch%100 == 0:
      model_save_path = pathtochekp + '/model_odt_{}.pth'.format(epoch)
      torch.save(model.state_dict(), model_save_path)
      print("Model saved to {}".format(model_save_path))
    
    # save the checkpoints for each epoch
    save_checkpoint(epoch, model)

# ========================================
# Print and plot the results
# ========================================

print("Training complete!")

print(f"Training losses: {losses_train}")
print(f"Validation losses: {losses_validation}")

# create the plot of the losses
directory_plots = "./plots"
# create the directory if it doesn't exist
if not os.path.exists(directory_plots):
    os.makedirs(directory_plots)

plot_losses(losses_train, losses_validation, num_epochs, f"{directory_plots}/_loss_{num_epochs}_epochs")
