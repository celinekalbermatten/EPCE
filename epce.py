import torch
from torch import nn
from torch.utils.data import DataLoader
import EPCE
import glob
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

import matplotlib.pyplot as plt

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
    plot_losses

)

from sklearn.model_selection import train_test_split


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

# Retrieve and print GPU information
gpu_info = get_gpu_info()
for gpu in gpu_info:
    print(f"GPU {gpu['index']} Name: {gpu['name']}")




#torch.cuda.set_device(1)
# Set the device after ensuring the correct index
if torch.cuda.is_available():
    torch.cuda.set_device(1)


opt = potions.Options().parse()

dataset = HDRDataset(mode="train", opt=opt)

# split dataset into training and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# create separate data loaders for training and validation
train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
print("Training samples: ", len(train_data_loader))
print("Validation samples: ", len(val_data_loader))


# Define the curve estimation model
model = EPCE.PPVisionTransformer()
# Define the loss function

l1 = torch.nn.L1Loss()
perceptual_loss = EPCE.VGGLoss()
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

"""
# ========================================
# GPU configuration
# ========================================

str_ids = opt.gpu_ids.split(",")
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)

# set GPU device
if len(opt.gpu_ids) > 0:
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= len(opt.gpu_ids)

    torch.cuda.set_device(opt.gpu_ids[0])

    if len(opt.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    model.cuda()
"""

# Print information about CUDA devices
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# Set the device (CPU or GPU)
device = torch.device("cuda:0")


# Move the model to the device
model.to(device)
# Set the model to training mode
model.train()
# Iterate over the dataset for multiple epochs
losses = []
losses_validation = []

opt.epochs = 200
num_epochs = 200

for epoch in range(opt.epochs):
    print(f"-------------- Epoch # {epoch} --------------")

    epoch_start = time.time()

    if epoch > opt.lr_decay_after:
        update_lr(optimizer, epoch, opt)
    total_loss = 0.0
    for batch in tqdm(train_data_loader):
    #for batch, data in enumerate(train_data_loader):
        # Move the batch to the device
        optimizer.zero_grad()
        input = batch['ldr_image']
        input = input.to(device)
        #input = data["ldr_image"].data.cuda()
        output_true = batch['hdr_image']
        output_true = output_true.to(device)
        #output_true = data["hdr_image"].data.cuda()
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
            #input_val = val_data["ldr_image"].data.cuda()

            ground_truth_val = batch['hdr_image']
            ground_truth_val = ground_truth_val.to(device)
            #ground_truth_val = val_data["hdr_image"].data.cuda()
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
    if epoch%10 == 0:
        save_ldr_image(img_tensor=input, batch=0, path="./training_results/ldr_e_{}_b_{}.jpg".format(epoch, batch + 1),)
            
        save_hdr_image(img_tensor=output, batch=0, path="./training_results/generated_hdr_e_{}_b_{}.hdr".format(epoch, batch + 1),)
            
        save_hdr_image(img_tensor=output_true, batch=0, path="./training_results/gt_hdr_e_{}_b_{}.hdr".format(epoch, batch + 1),)
      


    # Calculate average validation loss for the entire validation dataset
    average_val_loss = sum(val_losses) / len(val_losses)
    print(f"Average validation Loss: {average_val_loss}")
    losses_validation.append(average_val_loss.item())

    print(f"Training losses: {losses}")
    print(f"Validation losses: {losses_validation}")

    epoch_finish = time.time()
    time_taken = (epoch_finish - epoch_start)

    print("End of epoch {}. Time taken: {} s.".format(epoch, int(time_taken)))

    plot_losses(losses, losses_validation, num_epochs, f"plots/_loss_{num_epochs}_epochs")


