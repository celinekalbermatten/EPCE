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



#torch.cuda.set_device(1)
# Set the device after ensuring the correct index
#if torch.cuda.is_available():
    #torch.cuda.set_device(1)


# initalize the training options
opt = potions.Options().parse()

# ======================================
# Load the data
# ======================================

dataset = HDRDataset(mode="train", opt=opt)

# split dataset into training and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# create separate data loaders for training and validation
train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

# print the number of training and validation samples
print("Training samples: ", len(train_data_loader))
print("Validation samples: ", len(val_data_loader))

# ========================================
# Model initialization
# ========================================

# curve estimation model
model = EPCE.PPVisionTransformer()

# ========================================
# Initialization of losses and optimizer
# ========================================

# define the loss function
l1 = torch.nn.L1Loss()
# TODO: could be 
#perceptual_loss = VGGLoss()
perceptual_loss = EPCE.VGGLoss()
# define the optimizer
# TODO: could be
#optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# ==================================================
# Load the checkpoints if continuing training
# ==================================================

print(opt)

if opt.continue_train:
    try:
        start_epoch, model = load_checkpoint(model, opt.ckpt_path)

    except Exception as e:
        print(e)
        print("Checkpoint not found!")
        #start_epoch = 1
        #model.apply(weights_init)
else:
    #start_epoch = 1
    #model.apply(weights_init)
    # TODO: 
    print('else cest pas biengggggg')

if opt.print_model:
    print(model)

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

# TODO: delete later
# print information about CUDA devices
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

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

# TODO: maybe not necessary
# set the model to training mode
model.train()

# define the number of epochs
opt.epochs = 200
num_epochs = 200
print(f"# of epochs: {num_epochs}")

# initalize the loss lists
losses_train = []
losses_validation = []

for epoch in range(opt.epochs):
    print(f"-------------- Epoch # {epoch} --------------")

    epoch_start = time.time()
    total_loss = 0.0

    # check whether the learning rate needs to be updated
    if epoch > opt.lr_decay_after:
        update_lr(optimizer, epoch, opt)
        
    losses_epoch = []

    # training loop
    for batch in tqdm(train_data_loader):
        # move the batch to the device
        optimizer.zero_grad()

        # get the LDR images
        input = batch['ldr_image']
        input = input.to(device)

        # get the HDR images
        output_true = batch['hdr_image']
        output_true = output_true.to(device)

        # forward pass through the model
        output = model(input)

        l1_loss = 0
        vgg_loss = 0

        # compute the loss for the generated outputs
        for i in range(len(output)):
            l1_loss += l1(output[i], output_true[i])
            vgg_loss += perceptual_loss(output[i], output_true[i])

        # average over n iterations
        l1_loss /= len(output)
        vgg_loss /= len(output)

        # average over batches
        l1_loss = torch.mean(l1_loss)
        vgg_loss = torch.mean(vgg_loss)

        # TODO: why do we have FHDR loss function here?
        # FHDR loss function
        loss = l1_loss + (vgg_loss * 10)
        losses_epoch.append(loss.item())

        # backpropagate and optimization step
        loss.backward()
        optimizer.step()

        # output is the final reconstructed image so last in the array of outputs of n iterations
        output = output[-1]

        # backpropagate and optimization step
        loss.backward()
        optimizer.step()

        # accumulate the loss
        total_loss += loss.item()

        # save the results
        if (batch + 1) % opt.save_results_after == 0: 
            save_ldr_image(img_tensor=input, batch=0, path="./training_results/ldr_e_{}_b_{}.jpg".format(epoch, batch + 1),)
            
            save_hdr_image(img_tensor=output, batch=0, path="./training_results/generated_hdr_e_{}_b_{}.hdr".format(epoch, batch + 1),)
            
            save_hdr_image(img_tensor=output_true, batch=0, path="./training_results/gt_hdr_e_{}_b_{}.hdr".format(epoch, batch + 1),)
    

    print(f"Training loss: {losses_epoch[-1]}")
    losses_train.append(losses_epoch[-1])

    # average loss for the epoch
    # TODO: maybe this could also work (new)
    #average_loss = total_loss / len(train_data_loader)
    #losses_train.append(average_loss)

# ========================================
# Validation
# ========================================

    # validation loop -> set the model mode to evaluation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_batch, val_data in enumerate(val_data_loader):

            # get the LDR images
            input_val = val_data['ldr_image']
            input_val = input_val.to(device)

            # get the HDR images
            ground_truth_val = val_data['hdr_image']
            ground_truth_val = ground_truth_val.to(device)

            output_val = model(input_val)

            # calculate the validation loss
            l1_loss_val = 0
            vgg_loss_val = 0

            for image_val in output_val:
                l1_loss_val += l1(image_val, ground_truth_val)
                vgg_loss_val += perceptual_loss(image_val, ground_truth_val)


            """for i in range(len(output_val)):
              l1_loss_val += l1(output_val[i],ground_truth_val[i])
              vgg_loss_val += perceptual_loss(output_val[i], ground_truth_val[i])
              """

            # average over n iterations
            l1_loss_val /= len(output_val)
            vgg_loss_val /= len(output_val)

            # average over batches
            l1_loss_val = torch.mean(l1_loss_val)
            vgg_loss_val = torch.mean(vgg_loss_val)

            # final loss function
            val_loss = l1_loss_val + (vgg_loss_val * 10)
            val_losses.append(val_loss.item())

    # calculate average validation loss for the entire validation dataset
    average_val_loss = sum(val_losses) / len(val_losses)
    print(f"Average validation Loss: {average_val_loss}")
    losses_validation.append(average_val_loss)

    # set model back to training mode
    model.train()

    epoch_finish = time.time()
    time_taken = (epoch_finish - epoch_start)

    print("End of epoch {}. Time taken: {} s.".format(epoch, int(time_taken)))

    # save the checkpoints for each epoch
    save_checkpoint(epoch, model)

# ========================================
# Save the model
# ========================================

torch.save(model, 'epce.pth')

# ========================================
# Print the results
# ========================================

print("Training complete!")

print(f"Training losses: {losses_train}")
print(f"Validation losses: {losses_validation}")

plot_losses(losses_train, losses_validation, num_epochs, f"plots/_loss_{num_epochs}_epochs")
