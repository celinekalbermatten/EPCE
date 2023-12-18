import torch
from torch import nn
from torch.utils.data import DataLoader

import os
import time
from sklearn.model_selection import train_test_split

import EPCE_model
from EPCE_model import VGGLoss
from data_loader import HDRDataset
import options
from util import (make_required_directories, save_checkpoint, save_hdr_image, save_ldr_image, update_lr, plot_losses,)

# ======================================
# Information about GPUs -> DELETE later
# ======================================

import pynvml

def get_gpu_info():
    """
    Retrieve information about available GPUs using NVIDIA Management Library (NVML) via pynvml
    """
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

# retrieve and print GPU information including free memory
gpu_info = get_gpu_info()
for gpu in gpu_info:
    print(f"GPU {gpu['index']} Name: {gpu['name']}")
    print(f"   Total Memory: {gpu['memory_total'] / 1024**2} MB")
    print(f"   Used Memory : {gpu['memory_used'] / 1024**2} MB")
    print(f"   Free Memory : {gpu['memory_free'] / 1024**2} MB")
    print("-" * 20)

# ======================================
# Initial training options 
# ======================================

# initalize the training options
opt = options.Options().parse()
batch_size = 1
opt.batch_size = 1
opt.save_results_after = 10

# ======================================
# Load the data
# ======================================

dataset = HDRDataset(mode="train", opt=opt)

# split dataset into training and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# create separate data loaders for training and validation
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# print the number of training and validation samples
print("Training samples: ", len(train_data_loader) * batch_size)
print("Validation samples: ", len(val_data_loader) * batch_size)

# ========================================
# Model initialization and
# GPU configuration
# ========================================

# set the device -> CPU or GPU 
device = torch.device("cuda")

# curve estimation model
model = EPCE_model.PPVisionTransformer()
# move the model to the device
model.to(device)

# ========================================
# Initialization of losses and optimizer
# creation of directories
# ========================================

# define the loss function
l1 = torch.nn.L1Loss()
perceptual_loss = VGGLoss()
# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# create directories to save the training results
make_required_directories(mode="train")

# ========================================
# Training
# ========================================

# set the model to training mode
model.train()

# define the number of epochs
opt.epochs = 200
num_epochs = 200
print(f"Number of epochs: {num_epochs}")

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
    for batch, data in enumerate(train_data_loader):
        # move the batch to the device
        optimizer.zero_grad()

        # get the LDR images
        input = data["ldr_image"]
        input = input.to(device)
        # get the HDR images
        output_true = data["hdr_image"]
        output_true = output_true.to(device)

        # forward pass through the model
        output = model(input)
    
        l1_loss = 0
        vgg_loss = 0

        # compute the loss for the generated outputs
        for image in output:
            # add a dimension for the batch size
            image = image.unsqueeze(0)
            # expand the batch dimension to the desired batch size
            image = image.expand(batch_size, -1, -1, -1)
            # compute the loss
            l1_loss += l1(image, output_true)
            vgg_loss += perceptual_loss(image, output_true)

        # average over n iterations
        l1_loss /= len(output)
        vgg_loss /= len(output)

        # average over batches
        l1_loss = torch.mean(l1_loss)
        vgg_loss = torch.mean(vgg_loss)

        # loss function
        loss = l1_loss + (vgg_loss * 10)
        losses_epoch.append(loss.item())

        # backpropagation and optimization 
        loss.backward()
        optimizer.step()

        # accumulate the loss
        total_loss += loss.item()

        """
        # save the results
        if (batch + 1) % opt.save_results_after == 0: 
            save_ldr_image(img_tensor=input, batch=0, path="./training_results/ldr_e_{}_b_{}.jpg".format(epoch, batch + 1),)
            
            save_hdr_image(img_tensor=output, batch=0, path="./training_results/generated_hdr_e_{}_b_{}.hdr".format(epoch, batch + 1),)
            
            save_hdr_image(img_tensor=output_true, batch=0, path="./training_results/gt_hdr_e_{}_b_{}.hdr".format(epoch, batch + 1),)
        """

    print(f"Training loss: {losses_epoch[-1]}")
    losses_train.append(losses_epoch[-1])

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

            # forward pass through the model
            output_val = model(input_val)

            # calculate the validation loss
            l1_loss_val = 0
            vgg_loss_val = 0

            for image_val in output_val:
                # add a dimension for the batch size
                image_val = image_val.unsqueeze(0)
                # expand the batch dimension to the desired batch size
                image_val = image_val.expand(batch_size, -1, -1, -1)
                # compute the loss
                l1_loss_val += l1(image_val, ground_truth_val)
                vgg_loss_val += perceptual_loss(image_val, ground_truth_val)

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
    print(f"Validation Loss: {average_val_loss}")
    losses_validation.append(average_val_loss)

    # set model back to training mode
    model.train()

    # compute the time taken per epoch
    epoch_finish = time.time()
    time_taken = (epoch_finish - epoch_start)

    print("End of epoch {}. Time taken: {} s.".format(epoch, int(time_taken)))

# ========================================
# Save the model
# ========================================

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
