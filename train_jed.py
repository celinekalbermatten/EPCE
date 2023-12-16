import torch
from torch import nn
from torch.utils.data import DataLoader
import EPCE_model
from EPCE_model import VGGLoss
import glob
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

import matplotlib.pyplot as plt

import os
import time

from data_loader import HDRDataset, decrease_data_size
import options
from util import (
    load_checkpoint,
    make_required_directories,
    mu_tonemap,
    save_checkpoint,
    save_hdr_image,
    save_ldr_image,
    update_lr,
    plot_losses,
    print_gpu_info

)

from sklearn.model_selection import train_test_split


# ======================================
# Initial training options 
# ======================================

# initalize the training options
opt = options.Options().parse()
batch_size = 1
#batch_size = opt.batch_size

# ======================================
# Load the data
# ======================================

dataset = HDRDataset(mode="train", opt=opt)
#dataset = decrease_data_size(dataset)

# split dataset into training and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

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
#model = EPCE.PPVisionTransformer().to(dtype=torch.half)
model = EPCE_model.PPVisionTransformer()
#for name, param in model.named_parameters():
    #print(f"Parameter initial: {name}, Dtype: {param.dtype}")
# decrease the size of the model from torch.32 to torch.16
#model = model.half()
#for name, param in model.named_parameters():
    #print(f"Parameter after half transformation: {name}, Dtype: {param.dtype}")


# ========================================
# Initialization of losses and optimizer
# creation of directories
# ========================================

# define the loss function
l1 = torch.nn.L1Loss()
# TODO: could be 
perceptual_loss = VGGLoss()
#perceptual_loss = EPCE_ok_32.VGGLoss()
# define the optimizer
# TODO: could be
#optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# create directories to save the training results
make_required_directories(mode="train")

# ==================================================
# Load the checkpoints if continuing training
# ==================================================

print(opt)

"""
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
    """for batch in tqdm(train_data_loader):
        # move the batch to the device
        optimizer.zero_grad()

        # get the LDR images
        input = batch['ldr_image']
        input = input.to(device)
        input = input.to(dtype=torch.half)

        # get the HDR images
        output_true = batch['hdr_image']
        output_true = output_true.to(device)
        output_true = output_true.to(dtype=torch.half)"""

    for batch, data in enumerate(train_data_loader):
        # move the batch to the device
        optimizer.zero_grad()

        # get the LDR images
        input = data["ldr_image"]
        input = input.to(device)
        #print('initial input type:', input.dtype)
        #input = input.to(dtype=torch.half)
        #print('after transformation input type:', input.dtype)
        # get the HDR images
        output_true = data["hdr_image"]
        output_true = output_true.to(device)
        #output_true = output_true.to(dtype=torch.half)

        #print('batch size:', opt.batch_size)
        #print('batch size:', batch_size)

        # forward pass through the model
        #print('cuda before')
        #print_gpu_info
        output = model(input)
        #print('cuda after')
        #print_gpu_info

        l1_loss = 0
        vgg_loss = 0

        # compute the loss for the generated outputs
        """for i in range(len(output)):
            l1_loss += l1(output[i], output_true[i])
            vgg_loss += perceptual_loss(output[i], output_true[i])"""

        for image in output:
            # add a dimension for the batch size
            image = image.unsqueeze(0)
            # expand the batch dimension to the desired batch size
            image = image.expand(batch_size, -1, -1, -1)
            #image = image.to(dtype=torch.half)
            #output_true = output_true.to(dtype=torch.half)
            l1_loss += l1(image, output_true)
            vgg_loss += perceptual_loss(image, output_true)

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

        # backpropagate and optimization 
        #loss = loss.to('cpu')
        #print('loss type before backward prob:', input.dtype)
        loss.backward()
        optimizer.step()

        # output is the final reconstructed image so last in the array of outputs of n iterations
        #output = output[-1]

        # accumulate the loss
        total_loss += loss.item()

        # save the results
        #if (batch + 1) % opt.save_results_after == 0: 
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
            #input_val = input_val.to(dtype=torch.half)

            # get the HDR images
            ground_truth_val = val_data['hdr_image']
            ground_truth_val = ground_truth_val.to(device)
            #ground_truth_val = ground_truth_val.to(dtype=torch.half)

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


#torch.save(model, 'epce.pth')

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
