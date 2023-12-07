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
from options import Options
from util import (
    load_checkpoint,
    make_required_directories,
    mu_tonemap,
    save_checkpoint,
    save_hdr_image,
    save_ldr_image,
    update_lr,
)
opt = Options().parse()

dataset = HDRDataset(mode="train", opt=opt)
data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

print("Training samples: ", len(dataset))

# Define the curve estimation model
model = EPCE.Curve_Estimation()

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Move the model to the device
model.to(device)

# Set the model to training mode
model.train()

# Iterate over the dataset for multiple epochs
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in data_loader:
        # Move the batch to the device
        input = batch['ldr_image']
        output_true = batch['hdr_image']
        # Zero the gradients
        optimizer.zero_grad()
        transformer = EPCE.PPVisionTransformer()

        # Forward pass
        output = model(input,transformer.forward(input.float()))

        # Compute the loss
        loss = criterion(output, output_true)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # Accumulate the loss
        total_loss += loss.item()

    # Print the average loss for the epoch
    average_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}")

