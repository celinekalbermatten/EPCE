import numpy as np
import torch
import torch.nn as nn
from skimage.measure import compare_ssim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import HDRDataset
from options import Options
import EPCE_model
from util import make_required_directories, save_hdr_image, save_ldr_image

# ======================================
# Initial training options 
# ======================================

# initialise options
opt = Options().parse()
opt.log_scores = True

# define the batch size
batch_size = 1
opt.batch_size = 1

# print the configured options
print(opt)

# ======================================
# Load the data
# ======================================

dataset = HDRDataset(mode="test", opt=opt)
# make the images smaller by dividing them
#dataset = decrease_data_size(dataset)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# print the number of testing images
print("Testing samples: ", len(dataset))

# ========================================
# Model initialisation, 
# loading & GPU configuration
# ========================================

# initialize EPCE model
model = EPCE_model.PPVisionTransformer()

# ========================================
# GPU configuration
# ========================================

# set the device -> CPU or GPU 
device = torch.device("cuda")
# move the model to the device
model.to(device)

# ========================================
# Evaluation of the model, 
# computation of the evaluation metrics
# ========================================

# define the mean squared error loss
mse_loss = nn.MSELoss()

# load the checkpoint for the evaluation
model.load_state_dict(torch.load(opt.ckpt_path))

# make the necessary directories for saving the test results
make_required_directories(mode="test")

# initialize the evaluation metrics
avg_psnr = 0
avg_ssim = 0
avg_mse = 0

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

        # get the final output from the model
        #output = output[-1]

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
