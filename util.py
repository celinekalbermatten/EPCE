import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def load_checkpoint(model, ckpt_path):
    """
    Loads checkpoints for continuing training or evaluation.
    
    Parameters:
        model (torch.nn.Module): The model to load the checkpoint into.
        ckpt_path (str): Path to the checkpoint file.
    
    Returns:
        int: Start epoch from where training resumes.
        torch.nn.Module: Model with loaded checkpoint state.
    """
    start_epoch = np.loadtxt("./checkpoints/state.txt", dtype=int)
    model.load_state_dict(torch.load(ckpt_path))
    print("Resuming from epoch ", start_epoch)
    return start_epoch, model


def make_required_directories(mode):
    """
    Creates necessary directories based on the mode (train or test).
    
    Parameters:
        mode (str): Mode indicating whether it's for training or testing.
    """
    if mode == "train":
        if not os.path.exists("./checkpoints"):
            print("Making checkpoints directory")
            os.makedirs("./checkpoints")

        if not os.path.exists("./training_results"):
            print("Making training_results directory")
            os.makedirs("./training_results")
    elif mode == "test":
        if not os.path.exists("./test_results"):
            print("Making test_results directory")
            os.makedirs("./test_results")


def mu_tonemap(img):
    """
    Tonemaps HDR images using μ-law before computing loss.
    
    Parameters:
        img (torch.Tensor): HDR image tensor.
    
    Returns:
        torch.Tensor: Tonemapped HDR image tensor.
    """
    MU = 5000.0
    return torch.log(1.0 + MU * (img + 1.0) / 2.0) / np.log(1.0 + MU)


def write_hdr(hdr_image, path):
    """
    Writes HDR image in radiance (.hdr) format.
    
    Parameters:
        hdr_image (numpy.ndarray): HDR image as an array.
        path (str): Path to save the HDR image.
    """
    norm_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
    with open(path, "wb") as f:
        norm_image = (norm_image - norm_image.min()) / (
            norm_image.max() - norm_image.min()
        )  # normalisation function
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (norm_image.shape[0], norm_image.shape[1]))
        brightest = np.maximum(
            np.maximum(norm_image[..., 0], norm_image[..., 1]), norm_image[..., 2]
        )
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((norm_image.shape[0], norm_image.shape[1], 4), dtype=np.uint8)
        rgbe[..., 0:3] = np.around(norm_image[..., 0:3] * scaled_mantissa[..., None])
        rgbe[..., 3] = np.around(exponent + 128)
        rgbe.flatten().tofile(f)
        f.close()


def save_hdr_image(img_tensor, batch, path):
    """
    Pre-processes HDR image tensor before writing it.
    
    Parameters:
        img_tensor (torch.Tensor): HDR image tensor.
        batch (int): Batch index.
        path (str): Path to save the HDR image.
    """
    img = img_tensor.data[batch].cpu().float().numpy()
    img = np.transpose(img, (1, 2, 0))

    write_hdr(img.astype(np.float32), path)


def save_ldr_image(img_tensor, batch, path):
    """
    Pre-processes and writes LDR image tensor.
    
    Parameters:
        img_tensor (torch.Tensor): LDR image tensor.
        batch (int): Batch index.
        path (str): Path to save the LDR image.
    """
    img = img_tensor.data[batch].cpu().float().numpy()
    img = 255 * (np.transpose(img, (1, 2, 0)) + 1) / 2

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def save_checkpoint(epoch, model):
    """
    Saves a model checkpoint.
    
    Parameters:
        epoch (int): Epoch number.
        model (torch.nn.Module): Model to be saved.
    """
    checkpoint_path = os.path.join("./checkpoints", "epoch_" + str(epoch) + ".ckpt")
    latest_path = os.path.join("./checkpoints", "latest.ckpt")
    torch.save(model.state_dict(), checkpoint_path)
    torch.save(model.state_dict(), latest_path)
    np.savetxt("./checkpoints/state.txt", [epoch + 1], fmt="%d")
    print("Saved checkpoint for epoch", epoch)


def update_lr(optimizer, epoch, opt):
    """
    Linearly decays the model learning rate after specified epochs.
    
    Parameters:
        optimizer (torch.optim.Optimizer): Model optimizer.
        epoch (int): Current epoch.
        opt (argparse.Namespace): Command-line arguments.
    """
    new_lr = opt.lr - opt.lr * (epoch - opt.lr_decay_after) / (
        opt.epochs - opt.lr_decay_after
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

    print("Learning rate decayed. Updated LR is: %.6f" % new_lr)


def plot_losses(training_losses, validation_losses, num_epochs, path):
    """
    Plots the training and validation losses.
    
    Parameters:
        training_losses (list): List of training losses.
        validation_losses (list): List of validation losses.
        num_epochs (int): Total number of epochs.
        path (str): Path to save the plot.
    """
    plt.figure()
    plt.plot(np.linspace(1, num_epochs, num=num_epochs), training_losses, label="training")
    plt.plot(np.linspace(1, num_epochs, num=num_epochs), validation_losses, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(os.path.basename(os.path.normpath(path)))
    plt.legend()
    plt.savefig(path)


def print_gpu_info():
    """
    Prints GPU information.
    """
    try:
        output = subprocess.check_output(["nvidia-smi"])
        print(output.decode("utf-8"))  # Print the GPU information
    except subprocess.CalledProcessError as e:
        print("Error executing nvidia-smi:", e)


