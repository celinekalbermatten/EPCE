import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import random


class HDRDataset(Dataset):
    """
    Class fo a custom HDR dataset that returns a dictionary of LDR input image, HDR ground truth image and file path. 
    """
    def __init__(self, mode, opt):
        """
        Initialize the Dataset instance
        """
        self.batch_size = opt.batch_size

        # determine the dataset based on the mode
        if mode == "train":
            self.dataset_path = os.path.join(f"./dataset_old/clear/train")
            #self.dataset_path = os.path.join(f"./dataset_full_images/train")
        else:
            #self.dataset_path = os.path.join(f"./dataset_old/clear/test")
            self.dataset_path = os.path.join(f"./dataset_full_images/train")

        # define the paths for the LDR and HDR images
        self.ldr_data_path = os.path.join(self.dataset_path, "LDR")
        self.hdr_data_path = os.path.join(self.dataset_path, "HDR")

        # get the list of filenames for LDR and HDR images
        self.ldr_image_names = sorted(os.listdir(self.ldr_data_path))
        self.hdr_image_names = sorted(os.listdir(self.hdr_data_path))

    def __getitem__(self, index):
        """
        Get the element at index 'index' in the instance Dataset.
        """
        # get the path of the LDR image
        self.ldr_image_path = os.path.join(self.ldr_data_path, self.ldr_image_names[index])

        # transformations on LDR input images
        ldr_sample = Image.open(self.ldr_image_path).convert("RGB")
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
        transform_ldr = transforms.Compose(transform_list)
        ldr_tensor = transform_ldr(ldr_sample)

        # get the path of the HDR ground truth images
        self.hdr_image_path = os.path.join(self.hdr_data_path, self.hdr_image_names[index])

        # transformations on HDR ground truth 
        hdr_sample = cv2.imread(self.hdr_image_path, -1).astype(np.float32)
        transform_list = [transforms.Lambda(lambda img: torch.from_numpy(img.transpose((2, 0, 1)))),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
        transform_hdr = transforms.Compose(transform_list)
        hdr_tensor = transform_hdr(hdr_sample)

        # create a dictionary containing LDR and HDR tensors with the path of the LDR image
        sample_dict = {
            "ldr_image": ldr_tensor, #LDR image in tensor form 
            "hdr_image": hdr_tensor, #HDR image in tensor form 
            "path": self.ldr_image_path, #path of the LDR image
        }

        return sample_dict

    def __len__(self):
        """
        Return the number of LDR images that are taken in one batch.
        """
        return len(self.ldr_image_names) // self.batch_size * self.batch_size
    

def decrease_data_size(dataset):
    print('Initial size of dataset:', len(dataset))
    # List for the new dictionaries 
    new_data = []  

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

    random.seed(1)
    random.shuffle(new_data)

    print('Final size of dataset:', len(dataset))

    return new_data
