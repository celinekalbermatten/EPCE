import torch
from torchvision import datasets, transforms
import OpenEXR, Imath
import numpy as np

# Function to load HDR images
def load_hdr_image(path):
    exr_image = OpenEXR.InputFile(path)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = exr_image.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    rgb = [np.frombuffer(exr_image.channel(c, pt), dtype=np.float32) for c in "RGB"]
    rgb = [np.reshape(c, (size[1], size[0])) for c in rgb]

    return np.stack(rgb, -1)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load LDR dataset
ldr_dataset = datasets.ImageFolder('s/LDR_exposure_0', transform=transform)

# Load HDR dataset
hdr_dataset = datasets.DatasetFolder('s/HDR', loader=load_hdr_image, extensions='.hdr')

# Create data loaders
ldr_loader = torch.utils.data.DataLoader(ldr_dataset, batch_size=32, shuffle=True)
hdr_loader = torch.utils.data.DataLoader(hdr_dataset, batch_size=32, shuffle=True)