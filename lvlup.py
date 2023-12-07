import torch
from torch import nn
from torch.utils.data import DataLoader
from EPCE import  Curve_Estimation, ImageDataset
import glob
from torchvision import transforms
from torch.utils.data import DataLoader

# Obtenez une liste de tous les chemins d'images
hdr_paths = glob.glob('s/HDR/*.hdr')
ldr_paths = glob.glob('s/LDR_exposure_0/*.jpg')

# Définissez les transformations à appliquer aux images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Créez les datasets
hdr_dataset = ImageDataset(hdr_paths, transform=transform)
ldr_dataset = ImageDataset(ldr_paths, transform=transform)

# Créez les dataloaders
hdr_loader = DataLoader(hdr_dataset, batch_size=32, shuffle=True)
ldr_loader = DataLoader(ldr_dataset, batch_size=32, shuffle=True)
#. Création d'une instance du modèle
model = Curve_Estimation()

# 3. Définition de la fonction de perte
criterion = nn.MSELoss()

# 4. Définition de l'optimiseur
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Boucle d'entraînement
num_epochs = 10
for epoch in range(num_epochs):
    for ldr_images, hdr_images in zip(ldr_loader, hdr_loader):
        # Passe avant
        outputs = model(ldr_images, hdr_images)
        loss = criterion(outputs, hdr_images)
        
        # Passe arrière et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))