import os
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from PIL import Image

class UTKFaceDataset(Dataset):

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)

        image = Image.open(img_path)
        image = self.transform(image)

        parts = img_name.split('_')

        try:
            age = float(parts[0])
            gender = float(parts[1])
        except:
            age = 25.0
            gender = 0.0

        return image, torch.tensor(age), torch.tensor(gender)


datset_path = "UTKFace"

print("Loading dataset...")
utk_dataset = UTKFaceDataset(datset_path)

print(f"Total Images Found: {len(utk_dataset)}")

img, age, gender = utk_dataset[0]

print(f"Image Shape: {img.shape}") 
print(f"Age Label: {age.item()}")
print(f"Gender Label: {'Male' if gender.item() == 0 else 'Female'}")