import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os

class OmniFaceNet(nn.Module):

    def __init__(self):
        super(OmniFaceNet, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_shared = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        ) 

        self.emotion_head = nn.Linear(512, 7)
        self.gender_head = nn.Linear(512, 1)
        self.age_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.shared_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_shared(x)

        out_emotion = self.emotion_head(x)
        out_gender = torch.sigmoid(self.gender_head(x))
        out_age = self.age_head(x)

        return out_emotion, out_gender, out_age
    

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
        image = self.transform(Image.open(img_path))

        parts = img_name.split('_')

        try:
            age = float(parts[0])
            gender = float(parts[1])
        except:
            age, gender = 25.0, 0.0
        
        return image, torch.tensor(age), torch.tensor(gender)
    

class FERDataset(Dataset):

    def __init__(self, csv_file):
        print("Loading FER2013 CSV...")
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        emotion = int(row['emotion'])

        pixels = np.array(row['pixels'].split(), dtype='float32').reshape(48, 48)
        pixels = pixels / 255.0

        image = torch.tensor(pixels).unsqueeze(0)
        return image, torch.tensor(emotion)
    


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    model = OmniFaceNet().to(device)

    print("Preparing Data Loaders...")
    utk_data = UTKFaceDataset('UTKFace')
    fer_data = FERDataset('fer2013.csv')

    utk_loader = DataLoader(utk_data, batch_size=32, shuffle=True)
    fer_loader = DataLoader(fer_data, batch_size=32, shuffle=True)

    criterion_emotion = nn.CrossEntropyLoss()
    criterion_gender = nn.BCELoss()
    criterion_age = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    print("🚀 Starting Elite Multi-Task Training...")

    for epoch in range(epochs):
        model.train()
        fer_iter = iter(fer_loader)
        utk_iter = iter(utk_loader)

        total_batches = min(len(fer_loader), len(utk_loader))

        running_emotion_loss = 0.0
        running_utk_loss = 0.0

        for i in range(total_batches):
            images_fer, labels_emotion = next(fer_iter)
            images_fer, labels_emotion = images_fer.to(device), labels_emotion.to(device)

            optimizer.zero_grad()
            out_emotion, _, _ = model(images_fer)
            loss_emotion = criterion_emotion(out_emotion, labels_emotion)
            loss_emotion.backward()
            optimizer.step()

            running_emotion_loss += loss_emotion.item()

            image_utk, labels_age, labels_gender = next(utk_iter)
            image_utk = image_utk.to(device)
            labels_age = labels_age.to(device).float().unsqueeze(1)
            labels_gender = labels_gender.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            _, out_gender, out_age = model(image_utk)

            loss_gender = criterion_gender(out_gender, labels_gender)
            loss_age = criterion_age(out_age, labels_age)

            total_utk_loss = loss_gender + (loss_age * 0.01)
            total_utk_loss.backward()
            optimizer.step()

            running_utk_loss += total_utk_loss.item()

            if(i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{i+1}/{total_batches}] | " f"FER Loss: {(running_emotion_loss/100):.4f} | UTK Loss: {(running_utk_loss/100):.4f}")
                running_emotion_loss = 0.0
                running_utk_loss = 0.0

    torch.save(model.state_dict(), 'omni_face_model.pth')
    print("✅ Training Complete! Model saved as 'omni_face_model.pth'")
    

               