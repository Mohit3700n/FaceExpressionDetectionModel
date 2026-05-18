import cv2
import torch
import torch.nn as nn
import numpy as np

# 1. Exact Architecture (PyTorch ko batana padega ki weights kis structure me fit karne hain)
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 2. Initialization & Loading Weights
print("Loading Model...")
# map_location='cpu' is CRITICAL here. Colab me GPU tha, par yaha hum CPU par run karenge.
device = torch.device('cpu') 
model = EmotionCNN()
model.load_state_dict(torch.load('best_emotion_model.pth', map_location=device))

# Model ko strictly evaluation mode me dalo (Dropout off)
model.eval() 

# FER2013 Labels array
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# OpenCV Face Detector (Built-in Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. Start Webcam
cap = cv2.VideoCapture(0)
print("Webcam Started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Face detection humesha Grayscale par better aur fast hoti hai
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # faces returns array of coordinates [x, y, width, height]
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Step A: Bounding Box banao
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        # Step B: Chehre ko crop karo (Region of Interest)
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        # Step C: Preprocessing (Resize -> Normalize -> Tensorify -> Add Batch/Channel Dims)
        # 1. Resize to 48x48
        roi_gray = cv2.resize(roi_gray, (48, 48))
        # 2. Normalize [0, 1]
        roi_gray = roi_gray / 255.0 
        # 3. Convert to Float32 Tensor
        tensor_image = torch.tensor(roi_gray, dtype=torch.float32)
        # 4. Add Channel and Batch dimensions -> Final Shape: (1, 1, 48, 48)
        tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)
        
        # Step D: Inference (Prediction)
        with torch.no_grad(): # Gradient calculate mat karo, ye sirf test hai
            output = model(tensor_image)
            _, prediction = torch.max(output.data, 1)
            
        final_emotion = emotion_labels[prediction.item()]
        
        # Step E: Screen par result dikhao
        cv2.putText(frame, final_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Pro Emotion Detector', frame)

    # 'q' dabane par loop break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()