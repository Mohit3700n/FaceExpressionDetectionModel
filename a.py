import cv2
import torch
import torch.nn as nn
import numpy as np
import time

# ==========================================
# 1. EMOTION MODEL SETUP (Your Custom PyTorch Model)
# ==========================================
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

print("Loading Custom Emotion Model...")
# Map to CPU strictly for local lag-free inference
emotion_device = torch.device('cpu')
emotion_model = EmotionCNN()
emotion_model.load_state_dict(torch.load('best_emotion_model.pth', map_location=emotion_device, weights_only=True))
emotion_model.eval() # Freeze layers like Dropout
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ==========================================
# 2. FACE, AGE & GENDER SETUP (OpenCV DNN CPU-Optimized)
# ==========================================
print("Loading Face, Age, and Gender Models...")
# Face Detector (ResNet-10)
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Age & Gender Detectors (Levi & Hassner)
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

# Standard normalization values trained by the original authors
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# ==========================================
# 3. REAL-TIME INFERENCE PIPELINE
# ==========================================
cap = cv2.VideoCapture(0)
print("🚀 System Online! Camera Started. Press 'q' to quit.")

frame_count = 0
frame_skip = 5 # Run heavy ML calculations only every 5th frame to maintain 30+ FPS
cached_data = [] # Stores results for the in-between frames

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    h, w = frame.shape[:2]
    frame_count += 1
    
    # Calculate Live FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Only run the full Deep Learning pipeline on selected frames
    if frame_count % frame_skip == 0:
        cached_data = [] # Clear old bounding boxes
        
        # 1. Detect Face
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Threshold to filter out weak face detections
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Prevent negative coordinates if the face goes partially out of frame
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                
                # Crop the face from the frame
                face_crop = frame[startY:endY, startX:endX]
                if face_crop.shape[0] < 20 or face_crop.shape[1] < 20: 
                    continue

                # --- A. Predict Gender ---
                blob_face = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                gender_net.setInput(blob_face)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]

                # --- B. Predict Age ---
                age_net.setInput(blob_face)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]

                # --- C. Predict Emotion (Your PyTorch Model) ---
                # Preprocess specifically for your FER2013 training conditions
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (48, 48)) / 255.0
                tensor_face = torch.tensor(gray_face, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    output = emotion_model(tensor_face)
                    _, prediction = torch.max(output.data, 1)
                emotion = emotion_labels[prediction.item()]

                # Cache this data so it can be drawn on the next 4 frames without lag
                final_text = f"{gender}, {age} | {emotion}"
                cached_data.append({"box": (startX, startY, endX, endY), "text": final_text})

    # Draw the bounding boxes and text from the cache (Runs at full speed)
    for data in cached_data:
        startX, startY, endX, endY = data["box"]
        text = data["text"]
        
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
        cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display real-time FPS counter
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow("Ultimate Facial Analyzer", frame)

    # Press 'q' to break the loop and close the camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()