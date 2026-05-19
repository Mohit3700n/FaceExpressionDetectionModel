import cv2
import torch
import torch.nn as nn
import numpy as np
import time

# ==========================================
# 1. ARCHITECTURE (Training wala same structure zaroori hai)
# ==========================================
class OmniFaceNet(nn.Module):
    def __init__(self):
        super(OmniFaceNet, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc_shared = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(), nn.Dropout(0.5)
        )
        self.emotion_head = nn.Linear(512, 7)
        self.gender_head = nn.Linear(512, 1)
        self.age_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.shared_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_shared(x)
        return self.emotion_head(x), torch.sigmoid(self.gender_head(x)), self.age_head(x)

# ==========================================
# 2. LOAD MODELS
# ==========================================
print("Loading OmniFaceNet (CPU Mode)...")
model = OmniFaceNet()
# map_location='cpu' zaroori hai kyunki train GPU par hua tha, par chal CPU par raha hai
model.load_state_dict(torch.load('omni_face_model.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval() # Freeze dropout layers

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

print("Loading Face Detector...")
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# ==========================================
# 3. REAL-TIME PIPELINE
# ==========================================
cap = cv2.VideoCapture(0)
print("🚀 Custom Model Online! Press 'q' to quit.")

frame_skip = 5 
frame_count = 0
cached_data = []
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    h, w = frame.shape[:2]
    frame_count += 1
    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    if frame_count % frame_skip == 0:
        cached_data = []
        
        # Detect Face
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                
                face_crop = frame[startY:endY, startX:endX]
                if face_crop.shape[0] < 20 or face_crop.shape[1] < 20: continue

                # --- PREPROCESSING FOR OMNIFACENET ---
                # Model exactly 48x48 Grayscale expect kar raha hai (Normalize 0-1)
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (48, 48)) / 255.0
                tensor_face = torch.tensor(resized_face, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                # --- INFERENCE ---
                with torch.no_grad():
                    out_emotion, out_gender, out_age = model(tensor_face)
                
                # --- POST-PROCESSING ---
                # 1. Emotion: Jiski probability sabse zyada hai
                emotion = emotion_labels[torch.argmax(out_emotion).item()]
                
                # 2. Gender: 0 = Male, 1 = Female (Threshold 0.5)
                gender = "Male" if out_gender.item() < 0.5 else "Female"
                
                # 3. Age: Direct number
                age = int(out_age.item())

                final_text = f"{gender}, {age} | {emotion}"
                cached_data.append({"box": (startX, startY, endX, endY), "text": final_text})

    # Draw from cache
    for data in cached_data:
        startX, startY, endX, endY = data["box"]
        text = data["text"]
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
        cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow("Custom OmniFace AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()