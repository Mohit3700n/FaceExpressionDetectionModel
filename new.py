import cv2
import numpy as np
import os
import urllib.request
import time

# 1. Auto-Download Architecture & Weights (ResNet-10)
prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.exists("deploy.prototxt"):
    print("Downloading architecture file...")
    urllib.request.urlretrieve(prototxt_url, "deploy.prototxt")
    
if not os.path.exists("res10_300x300_ssd_iter_140000.caffemodel"):
    print("Downloading model weights...")
    urllib.request.urlretrieve(model_url, "res10_300x300_ssd_iter_140000.caffemodel")

# 2. Load the Deep Learning Face Detector
print("Loading OpenCV DNN Face Detector...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# 3. Start Webcam
cap = cv2.VideoCapture(0)
print("Lag-Free Camera Started. Press 'q' to quit.")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    h, w = frame.shape[:2]
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # 4. Preprocess frame for the neural network
    # Resize to 300x300 and apply mean subtraction (Standard practice for this model)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # 5. Pass blob through the network
    net.setInput(blob)
    detections = net.forward()

    # 6. Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] # Extract confidence probability
        
        # Sirf tabhi box draw karo jab system 60% se zyada sure ho ki wo face hai
        if confidence > 0.6: 
            # Get coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Draw green bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
            # Text dikhao
            text = f"Face: {confidence*100:.1f}%"
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow("Pro SSD Face Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()