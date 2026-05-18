import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

# --- Optimization Logic Variables ---
frame_skip = 10  # Model har 10th frame par chalega
frame_count = 0

# Purana result yaad rakhne ke liye variables
last_text = ""
last_box = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Heavy model sirf tab chalega jab frame_count 10, 20, 30... hoga
    if frame_count % frame_skip == 0:
        try:
            # detector_backend='opencv' CPU ke liye sabse fast hai
            results = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion', 'race'], 
                                       enforce_detection=False, detector_backend='opencv')
            result = results[0]
            
            # Data update karo
            region = result['region']
            last_box = (region['x'], region['y'], region['w'], region['h'])
            
            age = result['age']
            gender = max(result['gender'], key=result['gender'].get)
            emotion = result['dominant_emotion']
            race = result['dominant_race']
            
            last_text = f"{gender}, {age} | {emotion} | {race}"
        except Exception:
            # Agar face skip wale frame par detect na ho toh purana data clear mat karo
            pass 

    # Har frame par drawing cached data se hogi (No lag)
    if last_box is not None:
        x, y, w, h = last_box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, last_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('Facial Attribute Analyzer (Optimized)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()