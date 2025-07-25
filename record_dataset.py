import cv2
import mediapipe as mp
import pandas as pd
import time
import os

# Set how many samples per label
samples_per_label = 200

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Ask user for label
label = input("👉 Enter label (like A, B, hello): ").strip().upper()

# Define output folder and file path
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
csv_file = os.path.join(output_dir, f"{label}.csv")

# Generate column names
columns = [f"landmark_{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
columns.append("label")

# Load existing CSV or create new one with correct columns
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    df = pd.DataFrame(columns=columns)

# Start webcam
cap = cv2.VideoCapture(0)
print(f"🎥 Recording for label: {label} — Show your hand!")
count = 0

while cap.isOpened() and count < samples_per_label:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        data = []

        for lm in landmarks:
            data.extend([lm.x, lm.y, lm.z])

        data.append(label)
        df.loc[len(df)] = data
        count += 1

        mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Samples: {count}/{samples_per_label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "No hand detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Data Collector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
df.to_csv(csv_file, index=False)
print(f"✅ Done! Saved {count} samples for label '{label}' to {csv_file}")
