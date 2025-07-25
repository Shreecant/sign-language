import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque, Counter

# Load trained model
model = joblib.load("asl_model.pkl")

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Webcam
cap = cv2.VideoCapture(0)

# States
sentence = ""
last_prediction = ""
new_letter_ready = False
hand_visible = False
last_hand_time = time.time()
last_delete_time = None
space_added = False
prediction_buffer = deque(maxlen=5)

# Helper: extract landmarks from results
def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    landmarks = results.multi_hand_landmarks[0].landmark
    return [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    current_time = time.time()

    num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

    if num_hands >= 1:
        if not hand_visible:
            new_letter_ready = True

        hand_visible = True
        last_hand_time = current_time
        last_delete_time = None
        space_added = False

        if num_hands == 1:
            landmarks = extract_landmarks(results)
            if landmarks:
                X = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(X)[0]
                prediction_buffer.append(prediction)

                most_common = Counter(prediction_buffer).most_common(1)[0][0]

                if new_letter_ready and most_common != last_prediction:
                    sentence += most_common
                    last_prediction = most_common
                    new_letter_ready = False

                mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    else:
        if hand_visible:
            last_hand_time = current_time

        hand_visible = False
        time_since_no_hand = current_time - last_hand_time

        new_letter_ready = True

        # Add space after 3 seconds
        if time_since_no_hand > 3 and not space_added:
            if not sentence.endswith(" "):
                sentence += " "
            last_prediction = ""
            space_added = True
            last_delete_time = current_time

        # Delete letter after 5 seconds
        if time_since_no_hand > 5:
            if last_delete_time is None or (current_time - last_delete_time) >= 1:
                if len(sentence.strip()) > 0:
                    sentence = sentence[:-1]
                last_delete_time = current_time

    # === UI Enhancements ===

    # Sentence display with background
    text_size = cv2.getTextSize(f"Sentence: {sentence}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv2.rectangle(frame, (5, 10), (10 + text_size[0], 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Sentence: {sentence}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Instruction background bar
    height = frame.shape[0]
    cv2.rectangle(frame, (0, height - 100), (frame.shape[1], height), (0, 0, 0), -1)
    cv2.putText(frame, "✋ Remove hand for 3 sec = Add Space", (10, height - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "⌛ Remove hand for 5 sec = Delete Last Letter", (10, height - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Timer display when hand is removed
    if not hand_visible:
        time_since_no_hand = current_time - last_hand_time
        countdown = int(time_since_no_hand)
        action_text = ""

        if 3 <= countdown < 5:
            action_text = f"⌨ Adding space in {5 - countdown}s"
        elif countdown >= 5:
            action_text = "❌ Deleting letters..."

        if action_text:
            cv2.putText(frame, action_text, (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show final frame
    cv2.imshow("ASL Translator", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
