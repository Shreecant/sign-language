from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import numpy as np
import cv2
import mediapipe as mp

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("frame")
def handle_frame(data):
    try:
        # Strip off the header 'data:image/jpeg;base64,...'
        b64data = data.split(',')[1]
        img_data = base64.b64decode(b64data)
        np_img = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Flip and convert color
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = hands.process(rgb)

        # Return basic detection result
        if results.multi_hand_landmarks:
            emit("result", {"message": "Hand detected ✋", "count": len(results.multi_hand_landmarks)})
        else:
            emit("result", {"message": "No hand 😐", "count": 0})
    except Exception as e:
        emit("result", {"message": f"Error: {str(e)}", "count": -1})

if __name__ == "__main__":
    socketio.run(app, debug=True)
