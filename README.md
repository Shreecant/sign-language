# 🤟 Real-Time Sign Language to English Translator

A real-time web-based Sign Language recognition system built using Python, OpenCV, and MediaPipe. It detects hand gestures via webcam and translates them into English letters or simple words — designed to bridge communication gaps for the deaf and hard-of-hearing community.

## 🔥 Features

- 🖐️ Hand gesture recognition using MediaPipe
- 📖 Live translation from Sign Language to English
- 🧠 Trained on custom A-Z hand sign data
- ✨ Sentence builder with spacing and delete control
- 🔧 Custom dataset collection script (supports label-based saving like A.csv, hello.csv, etc.)
- 🧼 Robust prediction using Random Forest classifier
- 🚀 Built with Python, OpenCV, MediaPipe, Scikit-learn

## 📷 Live Demo

Coming soon...

## 💻 How It Works

1. *Capture Hand Landmarks* using MediaPipe
2. *Train ML Model* (RandomForest) on custom-collected landmark data
3. *Live Translator* uses webcam feed to detect and predict signs
4. *Sentence Builder* creates readable English text with:
   - Auto space after inactivity (3s)
   - Auto delete letters if hand missing for long (5s+)
   - Word-by-word prediction in future version

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/sign-language-translator
cd sign-language-translator

# (Optional) Setup virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
