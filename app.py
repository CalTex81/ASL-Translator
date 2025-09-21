import os
import time
import random
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, Response, jsonify

# --- TensorFlow / Keras imports ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
    TF_AVAILABLE = True
except Exception:
    tf = None
    load_model = None
    Input = Dense = Flatten = Conv2D = MaxPooling2D = None
    TF_AVAILABLE = False

# --- Optional OpenAI ---
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai = None
if OPENAI_API_KEY:
    try:
        import openai as _openai
        _openai.api_key = OPENAI_API_KEY
        openai = _openai
    except Exception:
        openai = None

app = Flask(__name__)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'keras_Model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'labels.txt')

# --- Load labels ---
class_names = []
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]
if not class_names:
    class_names = [chr(i) for i in range(65, 91)]

# --- Load model ---
model = None
model_loaded = False

if os.path.exists(MODEL_PATH) and TF_AVAILABLE:
    try:
        model = load_model(MODEL_PATH)
        model_loaded = True
        print("✅ Full model loaded.")
    except Exception as e:
        print("⚠️ Loading as weights-only...")
        try:
            from tensorflow.keras.models import Sequential
            model = Sequential(name="asl_model")
            model.add(Input(shape=(224, 224, 3)))
            model.add(Conv2D(32, (3,3), activation='relu'))
            model.add(MaxPooling2D())
            model.add(Flatten())
            model.add(Dense(len(class_names), activation='softmax'))
            model.load_weights(MODEL_PATH)
            model_loaded = True
            print("✅ Model weights loaded.")
        except Exception as e2:
            print("❌ Failed to load weights:", e2)
else:
    print("❌ keras_Model.h5 not found or TensorFlow unavailable. Using fallback letters.")

# --- Shared state ---
current_tentative_word = ''
current_corrected_word = ''
last_tentative = ''
cap = None
last_stable_letter = 'A'

# --- Helper functions ---
def fix_word_with_gpt(word: str) -> str:
    if not openai:
        return word
    try:
        prompt = f"Correct this possibly misspelled word from ASL detection: '{word}'. Reply with the corrected single word only."
        resp = openai.ChatCompletion.create(
            model='gpt-4o',
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip().split()[0]
    except Exception as e:
        print("OpenAI error:", e)
        return word

def predict_letter(frame: np.ndarray) -> str:
    global last_stable_letter
    if model_loaded:
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
            image_array = np.asarray(image).astype(np.float32)
            image_array = (image_array / 127.5) - 1.0
            image_array = np.expand_dims(image_array, axis=0)
            preds = model.predict(image_array, verbose=0)
            idx = int(np.argmax(preds))
            letter = class_names[idx] if idx < len(class_names) else last_stable_letter
        except Exception as e:
            print("Prediction error:", e)
            letter = last_stable_letter
    else:
        letter = random.choice(class_names)
    last_stable_letter = letter
    return letter

# --- Webcam generator ---
def generate_frames():
    global current_tentative_word, current_corrected_word, last_tentative, cap
    STABLE_THRESHOLD = 4
    letter_history = deque(maxlen=STABLE_THRESHOLD)
    letter_buffer = []
    last_correction_time = 0
    correction_cooldown = 1.0

    if cap is None or not getattr(cap, 'isOpened', lambda: False)():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            while True:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, 'No camera available', (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                ret2, buf = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        small = cv2.resize(frame, (224, 224))
        letter = predict_letter(small)
        letter_history.append(letter)

        if len(letter_history) == STABLE_THRESHOLD and all(x == letter_history[0] for x in letter_history):
            stable_letter = letter_history[0]
            if not letter_buffer or letter_buffer[-1] != stable_letter:
                letter_buffer.append(stable_letter)

        tentative = ''.join(letter_buffer)
        current_tentative_word = tentative

        now = time.time()
        if tentative and tentative != last_tentative and (now - last_correction_time) > correction_cooldown:
            current_corrected_word = fix_word_with_gpt(tentative)
            last_tentative = tentative
            last_correction_time = now

        display = frame.copy()
        cv2.putText(display, f'Tentative: {tentative}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display, f'Corrected: {current_corrected_word}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret2, buf = cv2.imencode('.jpg', display)
        if ret2:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buf.tobytes() + b'\r\n')

# --- Flask routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current')
def current():
    return jsonify({'tentative': current_tentative_word,
                    'corrected': current_corrected_word})

@app.route('/status')
def status():
    return jsonify({
        'model_path': MODEL_PATH,
        'model_loaded': model_loaded,
        'labels_count': len(class_names),
    })

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        try: cap.release()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass
