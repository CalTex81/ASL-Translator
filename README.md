# ASL Sign Language Detector (local)

Overview
--------
This project runs an ASL sign detector locally. It streams webcam frames to a Flask app (`/video_feed`) and builds tentative ASL words from predicted letters. When configured, the app will call the OpenAI API (only to correct detected ASL words) — GPT is used strictly for spelling correction of the tentative ASL word.

Files
-----
- `app.py` — Flask app that streams `/video_feed` and exposes `/current` for the corrected word.
- `keras_Model.h5` — your trained Keras model (place the real file here).
- `labels.txt` — newline-separated class labels (A-Z or custom labels).
- `templates/index.html` — frontend page that displays the video.

Setup
-----
1. Create and activate a Python virtual environment (recommended).

Windows PowerShell example:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Place your `keras_Model.h5` and `labels.txt` in the project root.

3. (Optional) Configure OpenAI for spelling correction:

```powershell
setx OPENAI_API_KEY "your_api_key_here"
# Then restart your shell/IDE so the env var is visible to processes
```

Run
---
Start the Flask app:

```powershell
python app.py
```

Open `http://localhost:5000/` in your browser. The video stream is at `/video_feed`.

Notes
-----
- The app will not send any other user text to GPT; only the ASL tentative word is sent for correction.
- If you run into model import errors, ensure `tensorflow` is installed and your Python version is compatible.
- For production or remote camera use, modify the `cv2.VideoCapture()` source and secure the OpenAI key handling.
