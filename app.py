from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os, cv2, json
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Constants
IMG_SIZE = (224, 224)
UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "saved_models/model_cnn.h5"
LABEL_MAP_PATH = "saved_models/label_map.json"

# Setup
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model and label map
model = tf.keras.models.load_model(MODEL_PATH)
label_map = json.load(open(LABEL_MAP_PATH))
inv_label_map = {v: k for k, v in label_map.items()}

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    file = request.files['image']
    if not file:
        return "No file uploaded.", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess image
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensures correct color channel order
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)
    index = np.argmax(pred)
    label = inv_label_map[index]
    confidence = float(np.max(pred)) * 100

    return render_template('predictionpage.html',
                           user_image=filename,
                           label=label.title(),
                           confidence=f"{confidence:.2f}%")

if __name__ == "__main__":
    app.run(debug=True)
