
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load saved model
model = tf.keras.models.load_model("melanoma_model.keras")

# Class names - same order used during training
class_names = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'seborrheic keratosis',
    'squamous cell carcinoma',
    'vascular lesion'
]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )

@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = None
    confidence = None
    image_path = None
    error = None

    if request.method == 'POST':

        if 'file' not in request.files:
            error = "Please select an image."

        else:
            file = request.files['file']

            if file.filename == '':
                error = "No image selected."

            elif not allowed_file(file.filename):
                error = "Only JPG, JPEG and PNG images are allowed."

            else:
                try:
                    # Save uploaded image
                    filename = secure_filename(file.filename)
                    image_path = os.path.join(
                        app.config['UPLOAD_FOLDER'],
                        filename
                    )

                    file.save(image_path)

                    # Load image
                    img = Image.open(image_path).convert("RGB")

                    # Resize image
                    img = img.resize((180, 180))

                    # Convert to array
                    img_array = np.array(img, dtype=np.float32)

                    # Add batch dimension
                    img_array = np.expand_dims(img_array, axis=0)

                    # Prediction using saved model
                    predictions = model.predict(
                        img_array, verbose=0
                    )

                    # Find predicted class
                    index = np.argmax(predictions[0])

                    prediction = class_names[index]

                    confidence = round(
                        float(predictions[0][index] * 100), 2
                    )

                except Exception as e:
                    error = "Prediction failed: " + str(e)

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        error=error
    )


if __name__ == '__main__':
    app.run(debug=True)
