from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model("eurosat_model2.keras")

# Define the class labels
class_labels = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

def preprocess_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")  # Convert to RGB (Ensures compatibility)
        image = image.resize((64, 64))
        image = np.array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except UnidentifiedImageError:
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    image = preprocess_image(file)

    if image is None:
        return jsonify({'error': 'Invalid image format'}), 400

    try:
        predictions = model.predict(image)
        predicted_class = class_labels[np.argmax(predictions)]
        class_probabilities = {class_labels[i]: float(predictions[0][i]) for i in range(len(class_labels))}

        return jsonify({
            'prediction': predicted_class,
            'class_probabilities': class_probabilities
        })
    except ValueError as e:
        return jsonify({'error': f'Model prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
