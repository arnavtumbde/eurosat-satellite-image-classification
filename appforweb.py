from flask import Flask, render_template, request, flash, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from PIL import Image

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flash messages

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model (do this once when the app starts)
try:
    model = tf.keras.models.load_model("eurosat_model2.keras")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define class labels (ensure these match your training classes)
class_labels = [
    'AnnualCrop', 
    'Forest', 
    'HerbaceousVegetation', 
    'Highway', 
    'Industrial',
    'Pasture', 
    'PermanentCrop', 
    'Residential', 
    'River', 
    'SeaLake'
]

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(64, 64)):
    """
    Preprocess the image for model prediction
    """
    try:
        # Open and resize the image
        img = Image.open(image_path)
        img = img.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension (model expects batches)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "GET":
        return render_template("index.html")  # You'll need to create this template
    
    if request.method == "POST":
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser submits empty file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Secure the filename and save to upload folder
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess the image
            processed_image = preprocess_image(filepath)
            
            if processed_image is None:
                flash('Error processing image')
                return redirect(request.url)
            
            # Make prediction if model is loaded
            if model is not None:
                try:
                    predictions = model.predict(processed_image)
                    predicted_class = class_labels[np.argmax(predictions[0])]
                    confidence = round(100 * np.max(predictions[0]), 2)
                    
                    return render_template("index.html", 
                                         prediction=predicted_class,
                                         confidence=confidence,
                                         uploaded_image=filename)
                except Exception as e:
                    flash(f'Prediction error: {str(e)}')
                    return redirect(request.url)
            else:
                flash('Model not loaded - prediction unavailable')
                return redirect(request.url)
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)