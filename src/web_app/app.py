"""
Flask web application for medical image classification.
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append('..')
from evaluation.visualization import generate_gradcam_tensorflow, apply_heatmap
from evaluation.metrics import evaluate_tensorflow_model

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'healthpulse_secret_key'

# Global variables
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data/models/model_best_val_acc.h5'))
UPLOAD_FOLDER = '../../data/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASS_LABELS = ['Normal', 'Pneumonia']  # Update based on your model

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize model
model = None

def load_tensorflow_model(model_path):
    """Load TensorFlow model from path."""
    return load_model(model_path)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction."""
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Normalize image
    image = image.astype(np.float32) / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def image_to_base64(image):
    """Convert image to base64 string."""
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and make prediction."""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Read and save file
        file_bytes = file.read()
        img_array = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save uploaded image
        upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
        cv2.imwrite(upload_path, image)
        
        # Preprocess image
        preprocessed_img = preprocess_image(image_rgb)
        
        # Make prediction
        global model
        if model is None:
            model = load_tensorflow_model(MODEL_PATH)
        
        prediction = model.predict(preprocessed_img)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class] * 100
        
        # Generate Grad-CAM visualization
        heatmap = generate_gradcam_tensorflow(model, preprocessed_img)
        grad_cam_img = apply_heatmap(image_rgb, heatmap)
        
        # Convert images to base64 for display
        original_b64 = image_to_base64(image_rgb)
        grad_cam_b64 = image_to_base64(grad_cam_img)
        
        # Return results
        result = {
            'original_image': original_b64,
            'grad_cam_image': grad_cam_b64,
            'prediction': {
                'class': CLASS_LABELS[predicted_class],
                'confidence': float(confidence),
                'probabilities': {CLASS_LABELS[i]: float(prediction[i]) * 100 for i in range(len(CLASS_LABELS))}
            }
        }
        
        return jsonify(result)
    
    flash('Invalid file format')
    return redirect(request.url)

@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page with model performance metrics."""
    # This would load pre-computed metrics or compute them on the fly
    # For simplicity, we're using dummy data here
    metrics = {
        'accuracy': 92.5,
        'precision': 93.2,
        'recall': 91.8,
        'f1': 92.5,
        'auc': 0.95,
        'confusion_matrix': [
            [350, 20],
            [30, 400]
        ],
        'class_names': CLASS_LABELS
    }
    
    return render_template('dashboard.html', metrics=metrics)

@app.route('/health')
def health_check():
    """API health check endpoint."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Load model on startup
    print("Loading model...")
    model = load_tensorflow_model(MODEL_PATH)
    print("Model loaded successfully!")
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=8000)