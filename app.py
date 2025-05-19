import os
import uuid
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import cv2
from datetime import datetime

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'tumorvision_secret_key')

# Configure upload folder
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'nii', 'nii.gz'}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Placeholder for the AI model (to be implemented with actual model)
class TumorDetectionModel:
    def __init__(self):
        # In a real application, load the trained model here
        # self.model = tf.keras.models.load_model('models/tumor_detection_model.h5')
        print("Initializing tumor detection model (placeholder)")
        
    def preprocess_image(self, image_path):
        # In a real application, implement proper preprocessing based on model requirements
        try:
            # For demonstration, just resize the image
            img = cv2.imread(image_path)
            if img is None:
                return None
            img = cv2.resize(img, (224, 224))
            img = img / 255.0  # Normalize
            return np.expand_dims(img, axis=0)  # Add batch dimension
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image_path):
        # In a real application, use the actual model for prediction
        # For demonstration, return mock results
        preprocessed = self.preprocess_image(image_path)
        if preprocessed is None:
            return None
        
        # Mock detection results (random for demonstration)
        import random
        tumor_detected = random.choice([True, False])
        confidence = random.uniform(0.7, 0.98) if tumor_detected else random.uniform(0.02, 0.3)
        
        result = {
            'tumor_detected': tumor_detected,
            'confidence': confidence
        }
        
        if tumor_detected:
            locations = ['Right frontal lobe', 'Left temporal lobe', 'Cerebellum', 'Brain stem', 'Parietal lobe']
            sizes = ['1.2 cm x 0.8 cm', '2.3 cm x 1.8 cm', '0.9 cm x 0.7 cm', '3.1 cm x 2.5 cm']
            recommendations = [
                'Further diagnostic imaging recommended',
                'Consider biopsy for definitive diagnosis',
                'Consult with neurosurgery team',
                'Follow-up scan in 3 months',
                'Radiation therapy may be beneficial',
                'Consider chemotherapy options',
                'Monitor for changes in symptoms'
            ]
            
            result.update({
                'location': random.choice(locations),
                'size': random.choice(sizes),
                'malignancy_probability': random.uniform(0.3, 0.9),
                'recommendations': random.sample(recommendations, k=random.randint(2, 4))
            })
            
        return result

# Initialize the model
model = TumorDetectionModel()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api-docs')
def api_docs():
    return render_template('api_docs.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Analyze the image
        results = model.predict(file_path)
        
        if results is None:
            flash('Error processing the image. Please try again with a different file.', 'danger')
            return redirect(url_for('index'))
        
        # Save relative path for template
        image_file = os.path.join('static', 'uploads', filename)
        
        return render_template('results.html', image_file=image_file, results=results)
    else:
        flash('File type not allowed. Please upload a valid image file.', 'danger')
        return redirect(url_for('index'))

# API Routes
@app.route('/api/v1/analyze', methods=['POST'])
def api_analyze():
    # Check for API key (in a real app, validate against database)
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return jsonify({
            'success': False,
            'error': {
                'code': 'missing_api_key',
                'message': 'API key is required'
            }
        }), 401
    
    # In a real app, validate the API key
    # if not validate_api_key(api_key):
    #     return jsonify({
    #         'success': False,
    #         'error': {
    #             'code': 'invalid_api_key',
    #             'message': 'Invalid API key'
    #         }
    #     }), 401
    
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': {
                'code': 'missing_file',
                'message': 'No file was provided'
            }
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': {
                'code': 'empty_filename',
                'message': 'Filename is empty'
            }
        }), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Get additional parameters
        scan_type = request.form.get('scan_type', 'auto')
        include_recommendations = request.form.get('include_recommendations', 'true').lower() == 'true'
        
        # Analyze the image
        results = model.predict(file_path)
        
        if results is None:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'processing_error',
                    'message': 'Error processing the image'
                }
            }), 500
        
        # If recommendations not requested, remove them
        if not include_recommendations and 'recommendations' in results:
            del results['recommendations']
        
        return jsonify({
            'success': True,
            'data': results
        })
    else:
        return jsonify({
            'success': False,
            'error': {
                'code': 'invalid_file_format',
                'message': 'The uploaded file is not a valid MRI scan format'
            }
        }), 400

@app.route('/api/v1/status', methods=['GET'])
def api_status():
    return jsonify({
        'status': 'operational',
        'version': '1.0.0',
        'uptime': '99.98%'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)