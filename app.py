from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Traffic Sign Recognition API",
    description="AI-powered traffic sign recognition using CNN",
    version="1.0.0"
)

# Traffic sign class labels
CLASSES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)', 
    2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 
    4: 'Speed limit (70km/h)', 
    5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 
    7: 'Speed limit (100km/h)', 
    8: 'Speed limit (120km/h)', 
    9: 'No passing', 
    10: 'No passing veh over 3.5 tons', 
    11: 'Right-of-way at intersection', 
    12: 'Priority road', 
    13: 'Yield', 
    14: 'Stop', 
    15: 'No vehicles', 
    16: 'Veh > 3.5 tons prohibited', 
    17: 'No entry', 
    18: 'General caution', 
    19: 'Dangerous curve left', 
    20: 'Dangerous curve right', 
    21: 'Double curve', 
    22: 'Bumpy road', 
    23: 'Slippery road', 
    24: 'Road narrows on the right', 
    25: 'Road work', 
    26: 'Traffic signals', 
    27: 'Pedestrians', 
    28: 'Children crossing', 
    29: 'Bicycles crossing', 
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing', 
    32: 'End speed + passing limits', 
    33: 'Turn right ahead', 
    34: 'Turn left ahead', 
    35: 'Ahead only', 
    36: 'Go straight or right', 
    37: 'Go straight or left', 
    38: 'Keep right', 
    39: 'Keep left', 
    40: 'Roundabout mandatory', 
    41: 'End of no passing', 
    42: 'End no passing veh > 3.5 tons'
}

# Global model variable
model = None

def load_model():
    """Load the trained traffic sign recognition model"""
    global model
    try:
        model = tf.keras.models.load_model('custom_model.keras')
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (32x32)
        image = image.resize((32, 32))
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32)
        image_array = image_array / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Error processing image")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    """Serve the main HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🚦 Traffic Sign Recognition</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                padding: 40px;
                max-width: 800px;
                width: 100%;
                text-align: center;
            }
            
            .header {
                margin-bottom: 30px;
            }
            
            .header h1 {
                color: #333;
                font-size: 2.5em;
                margin-bottom: 10px;
                font-weight: 700;
            }
            
            .header p {
                color: #666;
                font-size: 1.1em;
                line-height: 1.5;
            }
            
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 40px;
                margin: 30px 0;
                background: #f8f9ff;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            
            .upload-area:hover {
                border-color: #764ba2;
                background: #f0f2ff;
                transform: translateY(-2px);
            }
            
            .upload-area.dragover {
                border-color: #764ba2;
                background: #e8ebff;
                transform: scale(1.02);
            }
            
            .upload-icon {
                font-size: 3em;
                color: #667eea;
                margin-bottom: 15px;
            }
            
            .upload-text {
                font-size: 1.2em;
                color: #333;
                margin-bottom: 10px;
            }
            
            .upload-subtext {
                color: #666;
                font-size: 0.9em;
            }
            
            #imageInput {
                display: none;
            }
            
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 25px;
                font-size: 1em;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin: 10px;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            
            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .preview-section {
                display: none;
                margin: 30px 0;
            }
            
            .image-preview {
                max-width: 300px;
                max-height: 300px;
                border-radius: 10px;
                box-shadow: 0 10px 20px rgba(0,0,0,0.1);
                margin: 20px auto;
                display: block;
            }
            
            .result-section {
                display: none;
                margin: 30px 0;
                padding: 25px;
                background: linear-gradient(135deg, #f8f9ff 0%, #e8f4f8 100%);
                border-radius: 15px;
                border-left: 5px solid #667eea;
            }
            
            .result-title {
                font-size: 1.3em;
                color: #333;
                margin-bottom: 15px;
                font-weight: 600;
            }
            
            .prediction {
                font-size: 1.5em;
                color: #667eea;
                font-weight: 700;
                margin-bottom: 10px;
            }
            
            .confidence {
                font-size: 1.1em;
                color: #666;
            }
            
            .confidence-bar {
                width: 100%;
                height: 10px;
                background: #e0e0e0;
                border-radius: 5px;
                margin: 10px 0;
                overflow: hidden;
            }
            
            .confidence-fill {
                height: 100%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                transition: width 0.5s ease;
            }
            
            .loading {
                display: none;
                color: #667eea;
                font-size: 1.1em;
                margin: 20px 0;
            }
            
            .spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .error {
                color: #e74c3c;
                background: #ffeaea;
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid #e74c3c;
                margin: 20px 0;
                display: none;
            }
            
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            
            .stat-card {
                background: linear-gradient(135deg, #f8f9ff 0%, #e8f4f8 100%);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            
            .stat-number {
                font-size: 2em;
                font-weight: 700;
                color: #667eea;
            }
            
            .stat-label {
                color: #666;
                font-size: 0.9em;
                margin-top: 5px;
            }
            
            @media (max-width: 600px) {
                .container {
                    padding: 20px;
                    margin: 10px;
                }
                
                .header h1 {
                    font-size: 2em;
                }
                
                .upload-area {
                    padding: 20px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚦 Traffic Sign Recognition</h1>
                <p>Upload an image of a traffic sign and let our AI identify it with high accuracy!</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">43</div>
                    <div class="stat-label">Sign Types</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">98.9%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">CNN</div>
                    <div class="stat-label">Deep Learning</div>
                </div>
            </div>
            
            <div class="upload-area" onclick="document.getElementById('imageInput').click()">
                <div class="upload-icon">📸</div>
                <div class="upload-text">Click to upload or drag & drop</div>
                <div class="upload-subtext">Supports JPG, PNG, GIF (Max 5MB)</div>
            </div>
            
            <input type="file" id="imageInput" accept="image/*" onchange="handleImageUpload(event)">
            
            <div class="preview-section" id="previewSection">
                <img id="imagePreview" class="image-preview" alt="Preview">
                <br>
                <button class="btn" onclick="predictImage()" id="predictBtn">🔍 Recognize Sign</button>
                <button class="btn" onclick="resetApp()" style="background: #e74c3c;">🔄 Reset</button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                Analyzing image...
            </div>
            
            <div class="error" id="error"></div>
            
            <div class="result-section" id="resultSection">
                <div class="result-title">🎯 Recognition Result</div>
                <div class="prediction" id="prediction"></div>
                <div class="confidence" id="confidence"></div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidenceFill"></div>
                </div>
            </div>
        </div>
        
        <script>
            let uploadedFile = null;
            
            // Drag and drop functionality
            const uploadArea = document.querySelector('.upload-area');
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            });
            
            function handleImageUpload(event) {
                const file = event.target.files[0];
                if (file) {
                    handleFile(file);
                }
            }
            
            function handleFile(file) {
                // Validate file type
                if (!file.type.startsWith('image/')) {
                    showError('Please select a valid image file.');
                    return;
                }
                
                // Validate file size (5MB limit)
                if (file.size > 5 * 1024 * 1024) {
                    showError('File size must be less than 5MB.');
                    return;
                }
                
                uploadedFile = file;
                
                // Show preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    document.getElementById('imagePreview').src = e.target.result;
                    document.getElementById('previewSection').style.display = 'block';
                    hideError();
                    hideResult();
                };
                reader.readAsDataURL(file);
            }
            
            async function predictImage() {
                if (!uploadedFile) {
                    showError('Please select an image first.');
                    return;
                }
                
                showLoading();
                hideError();
                hideResult();
                
                const formData = new FormData();
                formData.append('file', uploadedFile);
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        showResult(result);
                    } else {
                        showError(result.detail || 'An error occurred during prediction.');
                    }
                } catch (error) {
                    showError('Network error. Please try again.');
                    console.error('Error:', error);
                } finally {
                    hideLoading();
                }
            }
            
            function showResult(result) {
                document.getElementById('prediction').textContent = result.predicted_class;
                document.getElementById('confidence').textContent = `Confidence: ${result.confidence}%`;
                document.getElementById('confidenceFill').style.width = `${result.confidence}%`;
                document.getElementById('resultSection').style.display = 'block';
            }
            
            function showLoading() {
                document.getElementById('loading').style.display = 'block';
                document.getElementById('predictBtn').disabled = true;
            }
            
            function hideLoading() {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('predictBtn').disabled = false;
            }
            
            function showError(message) {
                document.getElementById('error').textContent = message;
                document.getElementById('error').style.display = 'block';
            }
            
            function hideError() {
                document.getElementById('error').style.display = 'none';
            }
            
            function hideResult() {
                document.getElementById('resultSection').style.display = 'none';
            }
            
            function resetApp() {
                uploadedFile = null;
                document.getElementById('imageInput').value = '';
                document.getElementById('previewSection').style.display = 'none';
                hideError();
                hideResult();
                hideLoading();
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict_traffic_sign(file: UploadFile = File(...)):
    """Predict traffic sign from uploaded image"""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read and process the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_id = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)
        
        # Get class name
        predicted_class = CLASSES.get(predicted_class_id, "Unknown")
        
        return JSONResponse({
            "predicted_class": predicted_class,
            "class_id": predicted_class_id,
            "confidence": round(confidence, 2),
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "message": "Traffic Sign Recognition API is running!"
    }

@app.get("/classes")
async def get_classes():
    """Get all available traffic sign classes"""
    return {
        "classes": CLASSES,
        "total_classes": len(CLASSES)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
