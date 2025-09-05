# 🚦 Traffic Sign Recognition System

A comprehensive deep learning project for recognizing German traffic signs using Convolutional Neural Networks (CNN). This system achieves **98.90% accuracy** on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset Information](#-dataset-information)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [GPU Support](#-gpu-support)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Project Overview

This project implements a state-of-the-art traffic sign recognition system designed to classify 43 different types of German traffic signs. The system uses a custom CNN architecture with data augmentation and advanced training techniques to achieve high accuracy suitable for real-world applications.

### Key Achievements
- **98.90% Test Accuracy** on GTSRB dataset
- **43 Traffic Sign Classes** recognition
- **Robust Data Pipeline** with comprehensive preprocessing
- **Advanced Visualization** and analysis tools
- **GPU Acceleration** support for faster training

## ✨ Key Features

### 🔍 Comprehensive Data Analysis
- **Dataset Distribution Analysis**: Detailed visualization of class distributions
- **Class Imbalance Detection**: Automated identification of underrepresented classes
- **Statistical Insights**: Mean, standard deviation, and imbalance ratio calculations
- **Visual Exploration**: Interactive charts and sample image displays

### 🧠 Advanced Model Architecture
- **Custom CNN Design**: Optimized architecture with BatchNormalization and Dropout
- **Data Augmentation**: Rotation, zoom, shift, and shear transformations
- **Smart Callbacks**: Learning rate scheduling, early stopping, and model checkpointing
- **Regularization Techniques**: Dropout and batch normalization for better generalization

### 📊 Comprehensive Evaluation
- **Confusion Matrix Analysis**: Detailed per-class performance visualization
- **Classification Report**: Precision, recall, and F1-score for all classes
- **Visual Predictions**: Side-by-side comparison of actual vs predicted labels
- **Performance Metrics**: Multiple evaluation criteria for thorough assessment

## 📊 Dataset Information

### GTSRB (German Traffic Sign Recognition Benchmark)
- **Training Images**: 39,209 images across 43 classes
- **Test Images**: 12,630 images for evaluation
- **Image Size**: Resized to 32x32 pixels for training
- **Format**: RGB color images
- **Source**: Kaggle dataset via kagglehub

### Class Distribution Analysis
- **Average Images per Class**: 911.8 images
- **Most Common Class**: Speed limit (50km/h) - 2,250 images
- **Least Common Class**: Dangerous curve left - 210 images
- **Imbalance Ratio**: 10.7:1 (requires data augmentation)

### Traffic Sign Categories

| Category | Examples |
|----------|----------|
| **Speed Limits** | 20km/h, 30km/h, 50km/h, 60km/h, 70km/h, 80km/h, 100km/h, 120km/h |
| **Warning Signs** | Dangerous curves, bumpy road, slippery road, road narrows |
| **Prohibition Signs** | No passing, no entry, no vehicles, weight restrictions |
| **Mandatory Signs** | Keep right/left, roundabout mandatory, ahead only |
| **Priority Signs** | Priority road, yield, stop, right-of-way at intersection |

## 🏗️ Model Architecture

### Custom CNN Architecture
```python
Input Layer: (32, 32, 3)
├── Conv2D(16, 3x3) + ReLU
├── Conv2D(32, 3x3) + ReLU
├── MaxPool2D(2x2)
├── BatchNormalization
├── Conv2D(64, 3x3) + ReLU
├── Conv2D(128, 3x3) + ReLU
├── MaxPool2D(2x2)
├── BatchNormalization
├── Flatten
├── Dense(512) + ReLU
├── BatchNormalization
├── Dropout(0.5)
└── Dense(43, softmax)
```

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 30 (with early stopping)
- **Data Augmentation**: Rotation, zoom, shift, shear
- **Callbacks**: Learning rate scheduling, reduce LR on plateau, early stopping

## 📈 Results

### Model Performance
- **Test Accuracy**: 98.90%
- **Training Strategy**: 70/30 train-validation split
- **Data Augmentation**: Improved model generalization
- **Regularization**: Prevented overfitting

### Per-Class Performance Highlights
- **Perfect Accuracy (100%)**: Multiple classes including speed limits and directional signs
- **High Accuracy (>95%)**: Most classes achieve excellent performance
- **Challenging Classes**: Some warning signs with slightly lower but still strong performance

### Confusion Matrix Analysis
- **Strong Diagonal Pattern**: Indicates excellent classification performance
- **Minimal Misclassifications**: Very few off-diagonal predictions
- **Class-Specific Insights**: Detailed analysis available in the notebook

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- NVIDIA GPU with CUDA support (optional but recommended)
- 8GB+ RAM recommended

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/traffic-sign-recognition.git
cd traffic-sign-recognition
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv tfenv

# Activate virtual environment
# Windows:
tfenv\Scripts\activate
# macOS/Linux:
source tfenv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify GPU Setup (Optional)
```bash
python gpu.py
```

## 💻 Usage

### Option 1: Jupyter Notebook (Recommended)
```bash
# Start Jupyter Notebook
jupyter notebook

# Open notebook.ipynb and run all cells
```

### Option 2: Direct Model Usage
```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('custom_model.keras')

# Preprocess your image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Make prediction
image = preprocess_image('your_traffic_sign.jpg')
prediction = model.predict(image)
predicted_class = np.argmax(prediction)

print(f"Predicted class: {predicted_class}")
```

## 📁 Project Structure

```
Traffic Sign Recognition/
│
├── 📓 notebook.ipynb          # Main Jupyter notebook with complete pipeline
├── 🤖 custom_model.keras     # Trained CNN model
├── 🔧 gpu.py                 # GPU detection and setup utilities
├── 📋 requirements.txt       # Python dependencies
├── 📄 README.md             # Project documentation
├── 📜 LICENSE               # MIT license
│
└── 📁 tfenv/                # Virtual environment
    ├── Include/
    ├── Lib/
    ├── Scripts/
    └── share/
```

## 🔧 Technical Details

### Data Preprocessing Pipeline
1. **Image Loading**: OpenCV and PIL for robust image handling
2. **Resizing**: Standardization to 32x32 pixels
3. **Normalization**: Pixel values scaled to [0,1] range
4. **Shuffling**: Random data shuffling for better training
5. **Train-Validation Split**: 70/30 split with stratification

### Data Augmentation Techniques
```python
ImageDataGenerator(
    rotation_range=10,      # ±10 degrees rotation
    zoom_range=0.15,        # 15% zoom in/out
    width_shift_range=0.1,  # 10% horizontal shift
    height_shift_range=0.1, # 10% vertical shift
    shear_range=0.15,       # 15% shear transformation
    horizontal_flip=False,  # No horizontal flip (preserves sign meaning)
    vertical_flip=False,    # No vertical flip (preserves sign meaning)
    fill_mode="nearest"     # Fill empty pixels with nearest values
)
```

### Advanced Training Features
- **Learning Rate Scheduling**: Automatic decay over epochs
- **Reduce LR on Plateau**: Adaptive learning rate reduction
- **Early Stopping**: Prevents overfitting with patience mechanism
- **Model Checkpointing**: Saves best model weights automatically

## 🖥️ GPU Support

### CUDA Setup Verification
The project includes `gpu.py` for comprehensive GPU setup verification:

```bash
python gpu.py
```

This script checks:
- ✅ PyTorch CUDA availability
- ✅ TensorFlow GPU detection
- ✅ NVIDIA driver status
- ✅ CUDA/cuDNN compatibility
- ✅ GPU computation test

### Performance Benefits
- **Training Speed**: 3-5x faster with GPU acceleration
- **Batch Processing**: Efficient parallel processing
- **Memory Management**: Optimized GPU memory usage

### Troubleshooting GPU Issues
If GPU is not detected:
1. **Check NVIDIA Drivers**: Run `nvidia-smi`
2. **Verify CUDA Installation**: Ensure compatible CUDA version
3. **Install GPU TensorFlow**: `pip install tensorflow[and-cuda]`
4. **Check Environment Variables**: CUDA_VISIBLE_DEVICES

## 📊 Detailed Analysis Features

### Dataset Insights
- **Interactive Visualizations**: Bar charts and pie charts for class distribution
- **Statistical Analysis**: Comprehensive dataset statistics
- **Imbalance Detection**: Automated identification of data imbalance issues
- **Sample Visualization**: Random sample displays for data exploration

### Model Evaluation
- **Confusion Matrix**: Heatmap visualization of classification performance
- **Classification Report**: Detailed precision, recall, and F1-scores
- **Training History**: Loss and accuracy curves over epochs
- **Prediction Visualization**: Visual comparison of predictions vs ground truth

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute
1. **Bug Reports**: Report issues or unexpected behavior
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit pull requests with enhancements
4. **Documentation**: Improve documentation and examples
5. **Testing**: Help test the system with different datasets

### Development Setup
```bash
# Fork the repository
git fork https://github.com/yourusername/traffic-sign-recognition.git

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git commit -m "Add your feature description"

# Push to your fork
git push origin feature/your-feature-name

# Create pull request
```

### Code Style Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Include type hints where appropriate
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ❗ License and copyright notice required

## 🙏 Acknowledgments

### Dataset and Resources
- **GTSRB Dataset**: German Traffic Sign Recognition Benchmark creators
- **Kaggle**: Platform for dataset hosting and sharing
- **KaggleHub**: Easy dataset download and management

### Open Source Libraries
- **TensorFlow**: Deep learning framework
- **PyTorch**: Alternative deep learning framework
- **OpenCV**: Computer vision library
- **scikit-learn**: Machine learning utilities
- **Matplotlib/Seaborn**: Data visualization
- **NumPy/Pandas**: Data manipulation and analysis

### Community
- **Open Source Community**: For creating and maintaining amazing ML libraries
- **Research Community**: For advancing the field of computer vision
- **Contributors**: Everyone who helps improve this project

## 📞 Support and Contact

### Getting Help
- **Issues**: Create an issue on GitHub for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: Check this README and notebook comments

### Performance Tips
1. **Use GPU**: Significantly faster training with CUDA-compatible GPU
2. **Batch Size**: Adjust based on your GPU memory
3. **Data Augmentation**: Improves model generalization
4. **Early Stopping**: Prevents overfitting and saves training time

### Common Issues and Solutions
- **Memory Errors**: Reduce batch size or use data generators
- **GPU Not Detected**: Check CUDA/cuDNN installation
- **Poor Accuracy**: Increase training epochs or adjust learning rate
- **Overfitting**: Increase dropout rate or add more data augmentation

---

**⭐ If you find this project helpful, please consider giving it a star on GitHub!**

**🔄 Stay updated by watching the repository for new releases and improvements.**