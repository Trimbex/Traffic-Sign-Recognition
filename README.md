# Traffic Sign Recognition Project

A comprehensive machine learning project for recognizing German traffic signs using deep learning techniques.

## 🚦 Project Overview

This project implements a traffic sign recognition system using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The system can classify 43 different types of traffic signs commonly found on German roads.

## ✨ Features

- **Data Analysis**: Comprehensive analysis of the GTSRB dataset with 39,209 training images
- **Visualization**: Interactive charts showing data distribution across 43 traffic sign classes
- **Class Imbalance Detection**: Automated analysis of dataset balance with recommendations
- **CUDA Support**: GPU acceleration support for faster training and inference
- **Modern ML Stack**: Built with TensorFlow, PyTorch, and scikit-learn

## 📊 Dataset Information

- **Total Images**: 39,209 training images
- **Classes**: 43 different traffic sign categories
- **Source**: GTSRB (German Traffic Sign Recognition Benchmark)
- **Format**: RGB images with varying resolutions

### Traffic Sign Categories

The system recognizes various traffic signs including:
- Speed limits (20km/h to 120km/h)
- Warning signs (curves, slippery roads, etc.)
- Prohibition signs (no entry, no passing, etc.)
- Mandatory signs (keep right, roundabout, etc.)
- Priority and yield signs

## 🛠️ Technical Requirements

- Python 3.7+
- CUDA-compatible GPU (NVIDIA)
- Required packages:
  - TensorFlow
  - PyTorch
  - OpenCV
  - NumPy
  - Pandas
  - Matplotlib
  - scikit-learn
  - PIL (Pillow)
  - kagglehub

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/traffic-sign-recognition.git
   cd traffic-sign-recognition
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**:
   ```bash
   jupyter notebook notebook.ipynb
   ```

## 📈 Data Distribution Analysis

The project includes comprehensive data analysis:
- Bar charts showing image distribution across classes
- Pie charts displaying class proportions
- Statistical analysis of dataset balance
- Automated imbalance detection and recommendations

## 🔍 Key Findings

- **Class Imbalance**: Significant imbalance detected (10.7:1 ratio)
- **Most Common**: Speed limit 50km/h (2,250 images)
- **Least Common**: Dangerous curve left (210 images)
- **Recommendation**: Data augmentation for minority classes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- GTSRB dataset creators
- Kaggle for hosting the dataset
- Open source community for the amazing ML libraries

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.
