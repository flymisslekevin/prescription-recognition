# Prescription Label Recognition System

A real-time prescription label detection and OCR system using computer vision and machine learning. This project can detect prescription labels in images/video and extract text using OCR with confidence scoring.

## Features

- **Real-time prescription detection** using MobileNetV2 CNN
- **OCR text extraction** with confidence scoring
- **Multiple deployment options**: Camera-based, API integration, command-line
- **Sophisticated OCR confidence analysis** (character quality, linguistic coherence, etc.)
- **Model conversion utilities** (PyTorch to ONNX)
- **Training scripts** for custom model training

## Project Structure

```
prescriptionRecog/
├── API.py                 # Real-time camera with API integration
├── Camera.py              # Simple camera-based detection
├── Model.py               # Standalone prediction module
├── scripts/
│   ├── train.py          # Model training script
│   └── inference.py      # Command-line inference
├── dataset/              # Training/validation data
│   ├── train/
│   │   ├── labelT/       # Images with prescription labels
│   │   └── noLabelT/     # Images without labels
│   └── val/
│       ├── labelV/       # Validation images with labels
│       └── noLabelV/     # Validation images without labels
└── venv/                 # Virtual environment
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/prescription-recognition.git
   cd prescription-recognition
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision
   pip install opencv-python
   pip install pillow
   pip install pillow-heif
   pip install pytesseract
   pip install requests
   pip install onnx onnxruntime
   ```

4. **Install Tesseract OCR:**
   - **macOS:** `brew install tesseract`
   - **Ubuntu:** `sudo apt-get install tesseract-ocr`
   - **Windows:** Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

### 1. Real-time Camera Detection (API Integration)
```bash
python API.py
```
- Uses webcam for real-time detection
- Integrates with external OCR API
- Sophisticated confidence scoring
- Only sends high-confidence results to API

### 2. Simple Camera Detection
```bash
python Camera.py
```
- Basic real-time detection
- Local OCR processing
- No external API calls

### 3. Single Image Prediction
```bash
python Model.py
```
- Predicts on a single image
- Returns confidence score

### 4. Command-line Inference
```bash
python scripts/inference.py path/to/image.jpg
```
- Command-line tool for single images
- Configurable confidence threshold

### 5. Train Custom Model
```bash
python scripts/train.py
```
- Trains on your dataset
- Supports HEIC, JPG, PNG formats
- Data augmentation included

## Model Architecture

- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Task:** Binary classification (prescription label vs no label)
- **Input:** 224x224 RGB images
- **Output:** Sigmoid probability (0-1)

## OCR Confidence Scoring

The system includes a sophisticated OCR confidence analyzer that evaluates:

1. **Character Quality** (25%): OCR artifacts, unknown characters
2. **Linguistic Coherence** (30%): Letter frequency, n-gram analysis
3. **Structural Integrity** (20%): Word patterns, spacing
4. **Pattern Analysis** (15%): OCR confusion patterns, case consistency
5. **Contextual Validation** (10%): Dictionary words, numeric patterns

## Dataset Structure

Organize your training data as follows:
```
dataset/
├── train/
│   ├── labelT/     # Images containing prescription labels
│   └── noLabelT/   # Images without prescription labels
└── val/
    ├── labelV/     # Validation images with labels
    └── noLabelV/   # Validation images without labels
```

## API Integration

The system can integrate with external OCR APIs:
- **Format:** Base64 encoded images
- **Threshold:** Only sends images with OCR confidence > 65%
