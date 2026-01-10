# Concrete Crack Detection - ML Zoomcamp 2025 Capstone

An automated crack detection system using deep learning to classify concrete surface images as cracked or non-cracked. Built with MobileNetV2 for optimal deployment efficiency.

## Features

- **Deep Learning Model**: MobileNetV2-based classifier with high accuracy
- **Optimized**: TFLite model for fast inference and efficient deployment. TFLite format chosen for compatibility with Fly.io's Firecracker VMs
- **Efficient**: Selected for smaller size, faster inference, and lower cloud costs
- **REST API**: Flask-based API for easy integration
- **Containerized**: Docker image ready for cloud deployment
- **Fly.io Ready**: Configured for deployment on Fly.io platform

## Quick Start

### 1. Install Dependencies

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

### 2. Download Dataset

```bash
# Option 1: Using Kaggle CLI
kaggle datasets download -d arunrk7/surface-crack-detection
unzip surface-crack-detection.zip -d data/

# Option 2: Manual download from
# https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
```

### 3. Train Model

```bash
# Run training script
uv run python train.py

# Or train with custom settings
uv run python train.py --data-dir data/ --epochs 20
```

This will:
- Train MobileNetV2 model on the crack detection dataset
- Save the best model to [models/crack_detector.h5](models/crack_detector.h5)
- Convert and save TFLite model to [models/crack_detector.tflite](models/crack_detector.tflite)
- Save training metrics to [models/training_results.json](models/training_results.json)

### 4. Test API Locally

```bash
# Start Flask server
uv run python predict.py

# In another terminal, test the API
uv run python test_api.py --info
uv run python test_api.py --health

# Test with crack images (Positive)
uv run python test_api.py --image data/Positive/00001.jpg
uv run python test_api.py --image data/Positive/00007.jpg

# Test with non-crack images (Negative)
uv run python test_api.py --image data/Negative/00001.jpg
uv run python test_api.py --image data/Negative/00010.jpg
```

## Docker Deployment

### Build and Run Locally

```bash
# Build Docker image
docker build -t crack-detector:latest .

# Run container
docker run -p 8080:8080 crack-detector:latest

# Test the containerized API
curl http://localhost:8080/health
```

### Test with Sample Images

```bash
# Test with crack image (Positive)
curl -X POST -F "image=@data/Positive/00001.jpg" http://localhost:8080/predict

# Test with non-crack image (Negative)
curl -X POST -F "image=@data/Negative/00001.jpg" http://localhost:8080/predict
```

## Fly.io Deployment

### Prerequisites

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login (requires account)
fly auth login
```

### Deploy

```bash
# Launch app (first time only)
fly launch --name ml-2025-capstone-concrete-crack-detector

# Deploy
fly deploy

# Check status
fly status

# View logs
fly logs

# Get app URL
fly info
```

### Test Deployed API

```bash
# Test deployed endpoint
curl https://ml-2025-capstone-concrete-crack-detector.fly.dev/health

# Make prediction with crack image (Positive)
curl -X POST -F "image=@data/Positive/00001.jpg" \
  https://ml-2025-capstone-concrete-crack-detector.fly.dev/predict

# Make prediction with non-crack image (Negative)
curl -X POST -F "image=@data/Negative/00001.jpg" \
  https://ml-2025-capstone-concrete-crack-detector.fly.dev/predict
```

## Project Structure

```
.
├── README.md                           # Project documentation
├── pyproject.toml                      # Project dependencies (uv)
├── uv.lock                             # Locked dependencies
│
├── data/                               # Dataset (not in git)
│   ├── Positive/                       # Crack images (20,000)
│   └── Negative/                       # Non-crack images (20,000)
│
├── notebooks/
│   └── notebook.ipynb                  # EDA + model training experiments
│
├── models/                             # Trained models (not in git)
│   ├── crack_detector.tflite           # TFLite optimized model
│   └── training_results.json           # Training metrics
│
├── train.py                            # Training script
├── predict.py                          # Flask API for predictions
├── test_api.py                         # API testing script
├── Dockerfile                          # Container definition
└── fly.toml                            # Fly.io configuration
```

## API Endpoints

### GET `/`
Returns API information and available endpoints.

### GET `/health`
Health check endpoint.

### POST `/predict`
Make crack detection prediction on a single image.

**Parameters:**
- `image` (file): Image file to classify
- `threshold` (float, optional): Classification threshold (default: 0.5)

**Response:**
```json
{
  "prediction": "crack",
  "probability": 0.9876,
  "confidence": 0.9876,
  "threshold": 0.5,
  "filename": "image.jpg",
  "model_type": "TFLite"
}
```

### POST `/batch_predict`
Make predictions on multiple images.

**Example (Docker/localhost):**
```bash
# Predict multiple images in one request
curl -X POST \
  -F "image1=@data/Positive/00001.jpg" \
  -F "image2=@data/Negative/00001.jpg" \
  -F "image3=@data/Positive/00007.jpg" \
  http://localhost:8080/batch_predict
```

**Response:**
```json
{
  "results": [
    {
      "prediction": "crack",
      "probability": 0.9876,
      "confidence": 0.9876,
      "threshold": 0.5,
      "filename": "00001.jpg"
    },
    {
      "prediction": "no_crack",
      "probability": 0.0234,
      "confidence": 0.9766,
      "threshold": 0.5,
      "filename": "00001.jpg"
    },
    {
      "prediction": "crack",
      "probability": 0.9654,
      "confidence": 0.9654,
      "threshold": 0.5,
      "filename": "00007.jpg"
    }
  ],
  "count": 3
}
```

## Dataset

- **Name**: Concrete Crack Images for Classification
- **Total Images**: 40,000 balanced)
- **Classes**: 2 (Positive/Crack, Negative/No Crack)
- **Dimensions**: 227x227 pixels, RGB
- **Source**: [Kaggle](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)

## Model Architecture

**MobileNetV2** with transfer learning:
- Pre-trained on ImageNet
- Frozen base model (feature extraction)

### Model Selection Rationale

MobileNetV2 was selected over EfficientNetB0 despite both achieving similar accuracy (99.7% vs 99.8%):

**Key advantages:**
- **Smaller model size**: ~15MB vs ~18-20MB
- **Faster inference**: Optimized for mobile-first architecture with fewer operations
- **Lower memory footprint**: Better for concurrent request handling
- **Cost efficiency**: Lower bandwidth, storage, and compute costs
- **Cloud deployment**: Better suited for Fly.io's free tier and small instances

The 0.1% accuracy difference is statistically insignificant for this binary classification task, making deployment efficiency the decisive factor.

### Model Format: TFLite vs ONNX

TFLite was selected over ONNX for the deployment format due to Fly.io platform compatibility:

**Fly.io Compatibility:**
- Fly.io uses Firecracker microVMs with strict security policies that prevent executable stacks
- ONNX Runtime binaries require executable stack permissions, causing deployment failures
- TFLite works seamlessly with Fly.io's infrastructure without additional workarounds

**Additional Benefits:**
- **Smaller size**: ~5MB (TFLite) vs ~15MB (ONNX) - 66% reduction
- **Native TensorFlow**: Part of TensorFlow package, no additional dependencies
- **Mobile-optimized**: Originally designed for edge deployment
- **Wide adoption**: Battle-tested on millions of mobile devices

### Performance Metrics

- Accuracy: 99.7%
- Precision: 99.5%
- Recall: 99.9%
- F1-Score: 99.7%
- Model Size: ~14MB (Keras), ~5MB (TFLite)
- Inference Speed: ~20-30ms per image (CPU)

## Development

### Run Jupyter Notebook

```bash
# Start Jupyter
uv run jupyter notebook

# Open notebooks/notebook.ipynb
```

### Installing Additional Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name
```

## Environment Variables

Configure the API using environment variables:

- `TFLITE_PATH`: Path to TFLite model (default: `models/crack_detector.tflite`)
- `THRESHOLD`: Classification threshold (default: `0.5`)
- `PORT`: Server port (default: `8080`)
- `DEBUG`: Enable debug mode (default: `false`)