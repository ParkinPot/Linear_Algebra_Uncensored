# Linear Algebra - Image Deblurring (Inference-Only)

An easy-to-use application that demonstrates modern image deblurring with a pre-trained MPRNet model, while highlighting core linear algebra ideas: matrix convolution and cosine similarity.

## 📁 Project Structure

```
organized_project/
├── main.py                 # Main GUI Application
├── config.py              # Configuration settings
├── requirements.txt       # Dependencies
├── models/                # Neural network models
│   ├── MPRNet.py         # MPRNet architecture implementation
│   └── model_deblurring.pth # Pre-trained model weights
├── datasets/              # Image datasets
│   └── blurred_images/    # Blurred test images (300 images)
├── scripts/               # Python scripts
│   ├── deblur_demo.py     # Enhanced command-line demo
│   └── demo.py           # Official MPRNet demo
├── utils/                 # Utility functions
│   ├── __init__.py       # Package initialization
│   ├── image_utils.py    # Image processing utilities
│   ├── math_utils.py     # Mathematical operations
│   └── model_utils.py    # Model utilities
├── results/               # Output results
└── README.md             # This file
```

## 🚀 How to Run

### Option 1: GUI Application (Recommended)
```bash
python main.py
```
This launches a user-friendly GUI with:
- Image selection (browse or use samples)
- Real-time processing
- Side-by-side comparison
- Cosine similarity calculation
- **Mathematical Concepts Explanation**
- Professional interface

### Method 2: Command Line Interface
```bash
# Interactive mode
python scripts/deblur_demo.py

# Process specific sample image
python scripts/deblur_demo.py --sample 1

# Process custom image
python scripts/deblur_demo.py --image path/to/your/image.jpg

# Save results
python scripts/deblur_demo.py --sample 1 --save results/output.png
```

### Option 2: Official MPRNet Demo
```bash
python scripts/demo.py --task Deblurring --input_dir datasets/blurred_images --result_dir results
```

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- `torch>=1.9.0` - PyTorch framework
- `torchvision>=0.10.0` - Computer vision utilities
- `opencv-python>=4.5.0` - Image processing
- `matplotlib>=3.4.0` - Plotting and visualization
- `numpy>=1.21.0` - Numerical computing
- `Pillow>=8.3.0` - Image manipulation
- `scikit-image>=0.18.0` - Advanced image processing

## 🧭 How the Project Works

1. You load a blurred image via the GUI.
2. The image is resized to 512×512 and converted to a PyTorch tensor.
3. The pre-trained `MPRNet` model runs an inference pass and outputs a restored image.
4. The app computes cosine similarity between the original (blurred) and restored images as a simple numerical comparison.
5. The GUI displays both images side-by-side along with the cosine similarity score and explanatory math.

Key components:
- `models/MPRNet.py`: MPRNet network definition.
- `models/model_deblurring.pth`: Pre-trained weights (loaded at startup).
- `main.py`: Tkinter-based GUI to load images, run inference, and show results.

Notes:
- This build is inference-only (no training code or configs).
- GPU is auto-used if CUDA is available; otherwise falls back to CPU.

## ✨ Features

- **Multi-Stage Progressive Image Restoration** (MPRNet)
- **Real-time Processing** with optional GPU acceleration
- **Cosine Similarity Analysis** shown in the UI
- **Simple GUI**: Browse image or load sample by ID
- **Mathematical Concepts Viewer**: Short, friendly explanations

### Linear Algebra Applications
- **Matrix Convolution** (CNN kernels)
- **Vector Similarity** (cosine similarity)

## 🧠 Linear Algebra in This Project

- **Matrix Convolution (CNN kernels)**
  - What: Small matrices (kernels) slide over the image and compute weighted sums; stacking many learned kernels yields feature extraction and restoration.
  - Math: y[m,n] = Σ Σ x[i,j] · k[m-i, n-j]
  - Where in code: Implemented throughout `models/MPRNet.py` via the `conv(...)` helper and convolutional layers inside the model’s stages.

- **Cosine Similarity (vector-space comparison)**
  - What: Measures the angle between two vectors; close to 1 means highly similar direction.
  - Math: cos(θ) = (a · b) / (||a|| · ||b||)
  - Where in code: Computed in `main.py` in `calculate_cosine_similarity()` by flattening the original (blurred) and restored image tensors and using `torch.nn.functional.cosine_similarity`.

## 🎨 GUI Highlights
- Image selection (browse or numbered samples)
- Process button runs inference
- Two-pane display (blurred vs. restored)
- Cosine similarity shown beneath results
- Optional math explainer window

## 📈 Expected Performance (rough guides)
- CPU: ~2–5s per 512×512 image
- GPU: ~0.5–1s per 512×512 image
- Memory: ~2–4 GB RAM

## 📚 References

### Academic Papers
- **MPRNet**: "Multi-Stage Progressive Image Restoration" - CVPR 2021
- **Original Paper**: https://arxiv.org/abs/2102.02808

### Linear Algebra Resources
- **Convolution**: 2D convolution operations in deep learning
- **Matrix Operations**: Linear transformations and tensor operations
- **Vector Similarity**: Cosine similarity and distance metrics