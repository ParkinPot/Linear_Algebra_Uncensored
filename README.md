# Linear Algebra – Image Deblurring (Inference-Only)

This project demonstrates modern image deblurring using a pre-trained **MPRNet** model while illustrating key **linear algebra concepts** used in deep learning, including **matrix convolution** and **cosine similarity**.

The application provides both a **graphical interface** and a **command-line interface** for processing blurred images and visualizing restoration results.

---

# Project Structure

```
organized_project/
├── main.py                 # Main GUI application
├── config.py               # Configuration settings
├── requirements.txt        # Project dependencies

├── models/                 # Neural network models
│   ├── MPRNet.py           # MPRNet architecture implementation
│   └── model_deblurring.pth # Pre-trained model weights

├── datasets/               # Image datasets
│   └── blurred_images/     # Blurred test images (300 images)

├── scripts/                # Command-line scripts
│   ├── deblur_demo.py      # Enhanced CLI demo
│   └── demo.py             # Official MPRNet demo

├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── image_utils.py      # Image processing utilities
│   ├── math_utils.py       # Mathematical operations
│   └── model_utils.py      # Model utilities

├── results/                # Output directory
└── README.md               # Project documentation
```

---

# Overview

The system allows users to load a blurred image and restore it using a **pre-trained MPRNet model**.

Processing pipeline:

1. A blurred image is loaded through the GUI or CLI.
2. The image is resized to **512 × 512** and converted into a **PyTorch tensor**.
3. The **MPRNet model** performs an inference pass to produce a restored image.
4. The application computes **cosine similarity** between the blurred and restored images.
5. Results are displayed with a **side-by-side comparison and similarity score**.

This project is **inference-only** and does not include training code.

---

# Setup

## Prerequisites

* Python 3.8 or later
* CUDA-compatible GPU (optional but recommended)

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Key Dependencies

* `torch>=1.9.0` – PyTorch deep learning framework
* `torchvision>=0.10.0` – Computer vision utilities
* `opencv-python>=4.5.0` – Image processing
* `matplotlib>=3.4.0` – Visualization
* `numpy>=1.21.0` – Numerical computation
* `Pillow>=8.3.0` – Image handling
* `scikit-image>=0.18.0` – Advanced image processing

---

# Running the Application

## GUI Application (Recommended)

Run the graphical interface:

```bash
python main.py
```

The GUI provides:

* Image browsing or sample image selection
* Real-time image processing
* Side-by-side comparison of blurred and restored images
* Cosine similarity score
* Explanations of mathematical concepts used in the model

---

## Command-Line Interface

Interactive mode:

```bash
python scripts/deblur_demo.py
```

Process a sample image:

```bash
python scripts/deblur_demo.py --sample 1
```

Process a custom image:

```bash
python scripts/deblur_demo.py --image path/to/image.jpg
```

Save results:

```bash
python scripts/deblur_demo.py --sample 1 --save results/output.png
```

---

## Official MPRNet Demo

You may also run the original demo script:

```bash
python scripts/demo.py --task Deblurring --input_dir datasets/blurred_images --result_dir results
```

---

# Features

* Multi-stage progressive image restoration using **MPRNet**
* Graphical user interface for interactive use
* Command-line interface for batch testing
* Automatic GPU acceleration when CUDA is available
* Side-by-side visualization of results
* Cosine similarity comparison between images
* Educational explanations of underlying mathematical concepts

---

# Linear Algebra Concepts Demonstrated

## Matrix Convolution

Convolution is the core operation used in convolutional neural networks.

Small matrices called **kernels** slide across the input image and compute weighted sums, allowing the network to detect patterns and restore image details.

Mathematically:

[
y[m,n] = \sum_i \sum_j x[i,j] \cdot k[m-i, n-j]
]

Where:

* (x) represents the input image
* (k) represents the convolution kernel
* (y) represents the output feature map

In this project, convolution operations are implemented throughout the model in:

```
models/MPRNet.py
```

using convolutional layers.

---

## Cosine Similarity

Cosine similarity measures the **angle between two vectors** in a vector space.

It is used here to compare the similarity between the **blurred image** and the **restored image**.

[
\cos(\theta) = \frac{a \cdot b}{|a| |b|}
]

Where:

* (a) and (b) are flattened image vectors
* (\cdot) denotes the dot product
* (|\cdot|) denotes vector magnitude

The similarity is calculated in:

```
main.py
```

by flattening the image tensors and applying:

```
torch.nn.functional.cosine_similarity
```

---

# GUI Components

The graphical interface includes:

* Image selection (file browser or sample ID)
* Processing button to run inference
* Side-by-side display of blurred and restored images
* Cosine similarity output
* Optional mathematical explanation panel

---

# Performance

Typical processing times for a **512 × 512** image:

| Hardware | Time per Image |
| -------- | -------------- |
| CPU      | 2 – 5 seconds  |
| GPU      | 0.5 – 1 second |

Estimated memory usage:

* 2 – 4 GB RAM

---

# References

## Research Paper

MPRNet: Multi-Stage Progressive Image Restoration
CVPR 2021

[https://arxiv.org/abs/2102.02808](https://arxiv.org/abs/2102.02808)

---

## Related Topics

* Convolutional Neural Networks
* Linear Algebra for Deep Learning
* Image Restoration Techniques
* Vector Similarity Metrics
