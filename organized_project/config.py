#!/usr/bin/env python3
"""
Linear Algebra - Configuration File
Configuration settings for the Image Deblurring Application
"""

import os

# Application Settings
APP_NAME = "Linear Algebra - Image Deblurring"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Linear Algebra Project"

# Model Configuration
MODEL_CONFIG = {
    'name': 'MPRNet',
    'weights_file': 'model_deblurring.pth',
    'input_size': (512, 512),
    'channels': 3,
    'device': 'auto'  # 'auto', 'cpu', 'cuda'
}

# Image Processing Settings
IMAGE_CONFIG = {
    'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
    'default_size': (512, 512),
    'color_mode': 'RGB',
    'normalize_range': (0, 1)
}

# GUI Configuration
GUI_CONFIG = {
    'window_size': (1200, 800),
    'theme_colors': {
        'primary': '#2c3e50',
        'secondary': '#3498db',
        'success': '#27ae60',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'info': '#9b59b6',
        'light': '#ecf0f1',
        'dark': '#34495e',
        'background': '#f0f0f0'
    },
    'font_family': 'Arial',
    'title_font_size': 20,
    'subtitle_font_size': 12,
    'button_font_size': 10
}

# File Paths
PATHS = {
    'models_dir': 'models',
    'datasets_dir': 'datasets',
    'blurred_images_dir': 'datasets/blurred_images',
    'results_dir': 'results',
    'scripts_dir': 'scripts',
    'utils_dir': 'utils'
}

# Linear Algebra Settings
LINEAR_ALGEBRA_CONFIG = {
    'similarity_metrics': ['cosine', 'euclidean', 'manhattan'],
    'default_metric': 'cosine',
    'matrix_operations': ['convolution', 'pooling', 'activation'],
    'vector_operations': ['dot_product', 'cross_product', 'magnitude']
}

# Performance Settings
PERFORMANCE_CONFIG = {
    'batch_size': 1,
    'num_workers': 0,
    'pin_memory': True,
    'non_blocking': True,
    'mixed_precision': False
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'linear_algebra.log'
}

def get_model_path():
    """Get the full path to the model weights file"""
    return os.path.join(PATHS['models_dir'], MODEL_CONFIG['weights_file'])

def get_dataset_path():
    """Get the full path to the datasets directory"""
    return PATHS['datasets_dir']

def get_blurred_images_path():
    """Get the full path to the blurred images directory"""
    return PATHS['blurred_images_dir']

def get_results_path():
    """Get the full path to the results directory"""
    return PATHS['results_dir']

def ensure_directories():
    """Ensure all required directories exist"""
    for path in PATHS.values():
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

def get_device():
    """Get the appropriate device for computation"""
    import torch
    
    if MODEL_CONFIG['device'] == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        return MODEL_CONFIG['device']

def validate_config():
    """Validate the configuration"""
    errors = []
    
    # Check if model file exists
    model_path = get_model_path()
    if not os.path.exists(model_path):
        errors.append(f"Model weights file not found: {model_path}")
    
    # Check if datasets directory exists
    datasets_path = get_dataset_path()
    if not os.path.exists(datasets_path):
        errors.append(f"Datasets directory not found: {datasets_path}")
    
    return errors
