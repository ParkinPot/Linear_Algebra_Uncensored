#!/usr/bin/env python3
"""
Linear Algebra - Image Utilities
Image processing and manipulation utilities
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
from typing import Tuple, Optional, Union


def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load and preprocess an image
    
    Args:
        image_path: Path to the image file
        target_size: Target size (width, height) for resizing
    
    Returns:
        Preprocessed image as numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if target size is specified
    if target_size:
        image = cv2.resize(image, target_size)
    
    return image


def image_to_tensor(image: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """
    Convert numpy image to PyTorch tensor
    
    Args:
        image: Input image as numpy array
        device: Device to place tensor on
    
    Returns:
        Image as PyTorch tensor
    """
    transform = transforms.ToTensor()
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor.to(device)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy image
    
    Args:
        tensor: Input tensor
    
    Returns:
        Image as numpy array
    """
    # Remove batch dimension and convert to numpy
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose channels
    image = tensor.permute(1, 2, 0).cpu().numpy()
    
    return image


def save_image(image: np.ndarray, save_path: str, quality: int = 95) -> bool:
    """
    Save image to file
    
    Args:
        image: Image as numpy array
        save_path: Path to save the image
        quality: JPEG quality (1-100)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Save image
        success = cv2.imwrite(save_path, image_bgr)
        return success
    
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def resize_image(image: np.ndarray, size: Tuple[int, int], 
                interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize image
    
    Args:
        image: Input image
        size: Target size (width, height)
        interpolation: Interpolation method
    
    Returns:
        Resized image
    """
    return cv2.resize(image, size, interpolation=interpolation)


def normalize_image(image: np.ndarray, mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                   std: Tuple[float, float, float] = (0.5, 0.5, 0.5)) -> np.ndarray:
    """
    Normalize image
    
    Args:
        image: Input image
        mean: Mean values for normalization
        std: Standard deviation values for normalization
    
    Returns:
        Normalized image
    """
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(mean)) / np.array(std)
    return image


def denormalize_image(image: np.ndarray, mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                     std: Tuple[float, float, float] = (0.5, 0.5, 0.5)) -> np.ndarray:
    """
    Denormalize image
    
    Args:
        image: Input normalized image
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
    
    Returns:
        Denormalized image
    """
    image = image * np.array(std) + np.array(mean)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def create_image_grid(images: list, grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Create a grid of images
    
    Args:
        images: List of images
        grid_size: Grid size (rows, cols). If None, auto-calculate
    
    Returns:
        Grid image
    """
    if not images:
        raise ValueError("No images provided")
    
    # Auto-calculate grid size if not provided
    if grid_size is None:
        num_images = len(images)
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        grid_size = (rows, cols)
    
    rows, cols = grid_size
    h, w = images[0].shape[:2]
    
    # Create grid
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
        
        row = i // cols
        col = i % cols
        
        # Ensure image has 3 channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Resize image to fit grid cell
        img_resized = cv2.resize(img, (w, h))
        
        # Place image in grid
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = img_resized
    
    return grid


def apply_blur(image: np.ndarray, kernel_size: int = 15, sigma: float = 0) -> np.ndarray:
    """
    Apply Gaussian blur to image
    
    Args:
        image: Input image
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def apply_noise(image: np.ndarray, noise_type: str = 'gaussian', 
               intensity: float = 0.1) -> np.ndarray:
    """
    Apply noise to image
    
    Args:
        image: Input image
        noise_type: Type of noise ('gaussian', 'salt_pepper')
        intensity: Noise intensity
    
    Returns:
        Noisy image
    """
    if noise_type == 'gaussian':
        noise = np.random.normal(0, intensity * 255, image.shape)
        noisy_image = image + noise
    elif noise_type == 'salt_pepper':
        noisy_image = image.copy()
        salt_pepper = np.random.random(image.shape[:2])
        noisy_image[salt_pepper < intensity/2] = 0
        noisy_image[salt_pepper > 1 - intensity/2] = 255
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def get_image_info(image: np.ndarray) -> dict:
    """
    Get image information
    
    Args:
        image: Input image
    
    Returns:
        Dictionary with image information
    """
    info = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min_value': float(np.min(image)),
        'max_value': float(np.max(image)),
        'mean_value': float(np.mean(image)),
        'std_value': float(np.std(image))
    }
    
    if len(image.shape) == 3:
        info['channels'] = image.shape[2]
    else:
        info['channels'] = 1
    
    return info
