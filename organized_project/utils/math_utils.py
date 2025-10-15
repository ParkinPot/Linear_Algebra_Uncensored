#!/usr/bin/env python3
"""
Linear Algebra - Mathematical Utilities
Mathematical operations and linear algebra utilities
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List
import math


def cosine_similarity(vector1: torch.Tensor, vector2: torch.Tensor, 
                     dim: int = 0) -> torch.Tensor:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        vector1: First vector
        vector2: Second vector
        dim: Dimension along which to compute similarity
    
    Returns:
        Cosine similarity value
    """
    return F.cosine_similarity(vector1, vector2, dim=dim)


def euclidean_distance(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Euclidean distance between two vectors
    
    Args:
        vector1: First vector
        vector2: Second vector
    
    Returns:
        Euclidean distance
    """
    return torch.norm(vector1 - vector2, p=2)


def manhattan_distance(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Manhattan distance between two vectors
    
    Args:
        vector1: First vector
        vector2: Second vector
    
    Returns:
        Manhattan distance
    """
    return torch.norm(vector1 - vector2, p=1)


def dot_product(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    """
    Calculate dot product between two vectors
    
    Args:
        vector1: First vector
        vector2: Second vector
    
    Returns:
        Dot product
    """
    return torch.dot(vector1, vector2)


def vector_magnitude(vector: torch.Tensor) -> torch.Tensor:
    """
    Calculate magnitude (norm) of a vector
    
    Args:
        vector: Input vector
    
    Returns:
        Vector magnitude
    """
    return torch.norm(vector, p=2)


def normalize_vector(vector: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Normalize a vector to unit length
    
    Args:
        vector: Input vector
        dim: Dimension along which to normalize
    
    Returns:
        Normalized vector
    """
    return F.normalize(vector, p=2, dim=dim)


def matrix_multiplication(matrix1: torch.Tensor, matrix2: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication
    
    Args:
        matrix1: First matrix
        matrix2: Second matrix
    
    Returns:
        Result matrix
    """
    return torch.mm(matrix1, matrix2)


def convolution_2d(input_tensor: torch.Tensor, kernel: torch.Tensor, 
                  padding: Union[int, Tuple[int, int]] = 0,
                  stride: Union[int, Tuple[int, int]] = 1) -> torch.Tensor:
    """
    Perform 2D convolution
    
    Args:
        input_tensor: Input tensor
        kernel: Convolution kernel
        padding: Padding size
        stride: Stride size
    
    Returns:
        Convolved tensor
    """
    return F.conv2d(input_tensor, kernel, padding=padding, stride=stride)


def cross_correlation_2d(input_tensor: torch.Tensor, kernel: torch.Tensor,
                        padding: Union[int, Tuple[int, int]] = 0,
                        stride: Union[int, Tuple[int, int]] = 1) -> torch.Tensor:
    """
    Perform 2D cross-correlation
    
    Args:
        input_tensor: Input tensor
        kernel: Cross-correlation kernel
        padding: Padding size
        stride: Stride size
    
    Returns:
        Cross-correlated tensor
    """
    # Flip kernel for cross-correlation
    kernel_flipped = torch.flip(kernel, dims=[-2, -1])
    return F.conv2d(input_tensor, kernel_flipped, padding=padding, stride=stride)


def pooling_2d(input_tensor: torch.Tensor, kernel_size: Union[int, Tuple[int, int]],
               stride: Union[int, Tuple[int, int]] = None,
               padding: Union[int, Tuple[int, int]] = 0,
               pool_type: str = 'max') -> torch.Tensor:
    """
    Perform 2D pooling operation
    
    Args:
        input_tensor: Input tensor
        kernel_size: Pooling kernel size
        stride: Stride size
        padding: Padding size
        pool_type: Type of pooling ('max', 'avg')
    
    Returns:
        Pooled tensor
    """
    if stride is None:
        stride = kernel_size
    
    if pool_type == 'max':
        return F.max_pool2d(input_tensor, kernel_size, stride, padding)
    elif pool_type == 'avg':
        return F.avg_pool2d(input_tensor, kernel_size, stride, padding)
    else:
        raise ValueError(f"Unknown pooling type: {pool_type}")


def activation_function(x: torch.Tensor, activation: str = 'relu') -> torch.Tensor:
    """
    Apply activation function
    
    Args:
        x: Input tensor
        activation: Activation function type
    
    Returns:
        Activated tensor
    """
    if activation == 'relu':
        return F.relu(x)
    elif activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'tanh':
        return torch.tanh(x)
    elif activation == 'leaky_relu':
        return F.leaky_relu(x)
    elif activation == 'gelu':
        return F.gelu(x)
    else:
        raise ValueError(f"Unknown activation function: {activation}")


def compute_gradient(tensor: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
    """
    Compute gradient of a tensor
    
    Args:
        tensor: Input tensor
        create_graph: Whether to create computation graph
    
    Returns:
        Gradient tensor
    """
    if not tensor.requires_grad:
        tensor.requires_grad_(True)
    
    gradient = torch.autograd.grad(
        outputs=tensor.sum(),
        inputs=tensor,
        create_graph=create_graph,
        retain_graph=True
    )[0]
    
    return gradient


def jacobian_matrix(f: callable, x: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian matrix
    
    Args:
        f: Function
        x: Input tensor
    
    Returns:
        Jacobian matrix
    """
    batch_size = x.shape[0]
    output = f(x)
    output_dim = output.shape[1] if len(output.shape) > 1 else 1
    
    jacobian = torch.zeros(batch_size, output_dim, x.shape[1])
    
    for i in range(output_dim):
        if output_dim == 1:
            grad_output = torch.ones_like(output)
        else:
            grad_output = torch.zeros_like(output)
            grad_output[:, i] = 1
        
        grad = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=grad_output,
            retain_graph=True,
            create_graph=True
        )[0]
        
        jacobian[:, i, :] = grad
    
    return jacobian


def eigenvalue_decomposition(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute eigenvalue decomposition
    
    Args:
        matrix: Input matrix
    
    Returns:
        Eigenvalues and eigenvectors
    """
    eigenvalues, eigenvectors = torch.linalg.eig(matrix)
    return eigenvalues, eigenvectors


def singular_value_decomposition(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute singular value decomposition
    
    Args:
        matrix: Input matrix
    
    Returns:
        U, S, V matrices
    """
    U, S, V = torch.linalg.svd(matrix)
    return U, S, V


def matrix_rank(matrix: torch.Tensor) -> int:
    """
    Compute matrix rank
    
    Args:
        matrix: Input matrix
    
    Returns:
        Matrix rank
    """
    return torch.linalg.matrix_rank(matrix)


def matrix_determinant(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute matrix determinant
    
    Args:
        matrix: Input matrix
    
    Returns:
        Matrix determinant
    """
    return torch.linalg.det(matrix)


def matrix_trace(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute matrix trace
    
    Args:
        matrix: Input matrix
    
    Returns:
        Matrix trace
    """
    return torch.trace(matrix)


def frobenius_norm(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute Frobenius norm of a matrix
    
    Args:
        matrix: Input matrix
    
    Returns:
        Frobenius norm
    """
    return torch.norm(matrix, p='fro')


def condition_number(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute condition number of a matrix
    
    Args:
        matrix: Input matrix
    
    Returns:
        Condition number
    """
    U, S, V = torch.linalg.svd(matrix)
    return S[0] / S[-1]


def linear_interpolation(x: torch.Tensor, x1: float, x2: float, 
                        y1: float, y2: float) -> torch.Tensor:
    """
    Perform linear interpolation
    
    Args:
        x: Input values
        x1: First x coordinate
        x2: Second x coordinate
        y1: First y coordinate
        y2: Second y coordinate
    
    Returns:
        Interpolated values
    """
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def bilinear_interpolation(grid: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Perform bilinear interpolation
    
    Args:
        grid: 2D grid
        x: X coordinates
        y: Y coordinates
    
    Returns:
        Interpolated values
    """
    return F.grid_sample(grid.unsqueeze(0).unsqueeze(0), 
                        torch.stack([x, y], dim=-1).unsqueeze(0),
                        mode='bilinear', padding_mode='border', align_corners=False)
