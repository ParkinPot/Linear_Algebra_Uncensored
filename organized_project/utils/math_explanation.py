#!/usr/bin/env python3
"""
Linear Algebra - Mathematical Concepts Explanation
Detailed explanations of mathematical concepts used in deep learning
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import math


class LinearAlgebraExplainer:
    """
    Class to explain linear algebra concepts in deep learning
    """
    
    def __init__(self):
        self.concepts = {}
        self.examples = {}
    
    def explain_vector_operations(self) -> Dict[str, str]:
        """
        Explain vector operations used in neural networks
        """
        explanations = {
            "Dot Product": {
                "formula": "a · b = Σ(a_i * b_i) = ||a|| * ||b|| * cos(θ)",
                "explanation": "Measures similarity between two vectors. Used in attention mechanisms and similarity calculations.",
                "example": "If a = [1, 2, 3] and b = [4, 5, 6], then a·b = 1×4 + 2×5 + 3×6 = 32",
                "neural_network_use": "Used in fully connected layers, attention mechanisms, and cosine similarity"
            },
            
            "Cross Product": {
                "formula": "a × b = [a₂b₃ - a₃b₂, a₃b₁ - a₁b₃, a₁b₂ - a₂b₁]",
                "explanation": "Creates a vector perpendicular to both input vectors. Magnitude equals area of parallelogram.",
                "example": "If a = [1, 0, 0] and b = [0, 1, 0], then a×b = [0, 0, 1]",
                "neural_network_use": "Less common in standard neural networks, used in 3D computer vision"
            },
            
            "Vector Norm": {
                "formula": "||v||_p = (Σ|v_i|^p)^(1/p)",
                "explanation": "Measures the magnitude (length) of a vector. L2 norm is most common.",
                "example": "If v = [3, 4], then ||v||₂ = √(3² + 4²) = 5",
                "neural_network_use": "Used in regularization, normalization, and distance calculations"
            },
            
            "Cosine Similarity": {
                "formula": "cos(θ) = (a·b) / (||a|| * ||b||)",
                "explanation": "Measures the angle between two vectors, independent of magnitude.",
                "example": "Range: [-1, 1]. 1 = identical direction, 0 = orthogonal, -1 = opposite direction",
                "neural_network_use": "Used in loss functions, similarity metrics, and attention mechanisms"
            }
        }
        return explanations
    
    def explain_matrix_operations(self) -> Dict[str, str]:
        """
        Explain matrix operations used in neural networks
        """
        explanations = {
            "Matrix Multiplication": {
                "formula": "C = AB where C_ij = Σ(A_ik * B_kj)",
                "explanation": "Fundamental operation in neural networks. Each output element is a dot product of input row and weight column.",
                "example": "If A is 2×3 and B is 3×2, then C is 2×2",
                "neural_network_use": "Used in fully connected layers, attention mechanisms, and linear transformations"
            },
            
            "Convolution": {
                "formula": "y[m,n] = Σ Σ x[i,j] * k[m-i, n-j]",
                "explanation": "Sliding window operation that applies a kernel to extract features from images.",
                "example": "3×3 kernel slides over image, computing dot product at each position",
                "neural_network_use": "Core operation in CNNs for feature extraction from images"
            },
            
            "Transpose": {
                "formula": "A^T where A^T_ij = A_ji",
                "explanation": "Flips matrix along diagonal. Essential for backpropagation.",
                "example": "If A = [[1,2], [3,4]], then A^T = [[1,3], [2,4]]",
                "neural_network_use": "Used in backpropagation to compute gradients correctly"
            },
            
            "Matrix Inverse": {
                "formula": "AA^(-1) = I (identity matrix)",
                "explanation": "Matrix that when multiplied by original gives identity. Used in linear systems.",
                "example": "For 2×2 matrix A = [[a,b], [c,d]], inverse involves determinant",
                "neural_network_use": "Used in optimization algorithms and analytical solutions"
            }
        }
        return explanations
    
    def explain_tensor_operations(self) -> Dict[str, str]:
        """
        Explain tensor operations used in deep learning
        """
        explanations = {
            "Tensor Reshaping": {
                "formula": "Reshape tensor from shape A to shape B where prod(A) = prod(B)",
                "explanation": "Changes tensor dimensions without changing data. Essential for connecting different layer types.",
                "example": "Flatten 28×28×1 image to 784×1 vector for fully connected layer",
                "neural_network_use": "Used to connect convolutional and fully connected layers"
            },
            
            "Broadcasting": {
                "formula": "Automatic expansion of dimensions for element-wise operations",
                "explanation": "Allows operations between tensors of different shapes by automatically expanding smaller tensor.",
                "example": "Adding 32×1×1 bias to 32×28×28 feature map",
                "neural_network_use": "Used in batch normalization, bias addition, and attention mechanisms"
            },
            
            "Batch Processing": {
                "formula": "Process multiple samples simultaneously using vectorized operations",
                "explanation": "Enables efficient parallel processing by stacking samples into batches.",
                "example": "Process 32 images simultaneously instead of one at a time",
                "neural_network_use": "Fundamental for efficient training and inference"
            },
            
            "Gradient Computation": {
                "formula": "∂L/∂θ using automatic differentiation (backpropagation)",
                "explanation": "Computes derivatives of loss with respect to parameters using chain rule.",
                "example": "For y = f(g(x)), ∂y/∂x = ∂y/∂g * ∂g/∂x",
                "neural_network_use": "Essential for training - updates parameters to minimize loss"
            }
        }
        return explanations
    
    def explain_optimization_math(self) -> Dict[str, str]:
        """
        Explain optimization algorithms and their mathematical foundations
        """
        explanations = {
            "Gradient Descent": {
                "formula": "θ_{t+1} = θ_t - α∇L(θ_t)",
                "explanation": "Updates parameters in direction opposite to gradient (steepest descent).",
                "example": "If gradient points uphill, move downhill to minimize loss",
                "mathematical_intuition": "Uses first-order Taylor approximation: L(θ+Δθ) ≈ L(θ) + ∇L(θ)^T Δθ"
            },
            
            "Momentum": {
                "formula": "v_t = βv_{t-1} + α∇L(θ_t), θ_{t+1} = θ_t - v_t",
                "explanation": "Accumulates gradient information over time to smooth parameter updates.",
                "example": "Like a ball rolling down a hill - builds up speed in consistent directions",
                "mathematical_intuition": "Reduces oscillations and helps escape local minima"
            },
            
            "Adam Optimizer": {
                "formula": "m_t = β₁m_{t-1} + (1-β₁)g_t, v_t = β₂v_{t-1} + (1-β₂)g_t², θ_t = θ_{t-1} - α*m_t/√(v_t + ε)",
                "explanation": "Combines momentum (first moment) with adaptive learning rates (second moment).",
                "example": "Adapts learning rate per parameter based on gradient history",
                "mathematical_intuition": "Uses exponential moving averages for robust parameter updates"
            },
            
            "Learning Rate Scheduling": {
                "formula": "α_t = α₀ * decay_function(t)",
                "explanation": "Reduces learning rate over time to fine-tune convergence.",
                "example": "Start with large steps, gradually make smaller adjustments",
                "mathematical_intuition": "Balances exploration (large LR) with exploitation (small LR)"
            }
        }
        return explanations
    
    def explain_activation_functions(self) -> Dict[str, str]:
        """
        Explain activation functions and their mathematical properties
        """
        explanations = {
            "ReLU (Rectified Linear Unit)": {
                "formula": "f(x) = max(0, x)",
                "explanation": "Sets negative values to zero, keeps positive values unchanged.",
                "derivative": "f'(x) = 1 if x > 0, 0 if x ≤ 0",
                "mathematical_properties": "Non-linear, non-smooth at x=0, helps with vanishing gradient problem"
            },
            
            "Sigmoid": {
                "formula": "f(x) = 1 / (1 + e^(-x))",
                "explanation": "Squashes values to range [0, 1]. Smooth, differentiable everywhere.",
                "derivative": "f'(x) = f(x)(1 - f(x))",
                "mathematical_properties": "S-shaped curve, can suffer from vanishing gradients"
            },
            
            "Tanh": {
                "formula": "f(x) = (e^x - e^(-x)) / (e^x + e^(-x))",
                "explanation": "Squashes values to range [-1, 1]. Zero-centered version of sigmoid.",
                "derivative": "f'(x) = 1 - f(x)²",
                "mathematical_properties": "Better gradient flow than sigmoid due to zero-centering"
            },
            
            "Softmax": {
                "formula": "f(x_i) = e^(x_i) / Σ(e^(x_j))",
                "explanation": "Converts vector to probability distribution. Sum of outputs equals 1.",
                "derivative": "Complex, involves Jacobian matrix",
                "mathematical_properties": "Used in classification, outputs probabilities"
            }
        }
        return explanations
    
    def explain_loss_functions(self) -> Dict[str, str]:
        """
        Explain loss functions and their mathematical foundations
        """
        explanations = {
            "Mean Squared Error (MSE)": {
                "formula": "L = (1/n) * Σ(y_true - y_pred)²",
                "explanation": "Measures average squared difference between true and predicted values.",
                "mathematical_properties": "Differentiable, penalizes large errors more than small ones",
                "vector_form": "L = ||y_true - y_pred||²₂ / n (L2 norm)"
            },
            
            "Mean Absolute Error (MAE)": {
                "formula": "L = (1/n) * Σ|y_true - y_pred|",
                "explanation": "Measures average absolute difference. Less sensitive to outliers than MSE.",
                "mathematical_properties": "Non-differentiable at zero, more robust to outliers",
                "vector_form": "L = ||y_true - y_pred||₁ / n (L1 norm)"
            },
            
            "Cross-Entropy Loss": {
                "formula": "L = -Σ(y_true * log(y_pred))",
                "explanation": "Measures difference between true and predicted probability distributions.",
                "mathematical_properties": "Penalizes confident wrong predictions heavily",
                "vector_form": "L = -⟨y_true, log(y_pred)⟩ (dot product)"
            },
            
            "Huber Loss": {
                "formula": "L = {0.5*(y_true - y_pred)² if |error| ≤ δ, δ*(|error| - 0.5*δ) otherwise}",
                "explanation": "Combines MSE and MAE. Quadratic for small errors, linear for large errors.",
                "mathematical_properties": "Robust to outliers, differentiable everywhere",
                "vector_form": "Hybrid of L1 and L2 norms"
            }
        }
        return explanations
    
    def create_mathematical_visualizations(self):
        """
        Create visualizations of mathematical concepts
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Vector operations
        ax1 = axes[0, 0]
        vectors = np.array([[1, 2], [3, 1]])
        ax1.quiver([0, 0], [0, 0], vectors[:, 0], vectors[:, 1], angles='xy', scale_units='xy', scale=1)
        ax1.set_xlim(-1, 4)
        ax1.set_ylim(-1, 3)
        ax1.set_title('Vector Operations\n(Dot Product, Cross Product)')
        ax1.grid(True)
        ax1.set_aspect('equal')
        
        # 2. Matrix multiplication
        ax2 = axes[0, 1]
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[2, 0], [1, 3]])
        C = A @ B
        im = ax2.imshow(C, cmap='viridis')
        ax2.set_title('Matrix Multiplication\nA @ B = C')
        plt.colorbar(im, ax=ax2)
        
        # 3. Activation functions
        ax3 = axes[0, 2]
        x = np.linspace(-5, 5, 100)
        ax3.plot(x, np.maximum(0, x), label='ReLU')
        ax3.plot(x, 1/(1+np.exp(-x)), label='Sigmoid')
        ax3.plot(x, np.tanh(x), label='Tanh')
        ax3.set_title('Activation Functions')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Loss functions
        ax4 = axes[1, 0]
        x = np.linspace(-3, 3, 100)
        ax4.plot(x, x**2, label='MSE')
        ax4.plot(x, np.abs(x), label='MAE')
        ax4.plot(x, np.where(np.abs(x) <= 1, 0.5*x**2, np.abs(x)-0.5), label='Huber')
        ax4.set_title('Loss Functions')
        ax4.legend()
        ax4.grid(True)
        
        # 5. Gradient descent
        ax5 = axes[1, 1]
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2
        contour = ax5.contour(X, Y, Z, levels=20)
        ax5.clabel(contour, inline=True, fontsize=8)
        ax5.set_title('Gradient Descent\n(Minimizing x² + y²)')
        
        # 6. Convolution kernel
        ax6 = axes[1, 2]
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        im = ax6.imshow(kernel, cmap='RdBu', vmin=-2, vmax=2)
        ax6.set_title('Convolution Kernel\n(Edge Detection)')
        plt.colorbar(im, ax=ax6)
        
        plt.tight_layout()
        return fig
    
    def explain_linear_algebra_in_cnn(self) -> Dict[str, str]:
        """
        Explain how linear algebra is used in CNN operations
        """
        explanations = {
            "Convolution as Matrix Multiplication": {
                "concept": "Convolution can be expressed as matrix multiplication using Toeplitz matrices",
                "formula": "y = Conv2D(x, k) = reshape(Toeplitz(x) @ flatten(k))",
                "explanation": "Each convolution operation is equivalent to multiplying input with a structured weight matrix"
            },
            
            "Pooling as Downsampling": {
                "concept": "Pooling reduces spatial dimensions while preserving important information",
                "formula": "y[i,j] = max(x[2i:2i+2, 2j:2j+2]) for max pooling",
                "explanation": "Selects maximum (or average) value from local neighborhood"
            },
            
            "Batch Normalization": {
                "concept": "Normalizes inputs using mean and variance across batch dimension",
                "formula": "y = γ * (x - μ) / √(σ² + ε) + β",
                "explanation": "Ensures inputs have zero mean and unit variance for stable training"
            },
            
            "Attention Mechanism": {
                "concept": "Computes weighted combination of input features",
                "formula": "Attention(Q,K,V) = softmax(QK^T/√d_k)V",
                "explanation": "Uses matrix multiplication to compute attention weights and apply them to values"
            }
        }
        return explanations
    
    def generate_training_mathematical_summary(self, model, optimizer, loss_fn) -> str:
        """
        Generate a mathematical summary of the training setup
        """
        summary = []
        summary.append("MATHEMATICAL TRAINING SUMMARY")
        summary.append("=" * 50)
        
        # Model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary.append(f"\nMODEL ARCHITECTURE:")
        summary.append(f"- Total Parameters: {total_params:,}")
        summary.append(f"- Trainable Parameters: {trainable_params:,}")
        summary.append(f"- Model Size: {total_params * 4 / 1024**2:.2f} MB (assuming float32)")
        
        # Optimizer information
        summary.append(f"\nOPTIMIZER MATHEMATICS:")
        summary.append(f"- Type: {type(optimizer).__name__}")
        if hasattr(optimizer, 'param_groups'):
            lr = optimizer.param_groups[0]['lr']
            summary.append(f"- Learning Rate: {lr}")
            if 'betas' in optimizer.param_groups[0]:
                betas = optimizer.param_groups[0]['betas']
                summary.append(f"- Beta parameters: {betas}")
        
        # Loss function
        summary.append(f"\nLOSS FUNCTION:")
        summary.append(f"- Type: {type(loss_fn).__name__}")
        
        # Mathematical concepts
        summary.append(f"\nLINEAR ALGEBRA CONCEPTS:")
        summary.append("- Matrix multiplication in fully connected layers")
        summary.append("- Convolution operations for feature extraction")
        summary.append("- Gradient computation via backpropagation")
        summary.append("- Parameter updates using optimization algorithms")
        summary.append("- Vector operations for batch processing")
        
        return "\n".join(summary)


def demonstrate_mathematical_concepts():
    """
    Demonstrate mathematical concepts with examples
    """
    explainer = LinearAlgebraExplainer()
    
    print("LINEAR ALGEBRA CONCEPTS IN DEEP LEARNING")
    print("=" * 60)
    
    # Vector operations
    print("\n1. VECTOR OPERATIONS:")
    vector_concepts = explainer.explain_vector_operations()
    for concept, info in vector_concepts.items():
        print(f"\n{concept}:")
        print(f"  Formula: {info['formula']}")
        print(f"  Explanation: {info['explanation']}")
        print(f"  Neural Network Use: {info['neural_network_use']}")
    
    # Matrix operations
    print("\n\n2. MATRIX OPERATIONS:")
    matrix_concepts = explainer.explain_matrix_operations()
    for concept, info in matrix_concepts.items():
        print(f"\n{concept}:")
        print(f"  Formula: {info['formula']}")
        print(f"  Explanation: {info['explanation']}")
        print(f"  Neural Network Use: {info['neural_network_use']}")
    
    # Optimization
    print("\n\n3. OPTIMIZATION MATHEMATICS:")
    opt_concepts = explainer.explain_optimization_math()
    for concept, info in opt_concepts.items():
        print(f"\n{concept}:")
        print(f"  Formula: {info['formula']}")
        print(f"  Explanation: {info['explanation']}")
        print(f"  Mathematical Intuition: {info['mathematical_intuition']}")
    
    return explainer


if __name__ == "__main__":
    explainer = demonstrate_mathematical_concepts()
    
    # Create visualizations
    print("\n\nCreating mathematical visualizations...")
    fig = explainer.create_mathematical_visualizations()
    plt.show()
