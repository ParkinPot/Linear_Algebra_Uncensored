#!/usr/bin/env python3
"""
Linear Algebra - Model Utilities
Model loading, saving, and utility functions
"""

import torch
import torch.nn as nn
import os
import sys
from typing import Dict, Any, Optional, Tuple
import json

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from MPRNet import MPRNet


def load_model(model_path: str, device: str = 'auto') -> nn.Module:
    """
    Load a pre-trained model
    
    Args:
        model_path: Path to model weights
        device: Device to load model on ('auto', 'cpu', 'cuda')
    
    Returns:
        Loaded model
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = MPRNet()
    
    # Load state dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def save_model(model: nn.Module, save_path: str, 
              additional_info: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save model to file
    
    Args:
        model: Model to save
        save_path: Path to save model
        additional_info: Additional information to save
    
    Returns:
        True if successful, False otherwise
    """
    try:
        save_dict = {
            'state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'model_config': getattr(model, 'config', {}),
        }
        
        if additional_info:
            save_dict.update(additional_info)
        
        torch.save(save_dict, save_path)
        return True
    
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: Model to count parameters
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_model_size(model: nn.Module) -> float:
    """
    Get model size in MB
    
    Args:
        model: Model to get size
    
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 512, 512)) -> str:
    """
    Generate model summary
    
    Args:
        model: Model to summarize
        input_size: Input size for summary
    
    Returns:
        Model summary string
    """
    summary_lines = []
    summary_lines.append(f"Model: {model.__class__.__name__}")
    summary_lines.append("=" * 50)
    
    # Parameter count
    param_counts = count_parameters(model)
    summary_lines.append(f"Total Parameters: {param_counts['total_parameters']:,}")
    summary_lines.append(f"Trainable Parameters: {param_counts['trainable_parameters']:,}")
    summary_lines.append(f"Non-trainable Parameters: {param_counts['non_trainable_parameters']:,}")
    
    # Model size
    model_size = get_model_size(model)
    summary_lines.append(f"Model Size: {model_size:.2f} MB")
    
    # Input/Output shapes
    summary_lines.append(f"Input Size: {input_size}")
    
    # Try to get output size
    try:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(input_size)
            output = model(dummy_input)
            if isinstance(output, (list, tuple)):
                output_sizes = [list(out.shape) for out in output]
                summary_lines.append(f"Output Sizes: {output_sizes}")
            else:
                summary_lines.append(f"Output Size: {list(output.shape)}")
    except Exception as e:
        summary_lines.append(f"Could not determine output size: {e}")
    
    return "\n".join(summary_lines)


def freeze_model(model: nn.Module, freeze: bool = True) -> nn.Module:
    """
    Freeze or unfreeze model parameters
    
    Args:
        model: Model to freeze/unfreeze
        freeze: Whether to freeze parameters
    
    Returns:
        Modified model
    """
    for param in model.parameters():
        param.requires_grad = not freeze
    
    return model


def unfreeze_model(model: nn.Module) -> nn.Module:
    """
    Unfreeze model parameters
    
    Args:
        model: Model to unfreeze
    
    Returns:
        Modified model
    """
    return freeze_model(model, freeze=False)


def get_layer_activations(model: nn.Module, input_tensor: torch.Tensor, 
                         layer_names: Optional[list] = None) -> Dict[str, torch.Tensor]:
    """
    Get activations from specific layers
    
    Args:
        model: Model to get activations from
        input_tensor: Input tensor
        layer_names: Names of layers to get activations from
    
    Returns:
        Dictionary of layer activations
    """
    activations = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if layer_names is None or name in layer_names:
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations


def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient norms for each parameter
    
    Args:
        model: Model to compute gradient norms
    
    Returns:
        Dictionary of parameter names and their gradient norms
    """
    gradient_norms = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradient_norms[name] = param.grad.norm().item()
        else:
            gradient_norms[name] = 0.0
    
    return gradient_norms


def check_model_compatibility(model: nn.Module, input_shape: Tuple[int, ...]) -> bool:
    """
    Check if model is compatible with input shape
    
    Args:
        model: Model to check
        input_shape: Input shape to test
    
    Returns:
        True if compatible, False otherwise
    """
    try:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(input_shape)
            _ = model(dummy_input)
        return True
    except Exception as e:
        print(f"Model compatibility check failed: {e}")
        return False


def export_model_to_onnx(model: nn.Module, input_shape: Tuple[int, ...], 
                        output_path: str) -> bool:
    """
    Export model to ONNX format
    
    Args:
        model: Model to export
        input_shape: Input shape for export
        output_path: Path to save ONNX file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        return True
    
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return False


def create_model_config(model: nn.Module) -> Dict[str, Any]:
    """
    Create model configuration dictionary
    
    Args:
        model: Model to create config for
    
    Returns:
        Model configuration
    """
    config = {
        'model_class': model.__class__.__name__,
        'parameters': count_parameters(model),
        'model_size_mb': get_model_size(model),
        'device': next(model.parameters()).device.type,
        'dtype': next(model.parameters()).dtype,
    }
    
    # Add model-specific config if available
    if hasattr(model, 'config'):
        config['model_config'] = model.config
    
    return config


def save_model_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save model configuration to JSON file
    
    Args:
        config: Model configuration
        config_path: Path to save config
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def load_model_config(config_path: str) -> Dict[str, Any]:
    """
    Load model configuration from JSON file
    
    Args:
        config_path: Path to config file
    
    Returns:
        Model configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}
