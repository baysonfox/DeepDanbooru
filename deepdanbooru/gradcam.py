import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# This file contains test/example code for Grad-CAM functionality
# Updated for PyTorch implementation

def grad_cam_test(model, x, target):
    """Test function for gradient computation with PyTorch"""
    x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)
    
    # Forward pass
    output = model(x)
    
    # Compute loss for target
    loss = torch.sum(output * target)
    
    # Backward pass to get gradients
    loss.backward()
    
    return x.grad.detach().numpy()


def run_test():
    """Test the gradient computation functionality"""
    # Generate sample model
    model = nn.Sequential(
        nn.Linear(2, 2)
    )
    
    target = np.array([[1.0, 2.0]], dtype=np.float32)

    # Calculate gradient using numpy array
    input_numpy = np.array([[0.0, 0.0]])
    grad_output_numpy = grad_cam_test(model, input_numpy, target)
    print(f"PyTorch gradients: {grad_output_numpy}")


if __name__ == "__main__":
    run_test()
