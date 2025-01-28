import torch
import torch.nn as nn
import time
from fvcore.nn import FlopCountAnalysis, flop_count_table
from typing import Union, Tuple

def setup_model_device(
    model: nn.Module,
    device: str = 'cuda'
) -> nn.Module:
    """Prepares model for training by moving to device and setting up DataParallel if needed.
    
    Args:
        model: Neural network model
        device: Device to move model to ('cuda' or 'cpu')
    
    Returns:
        nn.Module: Prepared model
    """
    if device == 'cuda':
        model = model.cuda()
        print('Number of available CUDA GPUs:', torch.cuda.device_count())
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            
    return model

def measure_latency(
    model: nn.Module,
    device: str = 'cpu',
    iterations: int = 1000,
    input_shape: Tuple[int, ...] = (4, 3, 512, 1024)
) -> Tuple[float, float, float, float]:
    """Measures model latency and FPS statistics.
    
    Args:
        model: Model to evaluate
        device: Device to run measurements on
        iterations: Number of iterations for measurement
        input_shape: Shape of input tensor for measurement
    
    Returns:
        Tuple containing mean latency, std latency, mean FPS, std FPS
    """
    latencies = []
    fps_values = []

    for _ in range(iterations):
        image = torch.randn(input_shape).to(device)
        start = time.time()
        with torch.no_grad():
            _ = model(image)
        end = time.time()

        latency = end - start
        latencies.append(latency)
        fps_values.append(1 / latency)

    mean_latency = (torch.tensor(latencies).mean() * 1000)  # Convert to ms
    std_latency = (torch.tensor(latencies).std() * 1000)
    mean_fps = torch.tensor(fps_values).mean()
    std_fps = torch.tensor(fps_values).std()

    print(f"Mean Latency: {mean_latency:.2f} ms ± {std_latency:.2f} ms")
    print(f"Mean FPS: {mean_fps:.2f} ± {std_fps:.2f}")
    
    return mean_latency, std_latency, mean_fps, std_fps

def calculate_flops(
    model: nn.Module,
    device: str = 'cpu',
    input_shape: Tuple[int, ...] = (4, 3, 512, 1024)
) -> None:
    """Calculates and prints model FLOPs.
    
    Args:
        model: Model to analyze
        device: Device to run analysis on
        input_shape: Shape of input tensor for analysis
    """
    image = torch.zeros(input_shape).to(device)
    flops = FlopCountAnalysis(model, image)
    print(flop_count_table(flops))

def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a model.
    
    Args:
        model: Model to analyze
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
import torch

class IntRangeTransformer:
    """
    Transformer class for clamping tensor values to a specified integer range.
    Commonly used in semantic segmentation for label preprocessing.
    """
    def __init__(self, min_val: int = 0, max_val: int = 255):
        """
        Initialize the transformer with range bounds.
        
        Args:
            min_val: Minimum value in the range (inclusive)
            max_val: Maximum value in the range (inclusive)
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Apply the transformation to the input tensor.
        
        Args:
            sample: Input tensor to transform
            
        Returns:
            torch.Tensor: Transformed tensor with values clamped to [min_val, max_val]
                         and converted to torch.long type
        """
        sample = torch.clamp(sample, self.min_val, self.max_val)
        return sample.long()