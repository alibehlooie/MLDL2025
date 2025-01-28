import numpy as np
import pandas as pd
import sys
from typing import Dict, Any

def fast_hist(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Computes the confusion matrix for semantic segmentation.
    
    Args:
        a: Ground truth labels
        b: Predicted labels
        n: Number of classes
    
    Returns:
        np.ndarray: n x n confusion matrix
    """
    k = (a >= 0) & (a < n)
    return np.bincount(
        n * a[k].astype(int) + b[k],
        minlength=n ** 2
    ).reshape(n, n)

def per_class_iou(hist: np.ndarray) -> np.ndarray:
    """Computes per-class IoU (Intersection over Union) from confusion matrix.
    
    Args:
        hist: Confusion matrix
    
    Returns:
        np.ndarray: Array of IoU values for each class
    """
    epsilon = 1e-5
    return (np.diag(hist)) / (
        hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon
    )

def tabular_print(log_dict: Dict[str, Any]) -> None:
    """Prints dictionary data in a nicely formatted table.
    
    Args:
        log_dict: Dictionary containing data to print
    """
    df = pd.DataFrame({**log_dict}, index=[0])

    try:
        from prettytable import PrettyTable
    except ImportError:
        if not hasattr(tabular_print, 'warned'):
            print('PrettyTable is not available. Using DataFrame display.',
                  file=sys.stderr)
            tabular_print.warned = True
        print(df)
        return
    
    x = PrettyTable()
    for col in df.columns:
        x.add_column(col, df[col].values)
    print(x)