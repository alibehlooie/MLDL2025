import torch

def poly_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    init_lr: float,
    iter: int,
    lr_decay_iter: float = 1,
    max_iter: int = 300,
    power: float = 0.9
) -> float:
    """Implements polynomial decay of learning rate.
    
    The learning rate is updated according to the formula:
    lr = init_lr * (1 - iter/max_iter)^power
    
    Args:
        optimizer: The optimizer whose learning rate should be adjusted
        init_lr: Initial learning rate value
        iter: Current iteration number
        lr_decay_iter: How frequently decay occurs (default: 1)
        max_iter: Maximum number of iterations (default: 300)
        power: Polynomial power for decay (default: 0.9)
    
    Returns:
        float: The new learning rate
    """
    lr = init_lr * (1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr