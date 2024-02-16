import torch


def get_torch_device(use_gpu: bool = True, debug: bool = False) -> torch.device:
    """Obtains a torch device in which to perform computations

    Args:
        use_gpu (bool, optional): Use GPU if available. Defaults to True.
        debug (bool, optional): Whether to print debug information or not. Defaults to False.

    Returns:
        torch.device: Device in which to perform computations
    """    
    device = torch.device(
        'cuda:0' if use_gpu and torch.cuda.is_available() else
        'mps' if use_gpu and torch.backends.mps.is_available() else
        'cpu'
    )
    if debug: print("Device selected:", device)
    return device