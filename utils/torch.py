import torch


def detect_device(verbose=True) -> str:
    """auto-detect what device to use for torch"""
    device_name = "cpu"
    # Check if CUDA is available
    if torch.cuda.is_available():
        device_name = 'cuda'
        if verbose: print('CUDA is available device:', torch.cuda.get_device_name())
    elif torch.backends.mps.is_available():
        device_name = 'mps'
        if verbose: print('mps is available device')
    else:
        print('Using CPU device')

    return device_name
