import torch

def check_tensor(tensor, name="tensor", print_stats=False):
    """Check if tensor is on GPU and print basic stats."""
    if isinstance(tensor, (list, tuple)):
        print(f"{name} is a {type(tensor)} of length {len(tensor)}")
        for i, t in enumerate(tensor):
            check_tensor(t, f"{name}[{i}]", print_stats)
        return
        
    if not isinstance(tensor, torch.Tensor):
        print(f"{name} is not a tensor, but a {type(tensor)}")
        return
        
    print(f"\n=== {name} ===")
    print(f"Shape: {tensor.shape}")
    print(f"Device: {tensor.device}")
    print(f"Requires grad: {tensor.requires_grad}")
    
    if print_stats and tensor.numel() > 0:
        with torch.no_grad():
            print(f"Mean: {tensor.mean().item():.4f}")
            print(f"Std: {tensor.std().item():.4f}")
            print(f"Min: {tensor.min().item():.4f}")
            print(f"Max: {tensor.max().item():.4f}")

def check_gradients(model, name="model"):
    """Check if model parameters have gradients and are on GPU."""
    print(f"\n=== Checking gradients for {name} ===")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}:")
            print(f"  Shape: {param.grad.shape}")
            print(f"  Device: {param.grad.device}")
            print(f"  Grad mean: {param.grad.mean().item():.4e}")
            print(f"  Grad std: {param.grad.std().item():.4e}")
        else:
            print(f"{name} has no gradient") 