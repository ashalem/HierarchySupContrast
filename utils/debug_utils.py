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

def check_gradients(model, name="model", vanishing_threshold=1e-5, exploding_threshold=1e2):
    """Check if model parameters have gradients and are on GPU.
    Also checks for vanishing/exploding gradients.
    
    Args:
        model: The PyTorch model to check
        name: Name for logging
        vanishing_threshold: Gradient norm below this is considered vanishing
        exploding_threshold: Gradient norm above this is considered exploding
    """
    print(f"\n=== Checking gradients for {name} ===")
    has_vanishing = False
    has_exploding = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}:")
            print(f"  Shape: {param.grad.shape}")
            print(f"  Device: {param.grad.device}")
            print(f"  Grad norm: {grad_norm:.4e}")
            print(f"  Grad mean: {param.grad.mean().item():.4e}")
            print(f"  Grad std: {param.grad.std().item():.4e}")
            
            # Check for vanishing gradients
            if grad_norm < vanishing_threshold:
                has_vanishing = True
                print(f"  WARNING: Possibly vanishing gradient detected (norm: {grad_norm:.4e})")
            
            # Check for exploding gradients
            if grad_norm > exploding_threshold:
                has_exploding = True
                print(f"  WARNING: Possibly exploding gradient detected (norm: {grad_norm:.4e})")
                
            # Check for NaN or Inf
            if torch.isnan(param.grad).any():
                print(f"  WARNING: NaN detected in gradients")
            if torch.isinf(param.grad).any():
                print(f"  WARNING: Inf detected in gradients")
        else:
            print(f"{name} has no gradient")
    
    if has_vanishing:
        print("\nWARNING: Vanishing gradients detected in some parameters")
    if has_exploding:
        print("\nWARNING: Exploding gradients detected in some parameters") 