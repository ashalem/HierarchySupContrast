import torch

def check_tensor(tensor, name="Tensor"):
    """
    Checks a tensor for NaN and Inf values and prints warnings if any are found.

    Args:
        tensor (torch.Tensor): The tensor to check.
        name (str): A descriptive name for the tensor (used in print statements).
    """
    if torch.isnan(tensor).any():
        print(f"WARNING: NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"WARNING: Inf detected in {name}")

def check_gradients(model):
    """
    Checks gradients of all parameters in the model for NaN and Inf values.

    Args:
        model (torch.nn.Module): The model to check.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"WARNING: NaN detected in gradients of {name}")
            if torch.isinf(param.grad).any():
                print(f"WARNING: Inf detected in gradients of {name}") 