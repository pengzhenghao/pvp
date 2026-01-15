"""
LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning.

LoRA freezes the pretrained model weights and injects trainable low-rank 
decomposition matrices into each layer of the network. This significantly 
reduces the number of trainable parameters for downstream tasks.

Reference: https://arxiv.org/abs/2106.09685
"""

import torch
import torch.nn as nn
from typing import Optional, List, Set


class LoRALinear(nn.Module):
    """
    A linear layer with LoRA (Low-Rank Adaptation) support.
    
    Instead of modifying the original weights W directly, LoRA adds a low-rank
    decomposition: W' = W + (alpha/rank) * A @ B
    
    Where:
        - W: original frozen weights (in_features x out_features)
        - A: trainable matrix (in_features x rank)
        - B: trainable matrix (rank x out_features)
        - alpha: scaling factor for LoRA
        - rank: the rank of the decomposition (r << min(in_features, out_features))
    
    Args:
        original_linear: The original nn.Linear layer to wrap
        rank: The rank of the LoRA decomposition (default: 4)
        alpha: Scaling factor for LoRA (default: 1.0)
        dropout: Dropout probability for LoRA layers (default: 0.0)
    """
    
    def __init__(
        self, 
        original_linear: nn.Linear, 
        rank: int = 4, 
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # Get the device of the original linear layer
        device = original_linear.weight.device
        
        # LoRA matrices
        # A is initialized with Kaiming uniform (good for ReLU activations)
        # B is initialized with zeros so initial output = original output
        self.lora_A = nn.Parameter(torch.empty(in_features, rank, device=device))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device))
        
        # Initialize A with small random values
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        
        # Optional dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: original_output + scaled LoRA output
        """
        # Original forward pass (with frozen weights)
        original_output = self.original_linear(x)
        
        # LoRA forward pass
        # x @ A @ B scaled by alpha/rank
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into the original linear layer.
        This is useful for inference when you want to remove the LoRA overhead.
        
        Returns:
            A new nn.Linear with merged weights (no LoRA overhead)
        """
        merged_linear = nn.Linear(
            self.original_linear.in_features,
            self.original_linear.out_features,
            bias=self.original_linear.bias is not None
        )
        
        # Merge: W' = W + (alpha/rank) * A @ B
        with torch.no_grad():
            merged_weight = self.original_linear.weight.data + \
                           (self.lora_A @ self.lora_B).T * self.scaling
            merged_linear.weight.copy_(merged_weight)
            
            if self.original_linear.bias is not None:
                merged_linear.bias.copy_(self.original_linear.bias.data)
        
        return merged_linear
    
    @property
    def in_features(self):
        return self.original_linear.in_features
    
    @property
    def out_features(self):
        return self.original_linear.out_features
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, " \
               f"rank={self.rank}, alpha={self.alpha}"


class LoRAConv2d(nn.Module):
    """
    A Conv2d layer with LoRA (Low-Rank Adaptation) support.
    
    For Conv2d, we treat the kernel as a 2D matrix:
    (out_channels, in_channels * kernel_h * kernel_w)
    
    Args:
        original_conv: The original nn.Conv2d layer to wrap
        rank: The rank of the LoRA decomposition (default: 4)
        alpha: Scaling factor for LoRA (default: 1.0)
    """
    
    def __init__(
        self, 
        original_conv: nn.Conv2d, 
        rank: int = 4, 
        alpha: float = 1.0
    ):
        super().__init__()
        
        self.original_conv = original_conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_channels = original_conv.in_channels
        out_channels = original_conv.out_channels
        kernel_size = original_conv.kernel_size
        
        # Get the device of the original conv layer
        device = original_conv.weight.device
        
        # Flatten kernel dimensions
        kernel_dim = in_channels * kernel_size[0] * kernel_size[1]
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(kernel_dim, rank, device=device))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_channels, device=device))
        
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        
        # Freeze original weights
        for param in self.original_conv.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: original_output + scaled LoRA output
        """
        # Original convolution
        original_output = self.original_conv(x)
        
        # LoRA delta for convolution weights
        # Reshape LoRA contribution to conv kernel shape
        lora_delta = (self.lora_A @ self.lora_B).T * self.scaling
        lora_delta = lora_delta.view(
            self.original_conv.out_channels,
            self.original_conv.in_channels,
            *self.original_conv.kernel_size
        )
        
        # Apply LoRA delta as additional convolution
        lora_output = nn.functional.conv2d(
            x, 
            lora_delta,
            bias=None,
            stride=self.original_conv.stride,
            padding=self.original_conv.padding,
            dilation=self.original_conv.dilation,
            groups=self.original_conv.groups
        )
        
        return original_output + lora_output


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: Optional[Set[str]] = None,
    exclude_modules: Optional[Set[str]] = None,
    apply_to_conv: bool = False,
    dropout: float = 0.0,
    verbose: bool = True
) -> nn.Module:
    """
    Apply LoRA to a model by replacing specified linear layers with LoRALinear layers.
    
    Args:
        model: The model to modify
        rank: The rank of LoRA decomposition
        alpha: Scaling factor for LoRA
        target_modules: Set of module names to apply LoRA to. If None, applies to all Linear layers.
        exclude_modules: Set of module names to exclude from LoRA
        apply_to_conv: Whether to also apply LoRA to Conv2d layers
        dropout: Dropout probability for LoRA layers
        verbose: Whether to print information about replaced layers
    
    Returns:
        The modified model with LoRA layers
    """
    if exclude_modules is None:
        exclude_modules = set()
    
    replaced_layers = []
    
    def _replace_module(parent: nn.Module, name: str, module: nn.Module):
        """Replace a module with its LoRA version"""
        if isinstance(module, nn.Linear):
            lora_module = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, name, lora_module)
            replaced_layers.append(f"{name} (Linear: {module.in_features} -> {module.out_features})")
        elif apply_to_conv and isinstance(module, nn.Conv2d):
            lora_module = LoRAConv2d(module, rank=rank, alpha=alpha)
            setattr(parent, name, lora_module)
            replaced_layers.append(f"{name} (Conv2d: {module.in_channels} -> {module.out_channels})")
    
    def _apply_lora_recursive(parent: nn.Module, prefix: str = ""):
        """Recursively apply LoRA to child modules"""
        for name, module in parent.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Skip excluded modules
            if full_name in exclude_modules or name in exclude_modules:
                continue
            
            # Check if we should apply LoRA to this module
            should_apply = (target_modules is None or 
                          full_name in target_modules or 
                          name in target_modules)
            
            if should_apply and (isinstance(module, nn.Linear) or 
                               (apply_to_conv and isinstance(module, nn.Conv2d))):
                _replace_module(parent, name, module)
            else:
                # Recurse into child modules
                _apply_lora_recursive(module, full_name)
    
    _apply_lora_recursive(model)
    
    if verbose:
        print(f"=" * 60)
        print(f"LoRA Configuration:")
        print(f"  Rank: {rank}")
        print(f"  Alpha: {alpha}")
        print(f"  Scaling: {alpha/rank:.4f}")
        print(f"  Dropout: {dropout}")
        print(f"=" * 60)
        print(f"Replaced {len(replaced_layers)} layers with LoRA:")
        for layer_info in replaced_layers:
            print(f"  - {layer_info}")
        print(f"=" * 60)
    
    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get only the LoRA parameters from a model.
    
    Args:
        model: The model with LoRA layers
        
    Returns:
        List of LoRA parameters (lora_A and lora_B from each LoRALinear/LoRAConv2d)
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            lora_params.append(module.lora_A)
            lora_params.append(module.lora_B)
    return lora_params


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: The model to count parameters for
        only_trainable: If True, only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_non_lora_parameters(model: nn.Module) -> None:
    """
    Freeze all parameters in the model except LoRA parameters.
    
    Args:
        model: The model to freeze
    """
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False


def print_trainable_parameters(model: nn.Module, model_name: str = "Model") -> None:
    """
    Print information about trainable parameters in the model.
    
    Args:
        model: The model to analyze
        model_name: Name to display for the model
    """
    total_params = count_parameters(model, only_trainable=False)
    trainable_params = count_parameters(model, only_trainable=True)
    
    print(f"=" * 60)
    print(f"{model_name} Parameter Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.4f}%")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"=" * 60)


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights into the original layers.
    This removes the LoRA overhead and creates a standard model.
    
    Note: This modifies the model in-place.
    
    Args:
        model: The model with LoRA layers
        
    Returns:
        The model with merged weights (no LoRA layers)
    """
    def _merge_recursive(parent: nn.Module):
        for name, module in parent.named_children():
            if isinstance(module, LoRALinear):
                merged = module.merge_weights()
                setattr(parent, name, merged)
            elif isinstance(module, LoRAConv2d):
                # For Conv2d, we need to implement merge separately
                # For now, just recurse
                _merge_recursive(module)
            else:
                _merge_recursive(module)
    
    _merge_recursive(model)
    return model
