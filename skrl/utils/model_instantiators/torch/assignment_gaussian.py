"""
Model instantiator for assignment-augmented Gaussian policy.

This module provides a factory function to instantiate AssignmentGaussianModel
following the skrl model instantiator pattern.
"""

from typing import Optional, Tuple, Union
import inspect

import gymnasium
import torch

from skrl.models.torch import Model
from skrl.models.torch.assignment_gaussian import AssignmentGaussianModel


def assignment_gaussian_model(
    observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    device: Optional[Union[str, torch.device]] = None,
    clip_actions: bool = False,
    clip_log_std: bool = True,
    min_log_std: float = -20,
    max_log_std: float = 2,
    reduction: str = "sum",
    initial_log_std: float = 0,
    fixed_log_std: bool = False,
    # Assignment module parameters
    assignment_embed_dim: int = 64,
    num_targets: int = 5,
    obs_k_friends: int = 4,
    obs_k_target: int = 5,
    obs_k_friend_targetpos: int = 5,
    entropy_reg_weight: float = 0.01,
    num_heads: int = 4,
    use_quaternion: bool = True,
    # Control MLP parameters
    mlp_layers: list = None,
    return_source: bool = False,
    *args,
    **kwargs,
) -> Union[Model, str]:
    """
    Instantiate an optimized assignment-augmented Gaussian policy model.

    This model includes a differentiable target assignment module with:
    - Multi-head attention mechanism
    - Enhanced query features (with quaternion support)
    - Improved teammate coordination
    - Non-linear conflict penalty

    Args:
        observation_space: Observation/state space or shape (default: None).
                          If not None, num_observations will contain the size of that space
        action_space: Action space or shape (default: None).
                     If not None, num_actions will contain the size of that space
        device: Device on which tensors are allocated (default: None).
               If None, device will be "cuda" if available or "cpu"
        clip_actions: Whether to clip actions to action space bounds (default: False)
        clip_log_std: Whether to clip log standard deviations (default: True)
        min_log_std: Minimum value of log std (default: -20)
        max_log_std: Maximum value of log std (default: 2)
        reduction: Reduction method for log probability: "mean", "sum", "prod", "none" (default: "sum")
        initial_log_std: Initial value for log std parameter (default: 0)
        fixed_log_std: Whether to fix log std (no gradient) (default: False)
        assignment_embed_dim: Embedding dimension for assignment module (default: 64)
        num_targets: Number of enemy targets (default: 5)
        obs_k_friends: Max number of friends in observation (default: 4)
        obs_k_target: Number of targets in observation (default: 5)
        obs_k_friend_targetpos: Number of friend target positions in observation (default: 5)
        entropy_reg_weight: Weight for entropy regularization (default: 0.01)
        num_heads: Number of attention heads (default: 4)
        use_quaternion: Whether to use quaternion for enhanced query features (default: True)
        mlp_layers: Hidden layer sizes for control MLP (default: [512, 256, 128, 64])
        return_source: Whether to return source code instead of model instance (default: False)

    Returns:
        Model instance or source code string

    Example:
        >>> model = assignment_gaussian_model(
        ...     observation_space=env.observation_space,
        ...     action_space=env.action_space,
        ...     device="cuda",
        ...     assignment_embed_dim=64,
        ...     num_targets=5,
        ...     entropy_reg_weight=0.01,
        ...     num_heads=4,
        ...     use_quaternion=True,
        ...     mlp_layers=[512, 256, 128, 64]
        ... )
    """
    if return_source:
        return inspect.getsource(AssignmentGaussianModel)

    if mlp_layers is None:
        mlp_layers = [512, 256, 128, 64]

    return AssignmentGaussianModel(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        clip_actions=clip_actions,
        clip_log_std=clip_log_std,
        min_log_std=min_log_std,
        max_log_std=max_log_std,
        reduction=reduction,
        initial_log_std=initial_log_std,
        fixed_log_std=fixed_log_std,
        assignment_embed_dim=assignment_embed_dim,
        num_targets=num_targets,
        obs_k_friends=obs_k_friends,
        obs_k_target=obs_k_target,
        obs_k_friend_targetpos=obs_k_friend_targetpos,
        entropy_reg_weight=entropy_reg_weight,
        num_heads=num_heads,
        use_quaternion=use_quaternion,
        mlp_layers=mlp_layers,
    )
