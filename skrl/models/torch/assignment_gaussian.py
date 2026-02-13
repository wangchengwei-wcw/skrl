"""
Gaussian policy model with differentiable target assignment module.

This module implements an attention-based target assignment mechanism that learns
to softly assign targets to agents in a differentiable manner, enabling end-to-end
training with PPO gradients.
"""

from typing import Mapping, Optional, Tuple, Union, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space


class TargetAssignmentModule(nn.Module):
    """
    Differentiable target assignment using attention mechanism.

    For each agent i and target j, constructs relative feature tokens (Δp_ij, Δv_ij, d_ij)
    and uses query-key-value attention to compute soft assignment probabilities P_ij.
    """

    def __init__(self, embed_dim: Union[int, List[int]] = 64, num_targets: int = 5,
                 conflict_penalty_weight: float = 1.0):
        """
        Args:
            embed_dim: Embedding dimension(s) for query, key, value vectors
                       - If int: use default 2-layer structure [input_dim, 64, embed_dim]
                       - If list: use custom multi-layer structure [input_dim, *embed_dim]
            num_targets: Number of enemy targets (E)
            conflict_penalty_weight: Weight for penalizing assignment conflicts with teammates
        """
        super().__init__()
        self.num_targets = num_targets
        self.conflict_penalty_weight = conflict_penalty_weight

        # 处理embed_dim参数
        if isinstance(embed_dim, int):
            # 单个整数：使用默认两层结构
            self.embed_dim = embed_dim
            query_layers = [6, 64, embed_dim]
            key_layers = [7, 64, embed_dim]
            value_layers = [7, 64, embed_dim]
        else:
            # 列表：使用自定义多层结构
            self.embed_dim = embed_dim[-1]  # 最后一层是输出维度
            query_layers = [6] + list(embed_dim)
            key_layers = [7] + list(embed_dim)
            value_layers = [7] + list(embed_dim)

        # 构建query encoder
        query_net_layers = []
        for i in range(len(query_layers) - 1):
            query_net_layers.append(nn.Linear(query_layers[i], query_layers[i+1]))
            if i < len(query_layers) - 2:  # 最后一层不加激活
                query_net_layers.append(nn.ELU())
        self.agent_query_net = nn.Sequential(*query_net_layers)

        # 构建key encoder
        key_net_layers = []
        for i in range(len(key_layers) - 1):
            key_net_layers.append(nn.Linear(key_layers[i], key_layers[i+1]))
            if i < len(key_layers) - 2:
                key_net_layers.append(nn.ELU())
        self.target_key_net = nn.Sequential(*key_net_layers)

        # 构建value encoder
        value_net_layers = []
        for i in range(len(value_layers) - 1):
            value_net_layers.append(nn.Linear(value_layers[i], value_layers[i+1]))
            if i < len(value_layers) - 2:
                value_net_layers.append(nn.ELU())
        self.target_value_net = nn.Sequential(*value_net_layers)

    def forward(
        self,
        self_state: torch.Tensor,
        target_features: torch.Tensor,
        target_valid: Optional[torch.Tensor] = None,
        teammate_target_attention: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute soft target assignment and context aggregation with multi-agent coordination.

        Args:
            self_state: [B, 6] - Agent's own state (pos + vel)
            target_features: [B, E, 7] - Target relative features (rel_pos + rel_vel + dist)
            target_valid: [B, E] - Optional mask indicating valid targets (1 = valid)
            teammate_target_attention: [B, E] - Optional teammate attention scores for conflict avoidance

        Returns:
            context: [B, embed_dim] - Aggregated target context vector
            assignment_probs: [B, E] - Soft assignment probabilities (sum to 1)
            assignment_features: [B, 6] - Top-3 assignment probs + weighted target position
            conflict_score: [B] - Conflict score with teammates (higher = more conflict)
        """
        B = self_state.shape[0]
        E = target_features.shape[1]

        # Apply validity mask to target features (zero-out invalid targets)
        if target_valid is not None:
            if target_valid.dim() == 3:
                target_valid = target_valid.squeeze(-1)
            target_valid = target_valid.to(device=target_features.device, dtype=torch.bool)
            target_features = target_features * target_valid.to(target_features.dtype).unsqueeze(-1)

        # Encode query from agent's own state
        q = self.agent_query_net(self_state)  # [B, embed_dim]

        # Encode keys and values from target features
        k = self.target_key_net(target_features)  # [B, E, embed_dim]
        v = self.target_value_net(target_features)  # [B, E, embed_dim]

        # Compute attention scores: S_ij = q_i^T k_ij / sqrt(d)
        # [B, 1, embed_dim] @ [B, embed_dim, E] -> [B, 1, E] -> [B, E]
        scores = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1)
        scores = scores / math.sqrt(self.embed_dim)

        # Apply conflict penalty: reduce scores for targets that teammates are also attending to
        if teammate_target_attention is not None:
            # teammate_target_attention: [B, E] - aggregated attention from teammates
            # Higher values mean more teammates are interested in this target
            conflict_penalty = self.conflict_penalty_weight * teammate_target_attention
            scores = scores - conflict_penalty

        # Clamp scores to prevent overflow in softmax
        scores = torch.clamp(scores, min=-50, max=50)

        # Mask invalid targets before softmax
        if target_valid is not None:
            scores = scores.masked_fill(~target_valid, -50.0)

        # Check for NaN/Inf in scores
        if not torch.isfinite(scores).all():
            # Fallback to uniform distribution if scores are invalid
            assignment_probs = torch.ones_like(scores) / E
        else:
            # Apply softmax to get assignment probabilities
            assignment_probs = F.softmax(scores, dim=-1)  # [B, E]

            # Zero-out invalid targets and renormalize
            # TODO：是否可以在输入观测的时候就把无效目标的特征直接置零，这样就不需要在这里再处理一次了，或者前面softmax的时候就能让这些无效目标概率置零。
            if target_valid is not None:
                valid_f = target_valid.to(assignment_probs.dtype)
                assignment_probs = assignment_probs * valid_f
                denom = assignment_probs.sum(dim=-1, keepdim=True)
                assignment_probs = torch.where(
                    denom > 0,
                    assignment_probs / denom,
                    torch.ones_like(assignment_probs) / E,
                )

            # Final safety check
            if not torch.isfinite(assignment_probs).all():
                assignment_probs = torch.ones_like(scores) / E

        # Aggregate context: c_i = Σ_j P_ij * v_ij
        # [B, 1, E] @ [B, E, embed_dim] -> [B, 1, embed_dim] -> [B, embed_dim]
        context = torch.bmm(assignment_probs.unsqueeze(1), v).squeeze(1)

        # Extract assignment features for observation augmentation
        # Use top-3 probabilities + weighted-average target position (position-based, no unstable indices)
        top3_probs, _ = torch.topk(assignment_probs, k=min(3, E), dim=-1)
        weighted_target_pos = (assignment_probs.unsqueeze(-1) * target_features[:, :, :3]).sum(dim=1)  # [B, 3]
        assignment_features = torch.cat([top3_probs, weighted_target_pos], dim=-1)  # [B, 6]

        # Compute conflict score with teammates
        if teammate_target_attention is not None:
            # Conflict = overlap between my assignment and teammates' attention
            conflict_score = (assignment_probs * teammate_target_attention).sum(dim=-1)  # [B]
        else:
            conflict_score = torch.zeros(assignment_probs.shape[0], device=assignment_probs.device)

        return context, assignment_probs, assignment_features, conflict_score

class AssignmentGaussianModel(GaussianMixin, Model):
    """
    Gaussian policy with differentiable target assignment module.

    Architecture:
        Observations -> Parse Features -> Assignment Module -> Augmented Obs -> Control MLP -> Actions

    The assignment module learns to softly assign targets to agents using attention,
    and the resulting context is concatenated with observations before action generation.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device: Optional[Union[str, torch.device]] = None,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: str = "sum",
        initial_log_std: float = 0,
        fixed_log_std: bool = False,
        # Assignment module parameters
        assignment_embed_dim: Union[int, List[int]] = 64,
        num_targets: int = 5,
        entropy_reg_weight: float = 0.01,
        conflict_penalty_weight: float = 1.0,
        # Control MLP parameters
        mlp_layers: list = None,
    ):
        """
        Args:
            observation_space: Observation space
            action_space: Action space
            device: Device for computation
            clip_actions: Whether to clip actions
            clip_log_std: Whether to clip log std
            min_log_std: Minimum log std
            max_log_std: Maximum log std
            reduction: Reduction method for log prob
            initial_log_std: Initial log std value
            fixed_log_std: Whether to fix log std
            assignment_embed_dim: Embedding dimension(s) for assignment module
                                  - If int: use default 2-layer structure
                                  - If list: use custom multi-layer structure
            num_targets: Number of enemy targets
            entropy_reg_weight: Weight for entropy regularization
            conflict_penalty_weight: Weight for penalizing assignment conflicts with teammates
            mlp_layers: Hidden layer sizes for control MLP
        """
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        if mlp_layers is None:
            mlp_layers = [512, 256, 128, 64]

        self.assignment_embed_dim = assignment_embed_dim
        self.num_targets = num_targets
        self.entropy_reg_weight = entropy_reg_weight
        self.conflict_penalty_weight = conflict_penalty_weight

        # Assignment module
        self.assignment_module = TargetAssignmentModule(
            embed_dim=assignment_embed_dim,
            num_targets=num_targets,
            conflict_penalty_weight=conflict_penalty_weight
        )

        # Augmented observation dimension: original_obs + context(final_embed_dim) + assignment_feats(6)
        # Get final embedding dimension
        # NOTE：存疑，final_embed_dim的维度为什么是确定的，里面到底是做了什么？
        if isinstance(assignment_embed_dim, int):
            final_embed_dim = assignment_embed_dim
        else:
            final_embed_dim = assignment_embed_dim[-1]
        aug_obs_dim = self.num_observations + final_embed_dim + 6

        # Control MLP: augmented_obs -> actions
        layers = []
        in_dim = aug_obs_dim
        for hidden_dim in mlp_layers:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ELU()
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.num_actions))
        self.control_net = nn.Sequential(*layers)

        # Log std parameter for Gaussian policy
        if fixed_log_std:
            self.register_buffer(
                'log_std_parameter',
                torch.full((self.num_actions,), initial_log_std)
            )
        else:
            self.log_std_parameter = nn.Parameter(
                torch.full((self.num_actions,), initial_log_std),
                requires_grad=True
            )

        # Cache for assignment probabilities (for entropy regularization)
        self._assignment_probs_cache = None
        self._conflict_score_cache = None

    def _parse_observations(
        self,
        states: torch.Tensor,
        raw_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parse observation vector to extract self state, target relative positions/velocities, and teammate target info.

        Observation layout (dynamic):
            - friend_pos: 3*K_friends
            - friend_vel: 3*K_friends
            - self_pos: 3
            - self_vel: 3
            - target_info: 7*K_target (rel_pos(3) + rel_vel(3) + lock_flag(1))
            - friend_target_pos: K_friends * K_target * 3

        Args:
            states: [B, D] - Observation vector

        Returns:
            self_state: [B, 6] - Self position and velocity
            target_rel_pos: [B, E, 3] - Target relative positions
            target_rel_vel: [B, E, 3] - Target relative velocities (from environment)
            target_valid: [B, E] - Target validity mask (lock_flag > 0.5)
            teammate_target_pos: [B, K_friends, E, 3] - Teammate-to-target relative positions (aligned)
        """
        # BUG 观测中将obs_k_friends从5变为4训练就会崩，后续需要将这里的K_friends、K_target写入到yaml里，这样方便我修改
        if states.dim() != 2:
            raise ValueError(f"Expected states to be 2D [B, D], got shape: {tuple(states.shape)}")

        obs_dim = int(states.shape[-1])
        e = int(self.num_targets)
        base_dim = 6 + 7 * e
        rem = obs_dim - base_dim
        denom = 6 + 3 * e

        # This parser assumes friend_target_pos uses K_target slots (k_fixed == num_targets).
        # Then: obs_dim = 6*Kf + 6 + 7*E + 3*Kf*E = (6 + 3E)*Kf + (6 + 7E).
        if rem < 0 or rem % denom != 0:
            raise ValueError(
                f"Observation dimension mismatch for assignment parser. "
                f"obs_dim={obs_dim}, num_targets={e}, rem={rem}, denom={denom}. "
                "Expected obs_dim = (6 + 3*num_targets)*K_friends + (6 + 7*num_targets). "
                "Please align env obs_k_* settings and model num_targets."
            )

        k_friends = rem // denom
        if k_friends <= 0:
            raise ValueError(f"Invalid inferred K_friends={k_friends} from obs_dim={obs_dim}, num_targets={e}")

        friend_block_end = 6 * k_friends
        self_pos_start = friend_block_end
        self_vel_start = self_pos_start + 3
        self_end = self_vel_start + 3
        target_start = self_end
        target_end = target_start + 7 * e
        friend_target_start = target_end
        friend_target_end = friend_target_start + 3 * k_friends * e

        if friend_target_end != obs_dim:
            raise ValueError(
                f"Observation slicing mismatch: friend_target_end={friend_target_end}, obs_dim={obs_dim}. "
                "Please check observation construction and parser assumptions."
            )

        # Extract self state
        self_pos = states[:, self_pos_start:self_vel_start]  # [B, 3]
        self_vel = states[:, self_vel_start:self_end]  # [B, 3]
        self_state = torch.cat([self_pos, self_vel], dim=-1)  # [B, 6]

        # Extract target info: [B, 7*E] -> [B, E, 7]
        target_info = states[:, target_start:target_end].reshape(-1, e, 7)
        target_rel_pos = target_info[:, :, :3]   # [B, E, 3]
        target_rel_vel = target_info[:, :, 3:6]  # [B, E, 3] - from environment
        if raw_states is None:
            lock_src = target_info[:, :, 6]
        else:
            raw_target_info = raw_states[:, target_start:target_end].reshape(-1, e, 7)
            lock_src = raw_target_info[:, :, 6]
        target_valid = lock_src > 0.5
        target_rel_pos = target_rel_pos * target_valid.to(target_rel_pos.dtype).unsqueeze(-1)
        target_rel_vel = target_rel_vel * target_valid.to(target_rel_vel.dtype).unsqueeze(-1)

        # Extract teammate target positions: [B, 3*K_friends*E] -> [B, K_friends, E, 3]
        friend_target_pos = states[:, friend_target_start:friend_target_end].reshape(-1, k_friends, e, 3)

        return self_state, target_rel_pos, target_rel_vel, target_valid, friend_target_pos

    def _compute_teammate_attention(
        self,
        teammate_target_pos: torch.Tensor,
        target_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算队友对每个目标的关注度，用于冲突避免。

        使用基于距离的软注意力机制：距离越近的目标，队友的关注度越高。

        Args:
            teammate_target_pos: [B, K_friends, E, 3] - 队友到目标的相对位置
            target_valid: [B, E] - 目标有效性掩码

        Returns:
            teammate_attention: [B, E] - 队友对每个目标的聚合关注度分数
        """
        # 计算队友到目标的距离: [B, K_friends, E]
        teammate_target_dist = torch.norm(teammate_target_pos, dim=-1)

        # 使用反距离作为关注度（距离越近，关注度越高）
        # 添加小常数避免除零
        epsilon = 1e-3
        teammate_attention_scores = 1.0 / (teammate_target_dist + epsilon)  # [B, K_friends, E]

        # 应用目标有效性掩码
        if target_valid is not None:
            valid_mask = target_valid.unsqueeze(1)  # [B, 1, E]
            teammate_attention_scores = teammate_attention_scores * valid_mask.to(teammate_attention_scores.dtype)

        # 对每个队友的关注度进行归一化（softmax）
        teammate_attention_probs = F.softmax(teammate_attention_scores, dim=-1)  # [B, K_friends, E]

        # 聚合所有队友的关注度（求和）
        # 如果多个队友都关注同一个目标，该目标的总关注度会更高
        teammate_attention = teammate_attention_probs.sum(dim=1)  # [B, E]

        return teammate_attention

    def compute(self, inputs: Mapping[str, torch.Tensor], role: str = "") -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass: observations -> assignment -> augmented obs -> actions.

        Args:
            inputs: Dictionary with 'states' key containing observations
            role: Role identifier (unused)

        Returns:
            mean_actions: [B, num_actions] - Mean actions for Gaussian policy
            log_std: [num_actions] - Log standard deviation
            outputs: Dictionary with auxiliary information (assignment probs, entropy)
        """
        # Unflatten and extract states
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        if states.dim() > 2:
            states = states.view(states.shape[0], -1)
        raw_states = inputs.get("raw_states", None)
        if raw_states is not None:
            raw_states = unflatten_tensorized_space(self.observation_space, raw_states)
            if raw_states.dim() > 2:
                raw_states = raw_states.view(raw_states.shape[0], -1)

        # Parse observations (target velocity now comes directly from environment)
        self_state, target_rel_pos, target_rel_vel, target_valid, teammate_target_pos = self._parse_observations(states, raw_states=raw_states)

        # Compute target distances
        target_dist = torch.norm(target_rel_pos, dim=-1, keepdim=True)  # [B, E, 1]

        # Concatenate target features: [rel_pos, rel_vel, dist]
        target_features = torch.cat([
            target_rel_pos,  # [B, E, 3]
            target_rel_vel,  # [B, E, 3]
            target_dist      # [B, E, 1]
        ], dim=-1)  # [B, E, 7]

        # Compute teammate attention scores for conflict avoidance
        # teammate_target_pos: [B, K_friends, E, 3] (now correctly aligned to observer's target ordering)
        teammate_target_attention = self._compute_teammate_attention(
            teammate_target_pos, target_valid
        )  # [B, E]

        # Assignment module: compute soft assignment and context
        context, assignment_probs, assignment_feats, conflict_score = self.assignment_module(
            self_state, target_features, target_valid=target_valid,
            teammate_target_attention=teammate_target_attention
        )

        # Cache assignment probs and conflict score
        self._assignment_probs_cache = assignment_probs
        self._conflict_score_cache = conflict_score

        # Augment observations: [original_obs, context, assignment_feats]
        aug_obs = torch.cat([states, context, assignment_feats], dim=-1)

        # Generate mean actions through control MLP
        mean_actions = self.control_net(aug_obs)

        # Prepare outputs dictionary
        outputs = {
            'assignment_probs': assignment_probs,
            'assignment_entropy': self.get_assignment_entropy(),
            'conflict_score': conflict_score,
            'context': context,
        }

        return mean_actions, self.log_std_parameter, outputs

    def get_assignment_entropy(self) -> torch.Tensor:
        """
        Compute entropy of assignment distribution: -Σ_j P_ij log(P_ij).

        Higher entropy means more uniform assignment (less decisive).
        Lower entropy means peaked assignment (more decisive).

        Returns:
            entropy: Scalar tensor with mean entropy across batch
        """
        if self._assignment_probs_cache is None:
            return torch.tensor(0.0, device=self.device)

        # Compute entropy: -Σ_j P_ij log(P_ij)
        probs = self._assignment_probs_cache

        # Clamp probabilities to avoid log(0) and ensure numerical stability
        probs = torch.clamp(probs, min=1e-8, max=1.0)

        # Check for NaN/Inf in probs before computing entropy
        if not torch.isfinite(probs).all():
            return torch.tensor(0.0, device=self.device)

        entropy = -(probs * torch.log(probs)).sum(dim=-1)  # [B]

        # Check for NaN/Inf in entropy result
        if not torch.isfinite(entropy).all():
            return torch.tensor(0.0, device=self.device)

        return entropy.mean()

    def get_entropy_regularization(self) -> torch.Tensor:
        """
        Compute entropy regularization loss.

        Encourages peaked assignment distributions by penalizing high entropy.
        This helps prevent multiple agents from being uncertain about target assignment.

        Returns:
            reg_loss: Scalar regularization loss
        """
        entropy = self.get_assignment_entropy()
        # reg_loss = -self.entropy_reg_weight * entropy   # 鼓励高熵，让其更均匀地分配目标
        reg_loss = self.entropy_reg_weight * entropy   # 惩罚高熵，让其更确定地分配目标
        return reg_loss
