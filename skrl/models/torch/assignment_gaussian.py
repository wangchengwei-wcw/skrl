# FIXME：我的目标分配网络会不会用到了太多的目标信息，会不会可解释性太差了？信息用的太多说服力不太行啊。
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
    Differentiable target assignment using multi-head attention mechanism.

    For each agent i and target j, constructs relative feature tokens (Δp_ij, Δv_ij, d_ij)
    and uses multi-head query-key-value attention to compute soft assignment probabilities P_ij.
    """

    def __init__(self, embed_dim: Union[int, List[int]] = 64, num_targets: int = 5,
                 conflict_penalty_weight: float = 1.0, num_heads: int = 4,
                 use_quaternion: bool = True, num_functional_groups: int = 3):
        """
        Args:
            embed_dim: Embedding dimension(s) for query, key, value vectors
                       - If int: use default 2-layer structure [input_dim, 64, embed_dim]
                       - If list: use custom multi-layer structure [input_dim, *embed_dim]
            num_targets: Number of enemy targets (E)
            conflict_penalty_weight: Weight for penalizing assignment conflicts with teammates (deprecated, kept for compatibility)
            num_heads: Number of attention heads (default: 6, must be divisible by num_functional_groups)
            use_quaternion: Whether to use quaternion (10-dim) or just pos+vel (6-dim) for query
            num_functional_groups: Number of functional groups (default: 3 for team/collision/uncertainty)
        """
        # BUG：num_functional_groups不是3就会出错，如需修改，需按情况进行分析。
        super().__init__()
        self.num_targets = num_targets
        self.conflict_penalty_weight = conflict_penalty_weight  # Kept for backward compatibility
        self.num_heads = num_heads
        self.use_quaternion = use_quaternion
        self.num_functional_groups = num_functional_groups

        # 处理embed_dim参数
        if isinstance(embed_dim, int):
            # 单个整数：使用默认两层结构
            self.embed_dim = embed_dim
            # Query维度：根据是否使用四元数决定
            query_input_dim = 10 if use_quaternion else 6
            query_layers = [query_input_dim, 64, embed_dim]
            key_layers = [7, 64, embed_dim]
            value_layers = [7, 64, embed_dim]
        else:
            # 列表：使用自定义多层结构
            self.embed_dim = embed_dim[-1]  # 最后一层是输出维度
            query_input_dim = 10 if use_quaternion else 6
            query_layers = [query_input_dim] + list(embed_dim)
            key_layers = [7] + list(embed_dim)
            value_layers = [7] + list(embed_dim)

        # 确保embed_dim可以被num_heads整除
        if self.embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({num_heads})")

        # 确保num_heads可以被num_functional_groups整除
        if num_heads % num_functional_groups != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_functional_groups ({num_functional_groups})")

        self.head_dim = self.embed_dim // num_heads
        self.heads_per_group = num_heads // num_functional_groups
        self.group_dim = self.heads_per_group * self.head_dim  # 每个功能组的维度

        # 构建query encoder (所有功能组共享)
        query_net_layers = []
        for i in range(len(query_layers) - 1):
            query_net_layers.append(nn.Linear(query_layers[i], query_layers[i+1]))
            if i < len(query_layers) - 2:  # 最后一层不加激活
                query_net_layers.append(nn.ELU())
        self.agent_query_net = nn.Sequential(*query_net_layers)

        # 构建三个Key编码器（功能分离）
        # 每个编码器只输出该组需要的维度: heads_per_group × head_dim
        # 1. 团队分配最优头：输入8维 (7维目标特征 + 1维队友注意力)
        key_team_layers = [8, 64, self.group_dim]
        key_team_net_layers = []
        for i in range(len(key_team_layers) - 1):
            key_team_net_layers.append(nn.Linear(key_team_layers[i], key_team_layers[i+1]))
            if i < len(key_team_layers) - 2:
                key_team_net_layers.append(nn.ELU())
        self.target_key_net_team = nn.Sequential(*key_team_net_layers)

        # 2. 碰撞避免头：输入9维 (7维目标特征 + 2维碰撞风险)
        key_collision_layers = [9, 64, self.group_dim]
        key_collision_net_layers = []
        for i in range(len(key_collision_layers) - 1):
            key_collision_net_layers.append(nn.Linear(key_collision_layers[i], key_collision_layers[i+1]))
            if i < len(key_collision_layers) - 2:
                key_collision_net_layers.append(nn.ELU())
        self.target_key_net_collision = nn.Sequential(*key_collision_net_layers)

        # 3. 不确定性头：输入7维 (标准目标特征)
        key_unc_layers = [7, 64, self.group_dim]
        key_unc_net_layers = []
        for i in range(len(key_unc_layers) - 1):
            key_unc_net_layers.append(nn.Linear(key_unc_layers[i], key_unc_layers[i+1]))
            if i < len(key_unc_layers) - 2:
                key_unc_net_layers.append(nn.ELU())
        self.target_key_net_unc = nn.Sequential(*key_unc_net_layers)

        # 构建value encoder (所有功能组共享)
        value_net_layers = []
        for i in range(len(value_layers) - 1):
            value_net_layers.append(nn.Linear(value_layers[i], value_layers[i+1]))
            if i < len(value_layers) - 2:
                value_net_layers.append(nn.ELU())
        self.target_value_net = nn.Sequential(*value_net_layers)

        # 队友编码器 (用于路由网络)
        self.teammate_encoder = nn.Sequential(
            nn.Linear(6, 32),  # teammate rel_pos + rel_vel
            nn.ELU(),
            nn.Linear(32, 64)
        )

        # 路由网络 (动态权衡三个功能组)
        # 输入: [self_encoded, target_context, teammate_context] = [embed_dim + embed_dim + 64]
        routing_input_dim = self.embed_dim + self.embed_dim + 64
        self.routing_net = nn.Sequential(
            nn.Linear(routing_input_dim, 128),
            nn.ELU(),
            nn.Linear(128, num_functional_groups),
            nn.Softmax(dim=-1)
        )

        # Multi-head attention output projection
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _compute_collision_features(
        self,
        target_rel_pos: torch.Tensor,
        teammate_rel_pos: torch.Tensor,
        teammate_rel_vel: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算每个目标的碰撞风险特征。

        Args:
            target_rel_pos: [B, E, 3] - 目标相对位置
            teammate_rel_pos: [B, K, 3] - 队友相对位置
            teammate_rel_vel: [B, K, 3] - 队友相对速度

        Returns:
            collision_features: [B, E, 2] - (最小垂直距离, 最大接近速度)
        """
        B, E = target_rel_pos.shape[0], target_rel_pos.shape[1]
        K = teammate_rel_pos.shape[1]

        if K == 0:
            return torch.zeros(B, E, 2, device=target_rel_pos.device)        # 没有队友，返回零风险

        collision_features = []
        for e in range(E):
            target_pos = target_rel_pos[:, e:e+1, :]  # [B, 1, 3]

            # 计算"自机→目标"方向
            # FIXME：我观测传进来的目标相对位置是没有归一化的，但是我参数文件里有一个state_preprocessor是normalize_relative_position=True，这个会不会导致这个target_rel_pos是已经归一化了的？
            # 如果是的话这里就不需要再归一化了，如果不是的话这里就需要归一化
            direction_to_target = F.normalize(target_pos, dim=-1, eps=1e-6)  # [B, 1, 3]

            # 计算队友在"自机→目标"方向上的投影
            teammate_projection = (teammate_rel_pos * direction_to_target).sum(dim=-1, keepdim=True)  # [B, K, 1]

            # 计算队友到"自机→目标"路径的垂直距离
            teammate_perpendicular_vec = teammate_rel_pos - teammate_projection * direction_to_target  # [B, K, 3]
            teammate_perpendicular_dist = torch.norm(teammate_perpendicular_vec, dim=-1)  # [B, K]

            # 最小垂直距离（越小越危险）
            min_perp_dist = teammate_perpendicular_dist.min(dim=-1, keepdim=True)[0]  # [B, 1]

            # 队友的接近速度（正值表示正在接近）
            closing_speed = -(teammate_rel_vel * direction_to_target).sum(dim=-1)  # [B, K]
            max_closing_speed = closing_speed.max(dim=-1, keepdim=True)[0]  # [B, 1]

            collision_features.append(torch.cat([min_perp_dist, max_closing_speed], dim=-1))

        collision_features = torch.stack(collision_features, dim=1)  # [B, E, 2]

        return collision_features

    def _compute_teammate_attention(
        self,
        teammate_target_pos: torch.Tensor,
        target_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算队友对目标的注意力分数，用于团队分配最优头。

        Args:
            teammate_target_pos: [B, K_friends, E, 3] - 队友到目标的相对位置
            target_valid: [B, E] - 可选的目标有效性掩码 (1 = 有效)

        Returns:
            teammate_target_attention: [B, E] - 每个目标的聚合注意力分数
        """
        B, K, E = teammate_target_pos.shape[0], teammate_target_pos.shape[1], teammate_target_pos.shape[2]

        if K == 0:
            return None  # 没有队友，返回None

        # 计算队友到目标的简单距离注意力
        teammate_target_dist = torch.norm(teammate_target_pos, dim=-1)  # [B, K_friends, E]
        teammate_attention_scores = 1.0 / (teammate_target_dist + 1e-3)  # [B, K_friends, E]

        # 掩码无效/填充的目标，防止padding=0造成的污染
        if target_valid is not None:
            teammate_attention_scores = teammate_attention_scores.masked_fill(
                ~target_valid.unsqueeze(1), -50.0
            )  # [B, K_friends, E]

        teammate_target_attention = F.softmax(teammate_attention_scores, dim=-1).mean(dim=1)  # [B, E]

        return teammate_target_attention

    def forward(
        self,
        self_state: torch.Tensor,
        target_features: torch.Tensor,
        target_valid: Optional[torch.Tensor] = None,
        teammate_target_pos: Optional[torch.Tensor] = None,
        target_rel_pos: Optional[torch.Tensor] = None,
        teammate_rel_pos: Optional[torch.Tensor] = None,
        teammate_rel_vel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute soft target assignment with functionally-specialized multi-head attention.

        Args:
            self_state: [B, 6 or 10] - Agent's own state (pos + vel) or (pos + vel + quat)
            target_features: [B, E, 7] - Target relative features (rel_pos + rel_vel + dist)
            target_valid: [B, E] - Optional mask indicating valid targets (1 = valid)
            teammate_target_pos: [B, K_friends, E, 3] - Teammate-to-target relative positions (for team optimality head)
            target_rel_pos: [B, E, 3] - Target relative positions (for collision features)
            teammate_rel_pos: [B, K, 3] - Teammate relative positions (for collision features)
            teammate_rel_vel: [B, K, 3] - Teammate relative velocities (for collision features)

        Returns:
            context: [B, embed_dim] - Aggregated target context vector
            assignment_probs: [B, E] - Soft assignment probabilities (sum to 1)
            assignment_features: [B, 6] - Top-3 assignment probs + weighted target position
            routing_weights: [B, num_functional_groups] - Routing weights for interpretability
        """
        B = self_state.shape[0]
        E = target_features.shape[1]

        # Apply validity mask to target features (zero-out invalid targets)
        if target_valid is not None:
            if target_valid.dim() == 3:
                target_valid = target_valid.squeeze(-1)
            target_valid = target_valid.to(device=target_features.device, dtype=torch.bool)
            target_features = target_features * target_valid.to(target_features.dtype).unsqueeze(-1)

        # ========== Step 1: Encode Query (shared across all functional groups) ==========
        q = self.agent_query_net(self_state)  # [B, embed_dim]
        self_encoded = q  # Cache for routing network (avoid recomputation)
        q = q.view(B, self.num_heads, self.head_dim)  # [B, num_heads, head_dim]

        # ========== Step 3: Encode Keys (different for each functional group) ==========
        # 3.1 Team Assignment Optimality Head: augment with teammate attention
        if teammate_target_pos is not None and teammate_target_pos.shape[1] > 0:
            teammate_target_attention = self._compute_teammate_attention(
                teammate_target_pos, target_valid
            )  # [B, E]

            target_features_team = torch.cat([
                target_features,
                teammate_target_attention.unsqueeze(-1)
            ], dim=-1)  # [B, E, 8]
        else:
            # Fallback: use zeros if teammate attention not provided
            target_features_team = torch.cat([
                target_features,
                torch.zeros(B, E, 1, device=target_features.device)
            ], dim=-1)

        k_team = self.target_key_net_team(target_features_team)  # [B, E, embed_dim]
        k_team = k_team.view(B, E, self.heads_per_group, self.head_dim).transpose(1, 2)  # [B, heads_per_group, E, head_dim]

        # 3.2 Collision Avoidance Head: augment with collision risk features
        if target_rel_pos is not None and teammate_rel_pos is not None and teammate_rel_vel is not None:
            collision_features = self._compute_collision_features(
                target_rel_pos, teammate_rel_pos, teammate_rel_vel
            )  # [B, E, 2]
        else:
            # Fallback: use zeros if collision info not provided
            collision_features = torch.zeros(B, E, 2, device=target_features.device)

        target_features_collision = torch.cat([
            target_features,
            collision_features
        ], dim=-1)  # [B, E, 9]

        k_collision = self.target_key_net_collision(target_features_collision)  # [B, E, embed_dim]
        k_collision = k_collision.view(B, E, self.heads_per_group, self.head_dim).transpose(1, 2)  # [B, heads_per_group, E, head_dim]

        # 3.3 Uncertainty Head: standard target features
        k_unc = self.target_key_net_unc(target_features)  # [B, E, embed_dim]
        k_unc = k_unc.view(B, E, self.heads_per_group, self.head_dim).transpose(1, 2)  # [B, heads_per_group, E, head_dim]

        # ========== Step 4: Encode Values (shared across all functional groups) ==========
        v = self.target_value_net(target_features)  # [B, E, embed_dim]
        v = v.view(B, E, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, E, head_dim]

        # ========== Step 5: Compute attention scores for each functional group ==========
        # Split query for each functional group
        q_team = q[:, 0:self.heads_per_group, :]  # [B, heads_per_group, head_dim]
        q_collision = q[:, self.heads_per_group:2*self.heads_per_group, :]  # [B, heads_per_group, head_dim]
        q_unc = q[:, 2*self.heads_per_group:, :]  # [B, heads_per_group, head_dim]

        # Compute attention scores
        scores_team = torch.matmul(q_team.unsqueeze(2), k_team.transpose(2, 3)).squeeze(2) / math.sqrt(self.head_dim)  # [B, heads_per_group, E]
        scores_collision = torch.matmul(q_collision.unsqueeze(2), k_collision.transpose(2, 3)).squeeze(2) / math.sqrt(self.head_dim)  # [B, heads_per_group, E]
        scores_unc = torch.matmul(q_unc.unsqueeze(2), k_unc.transpose(2, 3)).squeeze(2) / math.sqrt(self.head_dim)  # [B, heads_per_group, E]

        # Average scores within each functional group
        team_scores = scores_team.mean(dim=1)  # [B, E]
        collision_scores = scores_collision.mean(dim=1)  # [B, E]
        unc_scores = scores_unc.mean(dim=1)  # [B, E]

        # ========== Step 6: Compute routing weights ==========
        # Compute target context (for routing decision)
        all_scores = torch.stack([team_scores, collision_scores, unc_scores], dim=1)  # [B, 3, E]
        avg_scores = all_scores.mean(dim=1)  # [B, E]

        # Mask invalid targets before softmax to prevent pollution
        if target_valid is not None:
            avg_scores = avg_scores.masked_fill(~target_valid, -50.0)

        v_reshaped = v.transpose(1, 2).contiguous().view(B, E, self.embed_dim)  # [B, E, embed_dim]
        target_context = torch.bmm(F.softmax(avg_scores, dim=-1).unsqueeze(1), v_reshaped).squeeze(1)  # [B, embed_dim]

        # Encode teammate context (for routing decision)
        if teammate_rel_pos is not None and teammate_rel_vel is not None:
            K = teammate_rel_pos.shape[1]
            if K > 0:
                teammate_features = torch.cat([teammate_rel_pos, teammate_rel_vel], dim=-1)  # [B, K, 6]
                teammate_context = self.teammate_encoder(teammate_features).mean(dim=1)  # [B, 64]
            else:
                teammate_context = torch.zeros(B, 64, device=self_state.device)
        else:
            teammate_context = torch.zeros(B, 64, device=self_state.device)

        # Compute routing weights
        routing_input = torch.cat([self_encoded, target_context, teammate_context], dim=-1)  # [B, embed_dim + embed_dim + 64]
        routing_weights = self.routing_net(routing_input)  # [B, num_functional_groups]

        # ========== Step 7: Combine functional groups with routing weights ==========
        final_scores = (
            routing_weights[:, 0:1] * team_scores +
            routing_weights[:, 1:2] * collision_scores +
            routing_weights[:, 2:3] * unc_scores
        )  # [B, E]

        # Clamp scores to prevent overflow in softmax
        final_scores = torch.clamp(final_scores, min=-50, max=50)

        # Mask invalid targets before softmax
        if target_valid is not None:
            final_scores = final_scores.masked_fill(~target_valid, -50.0)

        # Check for NaN/Inf in scores
        if not torch.isfinite(final_scores).all():
            # Fallback to uniform distribution if scores are invalid
            assignment_probs = torch.ones_like(final_scores) / E
        else:
            # Apply softmax to get assignment probabilities
            assignment_probs = F.softmax(final_scores, dim=-1)  # [B, E]

            # Zero-out invalid targets and renormalize
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
                assignment_probs = torch.ones_like(final_scores) / E

        # ========== Step 8: Aggregate context ==========
        context = torch.bmm(assignment_probs.unsqueeze(1), v_reshaped).squeeze(1)  # [B, embed_dim]
        context = self.out_proj(context)  # [B, embed_dim]

        # Extract assignment features for observation augmentation
        top3_probs, _ = torch.topk(assignment_probs, k=min(3, E), dim=-1)
        weighted_target_pos = (assignment_probs.unsqueeze(-1) * target_features[:, :, :3]).sum(dim=1)  # [B, 3]
        assignment_features = torch.cat([top3_probs, weighted_target_pos], dim=-1)  # [B, 6]

        return context, assignment_probs, assignment_features, routing_weights

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
        obs_k_friends: int = 4,
        obs_k_target: int = 5,
        obs_k_friend_targetpos: int = 5,
        entropy_reg_weight: float = 0.01,
        conflict_penalty_weight: float = 1.0,
        num_heads: int = 4,
        use_quaternion: bool = True,
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
            obs_k_friends: Max number of friends in observation (actual may be min(M-1, obs_k_friends))
            obs_k_target: Number of targets in observation
            obs_k_friend_targetpos: Number of friend target positions in observation
            entropy_reg_weight: Weight for entropy regularization
            conflict_penalty_weight: Weight for penalizing assignment conflicts with teammates
            num_heads: Number of attention heads (default: 4)
            use_quaternion: Whether to use quaternion (10-dim) or just pos+vel (6-dim) for query
            mlp_layers: Hidden layer sizes for control MLP
        """
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.assignment_embed_dim = assignment_embed_dim
        self.num_targets = num_targets
        self.obs_k_friends = obs_k_friends
        self.obs_k_target = obs_k_target
        self.obs_k_friend_targetpos = obs_k_friend_targetpos
        self.entropy_reg_weight = entropy_reg_weight
        self.conflict_penalty_weight = conflict_penalty_weight
        self.num_heads = num_heads
        self.use_quaternion = use_quaternion

        # Validate configuration - STRICT CHECK
        if obs_k_target != num_targets:
            raise ValueError(
                f"Configuration error: num_targets ({num_targets}) must equal "
                f"obs_k_target ({obs_k_target}). This mismatch causes dimension errors "
                f"in assignment probability flow and distillation. "
                f"Please update your config file to set both values equal."
            )

        # Assignment module
        self.assignment_module = TargetAssignmentModule(
            embed_dim=assignment_embed_dim,
            num_targets=num_targets,
            conflict_penalty_weight=conflict_penalty_weight,
            num_heads=num_heads,
            use_quaternion=use_quaternion
        )

        # Augmented observation dimension: original_obs + context(final_embed_dim) + assignment_feats(6)
        # Get final embedding dimension
        # NOTE：存疑，这里final_embed_dim的大小会影响后续control mlp的第一层输入维度，是否需要修改或者调大调小？
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

    def _parse_observations(
        self,
        states: torch.Tensor,
        raw_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parse observation vector to extract self state, target relative positions/velocities, and teammate target info.

        Observation layout (configured via obs_k_* parameters):
            - friend_pos: 3*K_friends_actual (where K_friends_actual = min(M-1, obs_k_friends))
            - friend_vel: 3*K_friends_actual
            - self_pos: 3
            - self_vel: 3
            - self_quat: 4 (if use_quaternion=True, extracted from raw_states)
            - target_info: 7*obs_k_target (rel_pos(3) + rel_vel(3) + lock_flag(1))
            - friend_target_pos: K_friends_actual * obs_k_friend_targetpos * 3

        Args:
            states: [B, D] - Observation vector
            raw_states: [B, D_raw] - Raw observation (for extracting quaternion)

        Returns:
            self_state: [B, 6 or 10] - Self position, velocity (and quaternion if use_quaternion=True)
            target_rel_pos: [B, E, 3] - Target relative positions (padded/truncated to num_targets)
            target_rel_vel: [B, E, 3] - Target relative velocities (from environment)
            target_valid: [B, E] - Target validity mask (lock_flag > 0.5)
            teammate_target_pos: [B, K_friends_actual, E, 3] - Teammate-to-target relative positions
            teammate_rel_pos: [B, K_friends_actual, 3] - Ego-to-teammate relative positions
        """
        if states.dim() != 2:
            raise ValueError(f"Expected states to be 2D [B, D], got shape: {tuple(states.shape)}")

        obs_dim = int(states.shape[-1])
        e = int(self.num_targets)
        k_target = self.obs_k_target
        k_friend_targetpos = self.obs_k_friend_targetpos

        # Dynamically infer actual K_friends from observation dimension
        # obs_dim = 6*Kf + 6 + 7*k_target + 3*Kf*k_friend_targetpos
        #         = (6 + 3*k_friend_targetpos)*Kf + (6 + 7*k_target)
        base_dim = 6 + 7 * k_target
        rem = obs_dim - base_dim
        denom = 6 + 3 * k_friend_targetpos

        if rem < 0 or rem % denom != 0:
            raise ValueError(
                f"Observation dimension mismatch! "
                f"obs_dim={obs_dim}, expected format: (6 + 3*obs_k_friend_targetpos)*K_friends + (6 + 7*obs_k_target). "
                f"Config: obs_k_target={k_target}, obs_k_friend_targetpos={k_friend_targetpos}, num_targets={e}. "
                f"Calculated: base_dim={base_dim}, rem={rem}, denom={denom}. "
                f"Please ensure model config matches environment config."
            )

        k_friends_actual = rem // denom
        if k_friends_actual <= 0 or k_friends_actual > self.obs_k_friends:
            raise ValueError(
                f"Invalid inferred K_friends_actual={k_friends_actual} from obs_dim={obs_dim}. "
                f"Expected 0 < K_friends_actual <= obs_k_friends({self.obs_k_friends}). "
                f"This likely means observation dimension doesn't match config."
            )

        # Extract teammate relative positions and velocities
        # Observation layout: [friend_pos(3*Kf), friend_vel(3*Kf), self_pos(3), self_vel(3), ...]
        friend_rel_pos = states[:, :3*k_friends_actual].reshape(-1, k_friends_actual, 3)  # [B, Kf, 3]
        friend_rel_vel = states[:, 3*k_friends_actual:6*k_friends_actual].reshape(-1, k_friends_actual, 3)  # [B, Kf, 3]

        friend_block_end = 6 * k_friends_actual
        self_pos_start = friend_block_end
        self_vel_start = self_pos_start + 3
        self_end = self_vel_start + 3
        target_start = self_end
        target_end = target_start + 7 * k_target
        friend_target_start = target_end
        friend_target_end = friend_target_start + 3 * k_friends_actual * k_friend_targetpos

        if friend_target_end != obs_dim:
            raise ValueError(
                f"Observation slicing mismatch: friend_target_end={friend_target_end}, obs_dim={obs_dim}. "
                f"k_friends_actual={k_friends_actual}, k_target={k_target}, k_friend_targetpos={k_friend_targetpos}. "
                "Please check observation construction and parser assumptions."
            )

        # Extract self state
        self_pos = states[:, self_pos_start:self_vel_start]  # [B, 3]
        self_vel = states[:, self_vel_start:self_end]  # [B, 3]

        # Add quaternion if use_quaternion=True
        # NOTE：现在还有没在观测中加入四元数，先不能使用
        if self.use_quaternion and raw_states is not None:
            # Quaternion is typically at the end of the raw observation
            # Assuming raw_states has format: [..., quat(4)]
            # We need to extract it from raw_states
            # For now, try to extract from the end of raw_states
            try:
                self_quat = raw_states[:, -4:]  # [B, 4] - last 4 elements as quaternion
                self_state = torch.cat([self_pos, self_vel, self_quat], dim=-1)  # [B, 10]
            except:
                # Fallback to 6-dim if quaternion extraction fails
                self_state = torch.cat([self_pos, self_vel], dim=-1)  # [B, 6]
        else:
            self_state = torch.cat([self_pos, self_vel], dim=-1)  # [B, 6]

        # Extract target info: [B, 7*k_target] -> [B, k_target, 7]
        target_info = states[:, target_start:target_end].reshape(-1, k_target, 7)

        # Pad or truncate to match num_targets (E) if k_target != E
        if k_target != e:
            if k_target < e:
                # Pad with zeros
                pad_size = e - k_target
                target_info = torch.cat([
                    target_info,
                    torch.zeros(target_info.shape[0], pad_size, 7, device=target_info.device, dtype=target_info.dtype)
                ], dim=1)
            else:
                # Truncate
                target_info = target_info[:, :e, :]

        target_rel_pos = target_info[:, :, :3]   # [B, E, 3]
        target_rel_vel = target_info[:, :, 3:6]  # [B, E, 3] - from environment

        if raw_states is None:
            lock_src = target_info[:, :, 6]
        else:
            raw_target_info = raw_states[:, target_start:target_end].reshape(-1, k_target, 7)
            if k_target != e:
                if k_target < e:
                    raw_target_info = torch.cat([
                        raw_target_info,
                        torch.zeros(raw_target_info.shape[0], e - k_target, 7, device=raw_target_info.device, dtype=raw_target_info.dtype)
                    ], dim=1)
                else:
                    raw_target_info = raw_target_info[:, :e, :]
            lock_src = raw_target_info[:, :, 6]

        target_valid = lock_src > 0.5
        target_rel_pos = target_rel_pos * target_valid.to(target_rel_pos.dtype).unsqueeze(-1)
        target_rel_vel = target_rel_vel * target_valid.to(target_rel_vel.dtype).unsqueeze(-1)

        # Extract teammate target positions: [B, 3*Kf*k_friend_targetpos] -> [B, Kf, k_friend_targetpos, 3]
        friend_target_pos = states[:, friend_target_start:friend_target_end].reshape(-1, k_friends_actual, k_friend_targetpos, 3)

        # Pad or truncate to match num_targets (E) if k_friend_targetpos != E
        if k_friend_targetpos != e:
            if k_friend_targetpos < e:
                # Pad with zeros
                pad_size = e - k_friend_targetpos
                friend_target_pos = torch.cat([
                    friend_target_pos,
                    torch.zeros(friend_target_pos.shape[0], k_friends_actual, pad_size, 3,
                               device=friend_target_pos.device, dtype=friend_target_pos.dtype)
                ], dim=2)
            else:
                # Truncate
                friend_target_pos = friend_target_pos[:, :, :e, :]

        return self_state, target_rel_pos, target_rel_vel, target_valid, friend_target_pos, friend_rel_pos, friend_rel_vel

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
        self_state, target_rel_pos, target_rel_vel, target_valid, teammate_target_pos, teammate_rel_pos, teammate_rel_vel = self._parse_observations(
            states, 
            raw_states=raw_states
        )

        # Compute target distances
        # NOTE:这里是归一化之后的相对位置（target_rel_pos）
        target_dist = torch.norm(target_rel_pos, dim=-1, keepdim=True)  # [B, E, 1]

        # Concatenate target features: [rel_pos, rel_vel, dist]
        target_features = torch.cat([
            target_rel_pos,  # [B, E, 3]
            target_rel_vel,  # [B, E, 3]
            target_dist      # [B, E, 1]
        ], dim=-1)  # [B, E, 7]

        # Assignment module: compute soft assignment with functionally-specialized attention
        # NOTE: TargetAssignmentModule.forward() will compute teammate attention internally, along with assignment_probs and context
        context, assignment_probs, assignment_feats, routing_weights = self.assignment_module(
            self_state, target_features, target_valid=target_valid,
            teammate_target_pos=teammate_target_pos,
            target_rel_pos=target_rel_pos,
            teammate_rel_pos=teammate_rel_pos,
            teammate_rel_vel=teammate_rel_vel
        )

        # Cache assignment probs and routing weights
        self._assignment_probs_cache = assignment_probs
        self._routing_weights_cache = routing_weights

        # Augment observations: [original_obs, context, assignment_feats]
        aug_obs = torch.cat([states, context, assignment_feats], dim=-1)

        # Generate mean actions through control MLP
        mean_actions = self.control_net(aug_obs)

        # Prepare outputs dictionary
        outputs = {
            'assignment_probs': assignment_probs,
            'assignment_entropy': self.get_assignment_entropy(),
            'context': context,
            'routing_weights': routing_weights,  # Add routing weights for logging
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
