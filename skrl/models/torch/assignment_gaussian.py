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
    Differentiable target assignment using standard multi-head attention.

    For each agent i and target j, constructs relative feature tokens (Δp_ij, Δv_ij, d_ij)
    and uses multi-head query-key-value attention to compute soft assignment probabilities P_ij.
    """

    def __init__(self, embed_dim: Union[int, List[int]] = 64, num_targets: int = 5,
                 num_heads: int = 4,
                 use_quaternion: bool = False):
        """
        Args:
            embed_dim: Embedding dimension(s) for query, key, value vectors
                       - If int: use default 2-layer structure [input_dim, 64, embed_dim]
                       - If list: use custom multi-layer structure [input_dim, *embed_dim]
            num_targets: Number of enemy targets (E)
            num_heads: Number of attention heads (must be divisible into embed_dim)
            use_quaternion: Whether to use quaternion (10-dim) or just pos+vel (6-dim) for query
        """
        super().__init__()
        self.num_targets = num_targets
        self.num_heads = num_heads
        self.use_quaternion = use_quaternion

        # 处理embed_dim参数
        # NOTE：yaml中隐藏层如果是列表就走else分支，这里qkv的每层维度都一样，但是可以设置的不一样。
        if isinstance(embed_dim, int):
            self.embed_dim = embed_dim
            query_input_dim = 11 if use_quaternion else 7
            query_layers = [query_input_dim, 64, embed_dim]
            key_layers = [12, 64, embed_dim]
            value_layers = [7, 64, embed_dim]
        else:
            self.embed_dim = embed_dim[-1]
            query_input_dim = 11 if use_quaternion else 7
            query_layers = [query_input_dim] + list(embed_dim)
            key_layers = [12] + list(embed_dim)
            value_layers = [7] + list(embed_dim)

        if self.embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({num_heads})")

        # 构建query encoder
        query_net_layers = []
        for i in range(len(query_layers) - 1):
            query_net_layers.append(nn.Linear(query_layers[i], query_layers[i+1]))
            if i < len(query_layers) - 2:
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

        # 创建一个标准的多头注意力
        self.mha = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True)

    def forward(
        self,
        self_state: torch.Tensor,
        target_features_k: torch.Tensor,
        target_features_v: Optional[torch.Tensor] = None,
        target_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute soft target assignment with standard multi-head attention.

        K uses cooperative features (who to attend to),
        V uses ego-centric features (what information to aggregate).

        Args:
            self_state: [B, 7 or 11] - Agent's own state (pos + vel + yaw) or (+ quat)
            target_features_k: [B, E, 12] - Target features for K (rel_pos + rel_vel + dist + team_min_dist + closer_fraction + team_max_closing + team_approaching_frac + team_min_eta)
            target_features_v: [B, E, 7]  - Target features for V (rel_pos + rel_vel + dist). If None, uses target_features_k[:, :, :7]
            target_valid: [B, E] - Optional mask indicating valid targets (1 = valid)

        Returns:
            context: [B, embed_dim] - Aggregated target context vector
            assignment_probs: [B, E] - Soft assignment probabilities (sum to 1)
            assignment_features: [B, 5] - assignment_entropy(1) + min_dist(1) + weighted_target_pos(3)
        """
        if target_features_v is None:
            target_features_v = target_features_k[:, :, :7]

        # Apply validity mask to target features (zero-out invalid targets)
        if target_valid is not None:
            if target_valid.dim() == 3:
                target_valid = target_valid.squeeze(-1)
            target_valid = target_valid.to(device=target_features_k.device, dtype=torch.bool)
            valid_f = target_valid.to(target_features_k.dtype).unsqueeze(-1)
            target_features_k = target_features_k * valid_f
            target_features_v = target_features_v * valid_f

        # Encode Q, K, V
        q = self.agent_query_net(self_state).unsqueeze(1)    # [B, 1, embed_dim]
        k = self.target_key_net(target_features_k)           # [B, E, embed_dim]
        v = self.target_value_net(target_features_v)         # [B, E, embed_dim]

        # key_padding_mask: True = 忽略该位置 (PyTorch MHA 约定)
        # 当某行所有目标都无效时，key_padding_mask 全为 True，
        # MHA 内部 softmax 输入全是 -inf，输出 NaN。
        if target_valid is not None:
            has_valid = target_valid.any(dim=-1)  # [B], True = 该行至少有一个有效目标
            safe_mask = target_valid.clone()
            safe_mask[~has_valid, 0] = True       # 全无效行：临时开放 slot 0
            key_padding_mask = ~safe_mask         # [B, E]
        else:
            has_valid = None
            key_padding_mask = None

        context, attn_weights = self.mha(
            q, k, v,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        context = context.squeeze(1)               # [B, embed_dim]
        assignment_probs = attn_weights.squeeze(1)  # [B, E]

        # 掩码无效目标并重归一化
        if target_valid is not None:
            valid_f = target_valid.to(assignment_probs.dtype)
            assignment_probs = assignment_probs * valid_f
            denom = assignment_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            assignment_probs = assignment_probs / denom
            # 全无效行：context 和 assignment_probs 置零（无目标可分配）
            has_valid_f = has_valid.to(assignment_probs.dtype).unsqueeze(-1)  # [B, 1]
            assignment_probs = assignment_probs * has_valid_f
            context = context * has_valid_f

        # 分配熵：衡量当前分配的确定性（低熵=果断，高熵=犹豫）[B, 1]
        assignment_entropy = -(assignment_probs * torch.log(assignment_probs.clamp(min=1e-8))).sum(dim=-1, keepdim=True)
        # 最近有效目标距离：target_features_v[:, :, 6] 是距离维 [B, 1]
        if target_valid is not None:
            dist_masked = target_features_v[:, :, 6].masked_fill(~target_valid, float('inf'))
            min_dist = dist_masked.min(dim=-1, keepdim=True)[0].clamp(max=1e4)  # [B, 1]
        else:
            min_dist = target_features_v[:, :, 6].min(dim=-1, keepdim=True)[0]   # [B, 1]
        weighted_target_pos = (assignment_probs.unsqueeze(-1) * target_features_v[:, :, :3]).sum(dim=1)  # [B, 3]
        assignment_features = torch.cat([assignment_entropy, min_dist, weighted_target_pos], dim=-1)  # [B, 5]

        # When all targets are invalid, zero out assignment_features to avoid misleading control_net
        if target_valid is not None:
            has_valid = target_valid.any(dim=-1, keepdim=True).to(assignment_features.dtype)  # [B, 1]
            assignment_features = assignment_features * has_valid

        return context, assignment_probs, assignment_features

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
        num_heads: int = 4,
        use_quaternion: bool = False,
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
            num_heads=num_heads,
            use_quaternion=use_quaternion
        )

        # Augmented observation dimension: original_obs + context(final_embed_dim) + assignment_feats(5)
        if isinstance(assignment_embed_dim, int):
            final_embed_dim = assignment_embed_dim
        else:
            final_embed_dim = assignment_embed_dim[-1]
        aug_obs_dim = self.num_observations + final_embed_dim + 5

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

        # LayerNorm for context and assignment_feats before concatenation
        self.context_norm = nn.LayerNorm(final_embed_dim)
        self.assignment_feats_norm = nn.LayerNorm(5)

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
        # obs_dim = 6*Kf + 7 + 7*k_target + 3*Kf*k_friend_targetpos
        #         = (6 + 3*k_friend_targetpos)*Kf + (7 + 7*k_target)
        # self state: pos(3) + vel(3) + yaw(1) = 7
        base_dim = 7 + 7 * k_target
        rem = obs_dim - base_dim
        denom = 6 + 3 * k_friend_targetpos

        if rem < 0 or rem % denom != 0:
            raise ValueError(
                f"Observation dimension mismatch! "
                f"obs_dim={obs_dim}, expected format: (6 + 3*obs_k_friend_targetpos)*K_friends + (7 + 7*obs_k_target). "
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
        self_end = self_vel_start + 3 + 1  # pos(3) + vel(3) + yaw(1)
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

        # Extract self state: pos(3) + vel(3) + yaw(1)
        self_pos = states[:, self_pos_start:self_vel_start]          # [B, 3]
        self_vel = states[:, self_vel_start:self_vel_start + 3]      # [B, 3]
        self_yaw = states[:, self_vel_start + 3:self_end]            # [B, 1]

        # Add quaternion if use_quaternion=True
        # NOTE：现在还有没在观测中加入四元数，先不能使用
        if self.use_quaternion and raw_states is not None:
            try:
                self_quat = raw_states[:, -4:]
                self_state = torch.cat([self_pos, self_vel, self_yaw, self_quat], dim=-1)  # [B, 11]
            except:
                self_state = torch.cat([self_pos, self_vel, self_yaw], dim=-1)  # [B, 7]
        else:
            self_state = torch.cat([self_pos, self_vel, self_yaw], dim=-1)  # [B, 7]

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

        # 提取未归一化的相对位置用于距离计算（raw_states 未经 RunningStandardScaler）
        if raw_states is not None:
            raw_target_rel_pos = raw_target_info[:, :, :3]  # [B, E, 3]
            raw_target_rel_pos = raw_target_rel_pos * target_valid.to(raw_target_rel_pos.dtype).unsqueeze(-1)
        else:
            raw_target_rel_pos = target_rel_pos  # fallback：无 raw_states 时只能用归一化版本

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

        return self_state, target_rel_pos, target_rel_vel, target_valid, friend_target_pos, friend_rel_pos, friend_rel_vel, raw_target_rel_pos

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
        self_state, target_rel_pos, target_rel_vel, target_valid, teammate_target_pos, teammate_rel_pos, teammate_rel_vel, raw_target_rel_pos = self._parse_observations(
            states,
            raw_states=raw_states
        )

        # 真实欧氏距离（用于 V 特征，raw_states 未经 RunningStandardScaler）
        target_dist = torch.norm(raw_target_rel_pos, dim=-1, keepdim=True)  # [B, E, 1]

        # 归一化空间距离（与 teammate_target_pos 同一空间，用于协同比较）
        target_dist_norm = torch.norm(target_rel_pos, dim=-1, keepdim=True)  # [B, E, 1]

        # NOTE:这里喂给valuenet的target_dist没有做归一化，发现未做归一化的距离放进去效果更好
        # V 特征：自身视角，7 维 [rel_pos, rel_vel, dist]
        target_features_v = torch.cat([
            target_rel_pos,  # [B, E, 3]
            target_rel_vel,  # [B, E, 3]
            target_dist,     # [B, E, 1]
        ], dim=-1)  # [B, E, 7]

        # K 特征：加入协同信息，12 维
        # teammate_target_pos [B, Kf, E, 3]（归一化空间，与 target_dist_norm 同一空间）
        # 注意：env 端会将冻结队友、无效敌机、padding 槽位的 teammate_target_pos 置零，
        # 导致 teammate_target_dist=0，必须用 mask 排除这些无效槽位，否则会污染聚合统计。
        teammate_target_dist = torch.norm(teammate_target_pos, dim=-1)                    # [B, Kf, E]
        teammate_valid = (teammate_target_dist > 1e-6)                                    # [B, Kf, E] bool
        valid_count = teammate_valid.float().sum(dim=1).unsqueeze(-1).clamp(min=1.0)      # [B, E, 1] 有效队友数

        # team_min_dist：无效槽位填充大值后再 min，避免被零值打成 0
        dist_for_min = teammate_target_dist.masked_fill(~teammate_valid, 1e6)             # [B, Kf, E]
        team_min_dist = dist_for_min.min(dim=1)[0].unsqueeze(-1).clamp(max=50.0)          # [B, E, 1]

        self_dist_norm_exp = target_dist_norm.squeeze(-1).unsqueeze(1)                    # [B, 1, E]
        # closer_fraction：只统计有效队友中比自己更近的比例
        closer_mask = (teammate_target_dist < self_dist_norm_exp) & teammate_valid         # [B, Kf, E]
        closer_fraction = closer_mask.float().sum(dim=1).unsqueeze(-1) / valid_count       # [B, E, 1]

        # === 队友-目标接近速度特征 ===
        # 队友相对目标的速度 = (teammate_vel - ego_vel) - (target_vel - ego_vel) = teammate_vel - target_vel
        teammate_to_target_vel = teammate_rel_vel.unsqueeze(2) - target_rel_vel.unsqueeze(1)  # [B, Kf, E, 3]

        # 从队友指向目标的单位方向向量（无效槽位 dist=0 → direction=0 → closing_rate=0）
        direction = teammate_target_pos / (teammate_target_dist.unsqueeze(-1) + 1e-8)  # [B, Kf, E, 3]

        # 接近速率：正值=队友正在接近目标，负值=远离
        closing_rate = (teammate_to_target_vel * direction).sum(dim=-1)  # [B, Kf, E]

        # 聚合统计（排除无效槽位）
        closing_for_max = closing_rate.masked_fill(~teammate_valid, -1e6)                              # [B, Kf, E]
        team_max_closing = closing_for_max.max(dim=1)[0].unsqueeze(-1).clamp(min=-50.0, max=50.0)      # [B, E, 1]
        approaching_and_valid = (closing_rate > 0) & teammate_valid                                    # [B, Kf, E]
        team_approaching_frac = approaching_and_valid.float().sum(dim=1).unsqueeze(-1) / valid_count   # [B, E, 1]

        # === 队友到达目标的预计时间 (ETA) ===
        # ETA = distance / closing_speed，仅对正在接近的有效队友有意义
        # 无效槽位和远离/静止的队友 ETA 设为极大值
        ETA_MAX = 200.0
        not_approaching = (closing_rate <= 1e-4) | (~teammate_valid)                                   # [B, Kf, E] bool
        closing_speed_safe = closing_rate.clamp(min=1e-4)                                              # [B, Kf, E] 避免除零
        eta = (teammate_target_dist / closing_speed_safe).clamp(max=ETA_MAX)                           # [B, Kf, E]
        eta = eta.masked_fill(not_approaching, ETA_MAX)                                                # 无效/远离/静止 → ETA_MAX

        team_min_eta = eta.min(dim=1)[0].unsqueeze(-1)                                                 # [B, E, 1]

        target_features_k_base = torch.cat([
            target_rel_pos,
            target_rel_vel,
            target_dist_norm,
        ], dim=-1)
        target_features_k = torch.cat([
            target_features_k_base,       # 7: rel_pos(3) + rel_vel(3) + dist_norm(1)
            team_min_dist,                # 1
            closer_fraction,              # 1
            team_max_closing,             # 1
            team_approaching_frac,        # 1
            team_min_eta,                 # 1
        ], dim=-1)  # [B, E, 12]

        # Assignment module: K uses cooperative features, V uses ego-centric features
        context, assignment_probs, assignment_feats = self.assignment_module(
            self_state, target_features_k, target_features_v, target_valid=target_valid
        )

        # Cache assignment probs
        self._assignment_probs_cache = assignment_probs

        # LayerNorm 对全零输入（所有目标无效时 context/assignment_feats 被置零）会产生 NaN，
        # 因为方差为 0 导致除以 0。修复：全零行临时替换为常数向量，LayerNorm 后再清零。
        if target_valid is not None:
            has_valid_ln = target_valid.any(dim=-1).to(context.dtype).unsqueeze(-1)  # [B, 1]
            # 全零行替换为 1.0（任意非零常数，让 LayerNorm 有非零方差）
            context_ln = torch.where(has_valid_ln > 0, context, torch.ones_like(context))
            feats_ln   = torch.where(has_valid_ln > 0, assignment_feats, torch.ones_like(assignment_feats))
            context_normed = self.context_norm(context_ln) * has_valid_ln
            feats_normed   = self.assignment_feats_norm(feats_ln) * has_valid_ln
        else:
            context_normed = self.context_norm(context)
            feats_normed   = self.assignment_feats_norm(assignment_feats)

        # Augment observations: [original_obs, context, assignment_feats]
        aug_obs = torch.cat([states, context_normed, feats_normed], dim=-1)

        # Generate mean actions through control MLP
        mean_actions = self.control_net(aug_obs)

        # Prepare outputs dictionary
        outputs = {
            'assignment_probs': assignment_probs,
            'assignment_entropy': self.get_assignment_entropy(),
            'context': context,
        }

        return mean_actions, self.log_std_parameter, outputs

    def get_assignment_entropy(self, visibility_mask=None) -> torch.Tensor:
        """
        Compute normalized entropy of assignment distribution: -Σ_j P_ij log(P_ij) / log(k_vis).

        Normalized to [0, 1] range regardless of number of visible targets.
        Used primarily for monitoring (diagnosing indecisiveness, switch-rate).

        Args:
            visibility_mask: [B, E] bool tensor, True = target is visible. If None, all targets assumed visible.

        Returns:
            entropy: Scalar tensor with mean normalized entropy across batch
        """
        if self._assignment_probs_cache is None:
            return torch.tensor(0.0, device=self.device)

        probs = self._assignment_probs_cache  # [B, E]
        probs = torch.clamp(probs, min=1e-8, max=1.0)

        if not torch.isfinite(probs).all():
            return torch.tensor(0.0, device=self.device)

        if visibility_mask is not None:
            vis = visibility_mask.to(probs.dtype)
            probs = probs * vis
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            k_vis = vis.sum(dim=-1).clamp(min=1.0)  # [B]
            norm = torch.log(k_vis).clamp(min=1e-6)  # [B]
        else:
            E = probs.shape[-1]
            norm = math.log(max(E, 2))

        entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)  # [B]
        entropy = entropy / norm  # 归一化到 [0, 1]

        if not torch.isfinite(entropy).all():
            return torch.tensor(0.0, device=self.device)

        return entropy.mean()

    def get_entropy_regularization(self, visibility_mask=None) -> torch.Tensor:
        """
        Compute entropy regularization loss (保留用于监控，训练推荐用 get_margin_regularization).

        Args:
            visibility_mask: [B, E] bool tensor, True = target is visible.

        Returns:
            reg_loss: Scalar regularization loss
        """
        entropy = self.get_assignment_entropy(visibility_mask=visibility_mask)
        reg_loss = self.entropy_reg_weight * entropy
        return reg_loss

    def get_margin_regularization(self, visibility_mask=None, margin=0.2, weight=None) -> torch.Tensor:
        """Top1 margin 正则：鼓励最大概率显著领先第二大概率。

        L_margin = weight * ReLU(margin - (p_top1 - p_top2))

        比熵正则更温和：只要求"第一名显著领先第二名"，不压制其余分布的探索性。

        Args:
            visibility_mask: [B, E] bool, True = 可见目标
            margin: 期望的最小领先差距 (default 0.2)
            weight: 正则权重,None 时用 self.entropy_reg_weight

        Returns:
            reg_loss: Scalar regularization loss
        """
        if self._assignment_probs_cache is None:
            return torch.tensor(0.0, device=self.device)

        probs = self._assignment_probs_cache  # [B, E]
        if weight is None:
            weight = self.entropy_reg_weight

        if visibility_mask is not None:
            vis = visibility_mask.to(probs.dtype)
            # 不可见目标概率置零，然后重新归一化
            probs = probs * vis
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            # 可见目标数 < 2 的样本跳过（无法比较 top1 vs top2）
            k_vis = vis.sum(dim=-1)  # [B]
            valid = (k_vis >= 2.0)  # [B] bool
        else:
            valid = torch.ones(probs.shape[0], dtype=torch.bool, device=probs.device)

        if not valid.any():
            return torch.tensor(0.0, device=probs.device)

        # top1, top2
        sorted_probs, _ = probs.sort(dim=-1, descending=True)
        p_top1 = sorted_probs[:, 0]  # [B]
        p_top2 = sorted_probs[:, 1]  # [B]
        gap = p_top1 - p_top2        # [B]

        # ReLU(margin - gap)：gap 不够大时产生惩罚
        loss_per_sample = F.relu(margin - gap)  # [B]

        # 只对有效样本求均值
        loss = (loss_per_sample * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

        return weight * loss

    def get_teacher_guided_margin_reg(self, teacher_probs, visibility_mask=None,
                                      margin=0.2, weight=None) -> torch.Tensor:
        """Teacher-guided margin 正则：老师认为该追的目标，应该在学生分布中领先其余目标。

        与 get_margin_regularization 的区别：
        - 原版推学生自己的 top1 领先 top2，如果 top1 选错了会越推越错
        - 本版用老师推荐的目标，在学生分布中必须至少比其他任何目标高出 margin；否则就按差多少罚多少

        L = weight * ReLU(margin - (p_student[teacher_top1] - max(p_student[others])))

        Args:
            teacher_probs: [B, E] teacher 的分配概率（Sinkhorn 输出）
            visibility_mask: [B, E] bool, True = 可见目标
            margin: 期望的最小领先差距 (default 0.2)
            weight: 正则权重, None 时用 self.entropy_reg_weight

        Returns:
            reg_loss: Scalar regularization loss
        """
        if self._assignment_probs_cache is None:
            return torch.tensor(0.0, device=self.device)

        probs = self._assignment_probs_cache  # [B, E]
        teacher = teacher_probs.detach()       # [B, E] 不回传梯度到 teacher
        if weight is None:
            weight = self.entropy_reg_weight

        if visibility_mask is not None:
            vis = visibility_mask.to(probs.dtype)
            # 不可见目标概率置零，然后重新归一化
            probs = probs * vis
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            # teacher 也做同样的 mask
            teacher = teacher * vis
            teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            # 可见目标数 < 2 的样本跳过
            k_vis = vis.sum(dim=-1)  # [B]
            valid = (k_vis >= 2.0)   # [B] bool
        else:
            valid = torch.ones(probs.shape[0], dtype=torch.bool, device=probs.device)

        if not valid.any():
            return torch.tensor(0.0, device=probs.device)

        # teacher argmax：老师认为该追的目标
        teacher_top1_idx = teacher.argmax(dim=-1)  # [B]

        # 学生在老师推荐目标上的概率
        p_teacher_target = probs.gather(1, teacher_top1_idx.unsqueeze(-1)).squeeze(-1)  # [B]

        # 学生在其余目标中的最大概率（排除老师推荐的那个）
        mask_teacher = torch.zeros_like(probs).scatter_(1, teacher_top1_idx.unsqueeze(-1), 1.0)
        probs_others = probs * (1.0 - mask_teacher)
        p_best_other = probs_others.max(dim=-1)[0]  # [B]

        gap = p_teacher_target - p_best_other  # [B]

        # ReLU(margin - gap)：gap 不够大时产生惩罚
        loss_per_sample = F.relu(margin - gap)  # [B]

        # 只对有效样本求均值
        loss = (loss_per_sample * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

        return weight * loss