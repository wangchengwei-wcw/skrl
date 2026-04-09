# BUG：当蒸馏权重变为零了之后，为什么有很多蒸馏里面的量还在被记录？？？
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import copy
import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.multi_agents.torch import MultiAgent
from skrl.resources.schedulers.torch import KLAdaptiveLR
import math
import numpy as np

# fmt: off
# [start-config-dict-torch]
IPPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class IPPO(MultiAgent):
    class _DistillState(nn.Module):
        _KEYS = [
            "_distill_kl_ema",
            "_distill_kl_ema_prev",
            "_distill_peak_kl",
            "_cached_distill_weight",
            "_distill_entropy_ema",
            "_distill_entropy_ema_prev",
            "_distill_info_gain_ema",
            "_distill_info_gain_baseline",
            "_distill_info_gain_warmup",
            "_distill_intercept_rate_ema",
            "_distill_success_rate_ema",
            "_distill_reward_ema",
            "_distill_reward_ema_scale",
            "_distill_perf_fast",
            "_distill_perf_slow",
            "_distill_perf_best",
            "_margin_reg_update_count",
            "_distill_latest_intercept_rate",
            "_distill_latest_success_rate",
            "_distill_total_timestep",
            "_distill_zero_streak",
            "_distill_permanently_off",
            "_distill_avg_visible_gate",
            "_distill_avg_top1_agreement",
            "_release_triggered",
            "_release_update_count",
            "_release_patience_counter",
        ]

        def __init__(self, agent):
            super().__init__()
            self._agent = agent

        def get_extra_state(self):
            state = {}
            for k in self._KEYS:
                v = getattr(self._agent, k, None)
                # 张量断开图并转 CPU，避免无意义图/设备问题
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu()
                state[k] = v
            return state

        def set_extra_state(self, state):
            if state is None:
                return

            for k in self._KEYS:
                if k not in state:
                    continue
                v = state[k]
                # 如果是 tensor，按当前 agent.device 放回去
                if isinstance(v, torch.Tensor):
                    v = v.to(self._agent.device)
                setattr(self._agent, k, v)

            # 防御性修正：避免旧 ckpt 缺字段时出现非法状态
            if getattr(self._agent, "_cached_distill_weight", None) is None:
                self._agent._cached_distill_weight = self._agent._distill_weight_init
            self._agent._cached_distill_weight = float(
                min(
                    self._agent._distill_weight_init,
                    max(self._agent._distill_weight_min, self._agent._cached_distill_weight)
                )
            )
            if getattr(self._agent, "_distill_total_timestep", None) is None:
                self._agent._distill_total_timestep = 0

    def __init__(
        self,
        possible_agents: Sequence[str],
        models: Mapping[str, Model],
        memories: Optional[Mapping[str, Memory]] = None,
        observation_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        action_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Independent Proximal Policy Optimization (IPPO)

        https://arxiv.org/abs/2011.09533

        :param possible_agents: Name of all possible agents the environment could generate
        :type possible_agents: list of str
        :param models: Models used by the agents.
                       External keys are environment agents' names. Internal keys are the models required by the algorithm
        :type models: nested dictionary of skrl.models.torch.Model
        :param memories: Memories to storage the transitions.
        :type memories: dictionary of skrl.memory.torch.Memory, optional
        :param observation_spaces: Observation/state spaces or shapes (default: ``None``)
        :type observation_spaces: dictionary of int, sequence of int or gymnasium.Space, optional
        :param action_spaces: Action spaces or shapes (default: ``None``)
        :type action_spaces: dictionary of int, sequence of int or gymnasium.Space, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        _cfg = copy.deepcopy(IPPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policies = {uid: self.models[uid].get("policy", None) for uid in self.possible_agents}
        self.values = {uid: self.models[uid].get("value", None) for uid in self.possible_agents}

        for uid in self.possible_agents:
            # checkpoint models
            self.checkpoint_modules[uid]["policy"] = self.policies[uid]
            self.checkpoint_modules[uid]["value"] = self.values[uid]

            # FIXME: might be problematic for parameter sharing
            # broadcast models' parameters in distributed runs
            if config.torch.is_distributed:
                logger.info(f"Broadcasting models' parameters")
                if self.policies[uid] is not None:
                    self.policies[uid].broadcast_parameters()
                    if self.values[uid] is not None and self.policies[uid] is not self.values[uid]:
                        self.values[uid].broadcast_parameters()

        # configuration
        self._param_sharing = self.cfg.get("param_sharing", True)
        self._param_sharing = True  # TODO: Non-parameter-sharing configurations are not supported for now

        self._learning_epochs = self._as_dict(self.cfg["learning_epochs"])
        self._mini_batches = self._as_dict(self.cfg["mini_batches"])
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self._as_dict(self.cfg["grad_norm_clip"])
        self._ratio_clip = self._as_dict(self.cfg["ratio_clip"])
        self._value_clip = self._as_dict(self.cfg["value_clip"])
        self._clip_predicted_values = self._as_dict(self.cfg["clip_predicted_values"])

        self._value_loss_scale = self._as_dict(self.cfg["value_loss_scale"])
        self._entropy_loss_scale = self._as_dict(self.cfg["entropy_loss_scale"])

        self._kl_threshold = self._as_dict(self.cfg["kl_threshold"])

        self._learning_rate = self._as_dict(self.cfg["learning_rate"])
        self._learning_rate_scheduler = self._as_dict(self.cfg["learning_rate_scheduler"])
        self._learning_rate_scheduler_kwargs = self._as_dict(self.cfg["learning_rate_scheduler_kwargs"])

        self._state_preprocessor = self._as_dict(self.cfg["state_preprocessor"])
        self._state_preprocessor_kwargs = self._as_dict(self.cfg["state_preprocessor_kwargs"])
        self._value_preprocessor = self._as_dict(self.cfg["value_preprocessor"])
        self._value_preprocessor_kwargs = self._as_dict(self.cfg["value_preprocessor_kwargs"])

        self._discount_factor = self._as_dict(self.cfg["discount_factor"])
        self._lambda = self._as_dict(self.cfg["lambda"])

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self._as_dict(self.cfg["time_limit_bootstrap"])

        self._mixed_precision = self.cfg["mixed_precision"]

        # distillation configuration
        self._distill_weight_init = self.cfg.get("distill_weight_init", 0.0)            # 初始蒸馏权重
        self._distill_anneal_type = self.cfg.get("distill_anneal_type", "linear")       # 蒸馏退火类型

        # Time-based annealing parameters
        self._distill_anneal_steps = self.cfg.get("distill_anneal_steps", 200000)       # 线性退火的步数
        self._distill_weight_min = self.cfg.get("distill_weight_min", 0.0)              # 最小蒸馏权重，防止完全放手后发散

        # KL-adaptive annealing parameters
        self._distill_target_kl = self.cfg.get("distill_target_kl", 0.1)                # KL-adaptive退火的目标KL散度，低于此值开始退火。学生要"学到多像老师"才开始放手？
        self._distill_kl_ema_momentum = self.cfg.get("distill_kl_ema_momentum", 0.99)   # KL指数移动平均的动量（平滑）。用多平滑的KL来判断"学到位了"？
        self._distill_entropy_ema_momentum = self.cfg.get("distill_entropy_ema_momentum", 0.99)  # Entropy指数移动平均的动量（平滑）。用多平滑的Entropy来判断"学到位了"？
        self._distill_kl_ema = None                                                     # 指数移动平均的KL（每个update只更新一次）
        self._distill_peak_kl = None                                                 # 最高KL（用于kl_ratio）
        self._cached_distill_weight = self._distill_weight_init                         # 缓存的权重，mini-batch中使用

        # ITAD (Information-Theoretic Adaptive Distillation) parameters
        self._distill_info_gain_alpha = self.cfg.get("distill_info_gain_alpha", 0.5)        # KL vs entropy 权重
        self._distill_info_gain_momentum = self.cfg.get("distill_info_gain_momentum", 0.95) # 信息增益 EMA 的动量
        self._distill_kl_ema_prev = None                                                    # 上一次 update 的 kl_ema（用于计算变化率）
        self._distill_entropy_ema = None                                                    # assignment entropy 的 EMA
        self._distill_entropy_ema_prev = None                                               # 上一次的 entropy EMA
        self._distill_info_gain_ema = None                                                  # 信息增益的 EMA
        self._distill_info_gain_baseline = None                                             # 信息增益的基准值（前几次的平均）
        self._distill_info_gain_warmup = 0                                                  # warmup 计数器
        self._distill_info_gain_warmup_n = self.cfg.get("distill_info_gain_warmup_n", 5)    # warmup 次数
        self._distill_v2_warmup_steps = self.cfg.get("distill_v2_warmup_steps", 50000)      # itad_v2 预热步数

        # ITAD退步检测：多指标 + 双EMA
        self._distill_perf_metric_key = self.cfg.get("distill_perf_metric_key", "log")  # infos中的key
        self._distill_perf_fast_momentum = self.cfg.get("distill_perf_fast_momentum", 0.95)   # 快EMA（短期）
        self._distill_perf_slow_momentum = self.cfg.get("distill_perf_slow_momentum", 0.98)   # 慢EMA（长期）
        self._distill_perf_weights = self.cfg.get("distill_perf_weights", {
            "reward": 0.3,
            "interception": 0.5,
            "success": 0.2,
        })
        # 三个指标各自的EMA（用于归一化后加权）
        self._distill_intercept_rate_ema = None
        self._distill_success_rate_ema = None
        self._distill_reward_ema = None
        self._distill_reward_ema_scale = None     # 奖励的scale（用于归一化到[0,1]范围）
        # 复合性能信号的双EMA
        self._distill_perf_fast = None            # 快EMA（跟踪短期）
        self._distill_perf_slow = None            # 慢EMA（跟踪长期趋势）
        self._distill_perf_best = None            # 历史最高（慢EMA的最高值）
        # 从infos累积的最新指标值
        self._distill_latest_intercept_rate = None
        self._distill_latest_success_rate = None

        self._distill_zero_streak = 0           # info_gain_ema 连续接近 0 的 update 次数（用于 cutoff）
        self._distill_permanently_off = False    # 蒸馏权重归零后永久锁死，不再上涨

        # ITAD v3: teacher confidence × student incompetence
        self._distill_avg_visible_gate = None    # 当前 update 的 visible_gate 均值（teacher 信号可执行性）
        self._distill_avg_top1_agreement = None  # 当前 update 的 top1_agreement 均值（student 学习缺口）

        self._optimal_assignment_sorted = None  # [N, M, K_target] from trainer
        self._global_assignment_sorted = None   # [N, M, K_target] global Sinkhorn teacher
        self._local_assignment_sorted = None    # [N, M, K_target] local heuristic teacher

        # ---- dual-teacher distillation parameters ----
        self._dual_teacher_enable = self.cfg.get("dual_teacher_enable", False)
        self._distill_alpha_global = self.cfg.get("distill_alpha_global", 0.6)       # global teacher weight in dual loss
        self._distill_lambda_consistency = self.cfg.get("distill_lambda_consistency", 0.5)  # student-consensus KL weight
        self._distill_eta_consensus = self.cfg.get("distill_eta_consensus", 0.6)     # global weight in consensus teacher

        # ---- release / anneal schedule (single-direction) ----
        self._release_triggered = False          # 是否已触发 release
        self._release_update_count = 0           # 触发后已退火的 update 次数
        self._release_patience_counter = 0       # 连续满足 trigger 条件的 update 次数
        self._release_patience_updates = self.cfg.get("distill_release_patience_updates", 20)
        self._release_success_threshold = self.cfg.get("distill_release_success_threshold", 0.85)
        self._release_interception_threshold = self.cfg.get("distill_release_interception_threshold", 0.85)
        self._release_anneal_updates = self.cfg.get("distill_release_anneal_updates", 200)
        self._release_mode = self.cfg.get("distill_release_mode", "success")
        self._margin_reg_update_count = 0  # margin reg warmup 计数器（按 update 次数）

        self._distill_total_timestep = 0        # 累计训练步数，续训时从 checkpoint 恢复，不随 trainer 重置
        self._distill_last_trainer_timestep = 0 # 上一次 update 时 trainer 的 timestep，用于计算 delta

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        self.optimizers = {}
        self.schedulers = {}

        agent_0 = self.possible_agents[0]
        for uid in self.possible_agents:
            if self._param_sharing and uid != agent_0:
                self.optimizers[uid] = self.optimizers[agent_0]
                self.checkpoint_modules[uid]["optimizer"] = self.optimizers[uid]

                if self._learning_rate_scheduler[uid] is not None:
                    self.schedulers[uid] = self.schedulers[agent_0]

                self._state_preprocessor[uid] = self._state_preprocessor[agent_0]
                if "state_preprocessor" in self.checkpoint_modules[agent_0]:
                    self.checkpoint_modules[uid]["state_preprocessor"] = self.checkpoint_modules[agent_0]["state_preprocessor"]

                self._value_preprocessor[uid] = self._value_preprocessor[agent_0]
                if "value_preprocessor" in self.checkpoint_modules[agent_0]:
                    self.checkpoint_modules[uid]["value_preprocessor"] = self.checkpoint_modules[agent_0]["value_preprocessor"]

                continue

            policy = self.policies[uid]
            value = self.values[uid]
            if policy is not None and value is not None:
                if policy is value:
                    optimizer = torch.optim.Adam(policy.parameters(), lr=self._learning_rate[uid])
                else:
                    optimizer = torch.optim.Adam(
                        itertools.chain(policy.parameters(), value.parameters()), lr=self._learning_rate[uid]
                    )
                self.optimizers[uid] = optimizer
                if self._learning_rate_scheduler[uid] is not None:
                    self.schedulers[uid] = self._learning_rate_scheduler[uid](
                        optimizer, **self._learning_rate_scheduler_kwargs[uid]
                    )

            self.checkpoint_modules[uid]["optimizer"] = self.optimizers[uid]

            # set up preprocessors
            if self._state_preprocessor[uid] is not None:
                self._state_preprocessor[uid] = self._state_preprocessor[uid](**self._state_preprocessor_kwargs[uid])
                self.checkpoint_modules[uid]["state_preprocessor"] = self._state_preprocessor[uid]
            else:
                self._state_preprocessor[uid] = self._empty_preprocessor

            if self._value_preprocessor[uid] is not None:
                self._value_preprocessor[uid] = self._value_preprocessor[uid](**self._value_preprocessor_kwargs[uid])
                self.checkpoint_modules[uid]["value_preprocessor"] = self._value_preprocessor[uid]
            else:
                self._value_preprocessor[uid] = self._empty_preprocessor

        self._distill_state = self._DistillState(self)
        for uid in self.possible_agents:
            self.checkpoint_modules[uid]["distill_state"] = self._distill_state

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memories
        if self.memories:
            for uid in self.possible_agents:
                self.memories[uid].create_tensor(name="states", size=self.observation_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="actions", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="rewards", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="terminated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="truncated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="log_prob", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="values", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="returns", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="advantages", size=1, dtype=torch.float32)

                # tensors sampled during training
                self._tensors_names = ["states", "actions", "log_prob", "values", "returns", "advantages"]

                # distillation: create tensor for optimal assignment if enabled
                if self._distill_weight_init > 0:
                    # num_targets from policy config (default 5)
                    num_targets = 5
                    if hasattr(self.policies[uid], 'num_targets'):
                        num_targets = self.policies[uid].num_targets
                    self._distill_num_targets = num_targets
                    if self._dual_teacher_enable:
                        self.memories[uid].create_tensor(name="global_assignment", size=num_targets, dtype=torch.float32)
                        self.memories[uid].create_tensor(name="local_assignment", size=num_targets, dtype=torch.float32)
                        self._tensors_names = ["states", "actions", "log_prob", "values", "returns", "advantages", "global_assignment", "local_assignment"]
                    else:
                        self.memories[uid].create_tensor(name="optimal_assignment", size=num_targets, dtype=torch.float32)
                        self._tensors_names = ["states", "actions", "log_prob", "values", "returns", "advantages", "optimal_assignment"]

        # Validate distillation configuration
        if self._distill_weight_init > 0:
            logger.info("Distillation enabled, validating configuration...")
            for uid in self.possible_agents:
                policy = self.policies[uid]
                if hasattr(policy, 'num_targets') and hasattr(policy, 'obs_k_target'):
                    num_targets = policy.num_targets
                    obs_k_target = policy.obs_k_target
                    if num_targets != obs_k_target:
                        logger.warning(
                            f"Agent {uid}: num_targets ({num_targets}) != obs_k_target ({obs_k_target}). "
                            f"This may cause suboptimal distillation. Recommend setting both to same value in config."
                        )
                    else:
                        logger.info(f"Agent {uid}: num_targets == obs_k_target == {num_targets} ✓")

        # create temporary variables needed for storage and computation
        self._current_log_prob = []
        self._current_next_states = []

    def _update_distill_ema(self, avg_raw_kl: float, avg_entropy: float = None,
                            avg_visible_gate: float = None, avg_top1_agreement: float = None):
        """
        # EMA 就是 Exponential Moving Average,中文一般叫 指数滑动平均 或 指数加权平均。它是一种对时间序列数据进行平滑处理的方法，能够更稳定地跟踪数据的趋势，减少噪声的影响。
        每个update结束时调用一次,也就是收集完一次rollout数据后调用一次。
        1.更新 KL 的 EMA(teacher vs student 的分配差距),KL EMA大说明学生和老师的差距大
        2.在 ITAD 模式下更新 分配熵的 EMA,熵ema高说明student 的 assignment_probs 更均匀、更犹豫,低说明student 的分配更尖锐、更确定
        3.用 KL 与熵的"下降速率"计算 信息增益 info_gain,再对它做 EMA,并在 warmup 阶段估计 baseline

        Args:
            avg_raw_kl: 本次 update 中所有 mini-batch 的 KL 平均值（老师和学生的差距）
            avg_entropy: 本次 update 的 assignment entropy,越大说明学生的分配越均匀（不确定），越小说明学生的分配越尖锐（确定）。这个值在 ITAD 模式下才有意义。
        """
        if self._distill_weight_init <= 0:
            return

        # 存储 update 级别的 visible_gate 和 top1_agreement（用于 ITAD v3 权重计算）
        if avg_visible_gate is not None:
            self._distill_avg_visible_gate = avg_visible_gate
        if avg_top1_agreement is not None:
            self._distill_avg_top1_agreement = avg_top1_agreement

        # 记录KL的历史最大值（用于 hybrid 模式的 kl_ratio 归一化，比初始值更稳定）
        if self._distill_peak_kl is None:
            self._distill_peak_kl = avg_raw_kl
        else:
            self._distill_peak_kl = max(self._distill_peak_kl, avg_raw_kl)

        # 保存上一次的 kl_ema（用于 ITAD 计算变化率）
        self._distill_kl_ema_prev = self._distill_kl_ema

        # 更新KL的指数移动平均（每个update只更新一次）
        # ema = momentum * ema + (1 - momentum) * KL
        if self._distill_kl_ema is None:
            self._distill_kl_ema = avg_raw_kl
        else:
            self._distill_kl_ema = (
                self._distill_kl_ema_momentum * self._distill_kl_ema +
                (1 - self._distill_kl_ema_momentum) * avg_raw_kl
            )

        # ITAD: 更新 entropy EMA 和信息增益
        if self._distill_anneal_type in ("itad", "itad_v2") and avg_entropy is not None:
            # 1) 拦截率EMA
            if self._distill_latest_intercept_rate is not None:
                v = self._distill_latest_intercept_rate
                if self._distill_intercept_rate_ema is None:
                    self._distill_intercept_rate_ema = v
                else:
                    self._distill_intercept_rate_ema = 0.95 * self._distill_intercept_rate_ema + 0.05 * v

            # 2) 成功率EMA（ITAD 路径）
            if self._distill_latest_success_rate is not None:
                v = self._distill_latest_success_rate
                if self._distill_success_rate_ema is None:
                    self._distill_success_rate_ema = v
                else:
                    self._distill_success_rate_ema = 0.95 * self._distill_success_rate_ema + 0.05 * v

        # dual-teacher release 路径：success_rate_ema 需要在所有 anneal_type 下更新
        if self._dual_teacher_enable and self._distill_latest_success_rate is not None:
            v = self._distill_latest_success_rate
            if self._distill_success_rate_ema is None:
                self._distill_success_rate_ema = v
            else:
                self._distill_success_rate_ema = 0.95 * self._distill_success_rate_ema + 0.05 * v

        # ITAD: 奖励EMA 和复合性能信号（仅 itad/itad_v2 路径）
        if self._distill_anneal_type in ("itad", "itad_v2") and avg_entropy is not None:
            # 3) 奖励EMA（归一化到[0,1]）
            if len(self._track_rewards) > 0:
                raw_reward = float(np.mean(self._track_rewards))
                if self._distill_reward_ema is None:
                    self._distill_reward_ema = raw_reward
                    self._distill_reward_ema_scale = max(abs(raw_reward), 1e-6)
                else:
                    self._distill_reward_ema = 0.95 * self._distill_reward_ema + 0.05 * raw_reward
                    self._distill_reward_ema_scale = max(self._distill_reward_ema_scale, abs(raw_reward))

            # 4) 计算复合性能信号（加权，归一化到[0,1]范围）
            w = self._distill_perf_weights
            components = []
            total_w = 0.0
            if self._distill_intercept_rate_ema is not None:
                components.append(w.get("interception", 0.5) * self._distill_intercept_rate_ema)
                total_w += w.get("interception", 0.5)
            if self._distill_success_rate_ema is not None:
                components.append(w.get("success", 0.2) * self._distill_success_rate_ema)
                total_w += w.get("success", 0.2)
            if self._distill_reward_ema is not None and self._distill_reward_ema_scale is not None:
                norm_reward = 1.0 / (1.0 + math.exp(-self._distill_reward_ema / self._distill_reward_ema_scale))
                components.append(w.get("reward", 0.3) * norm_reward)
                total_w += w.get("reward", 0.3)

            if total_w > 0 and len(components) > 0:
                composite_perf = sum(components) / total_w

                # 5) 双EMA更新
                fast_m = self._distill_perf_fast_momentum
                slow_m = self._distill_perf_slow_momentum
                if self._distill_perf_fast is None:
                    self._distill_perf_fast = composite_perf
                    self._distill_perf_slow = composite_perf
                else:
                    self._distill_perf_fast = fast_m * self._distill_perf_fast + (1 - fast_m) * composite_perf      # 短期内性能的变化
                    self._distill_perf_slow = slow_m * self._distill_perf_slow + (1 - slow_m) * composite_perf      # 长期性能的趋势

                # 6) 只在warmup结束后跟踪历史最高（用慢EMA）
                if self._distill_info_gain_warmup > self._distill_info_gain_warmup_n:
                    if self._distill_perf_best is None:
                        self._distill_perf_best = self._distill_perf_slow
                    else:
                        best_decay = self.cfg.get("distill_perf_best_decay", 0.999)
                        self._distill_perf_best = max(self._distill_perf_slow, best_decay * self._distill_perf_best)

            # 7) 更新熵的EMA（student 的分配不确定性）
            self._distill_entropy_ema_prev = self._distill_entropy_ema
            if self._distill_entropy_ema is None:
                self._distill_entropy_ema = avg_entropy
            else:
                self._distill_entropy_ema = (
                    self._distill_entropy_ema_momentum * self._distill_entropy_ema +
                    (1 - self._distill_entropy_ema_momentum) * avg_entropy
                )

            # 计算信息增益
            if self._distill_kl_ema_prev is not None and self._distill_entropy_ema_prev is not None:
                eps = 1e-8
                # KL 变化率：正值 = KL 在下降 = student 在学习
                # teacher-student KL 的相对下降率
                    # kl_rate > 0：KL 在下降，student 更像 teacher
                    # kl_rate = 0：基本没进展
                    # kl_rate < 0：KL 在升高，student 反而更不像 teacher
                delta_kl = self._distill_kl_ema_prev - self._distill_kl_ema # delta_kl 为正值：KL 下降（student 更像 teacher）
                kl_rate = delta_kl / max(self._distill_kl_ema_prev, eps)    # 相对下降率

                # 熵变化率：正值 = 熵在下降 = 决策更确定
                # student assignment entropy 的相对下降率
                    # entropy_rate > 0：熵下降，student 更确定
                    # entropy_rate = 0：确定性没怎么变
                    # entropy_rate < 0：熵上升，student 更犹豫了
                delta_h = self._distill_entropy_ema_prev - self._distill_entropy_ema
                entropy_rate = delta_h / max(self._distill_entropy_ema_prev, eps)

                # 信息增益 = 加权和
                # alpha 越大 → 更相信 KL 的下降（更像 teacher）
                # alpha 越小 → 更相信 熵 的下降（更确定）
                # info_gain ：
                    # 大且为正：teacher 还在持续带来帮助（学生在变像、变确定）
                    # 接近 0：teacher 边际贡献变小
                    # 为负：整体在退步（不像 teacher 了/更犹豫了）
                alpha = self._distill_info_gain_alpha
                info_gain = alpha * kl_rate + (1 - alpha) * entropy_rate

                # 更新信息增益 EMA
                if self._distill_info_gain_ema is None:
                    self._distill_info_gain_ema = info_gain
                else:
                    self._distill_info_gain_ema = (
                        self._distill_info_gain_momentum * self._distill_info_gain_ema +
                        (1 - self._distill_info_gain_momentum) * info_gain
                    )

                # baseline 是用来回答：当前的 teacher 帮助，到底还算不算“明显”
                # warmup 后继续用慢速 EMA 跟踪 baseline，防止 baseline 过时
                self._distill_info_gain_warmup += 1
                success_ema_threshold = self.cfg.get("distill_info_gain_success_ema_threshold", 0.0)
                success_ema_ok = (
                    self._distill_success_rate_ema is not None and
                    self._distill_success_rate_ema >= success_ema_threshold
                ) if success_ema_threshold > 0.0 else True
                # warmup 结束条件：次数满 AND success_ema 达标，两者同时满足才固定 baseline
                # 任意一个不满足就继续收集
                warmup_not_done = (
                    self._distill_info_gain_warmup <= self._distill_info_gain_warmup_n or
                    not success_ema_ok
                )
                if warmup_not_done:
                    if self._distill_info_gain_baseline is None:
                        self._distill_info_gain_baseline = abs(info_gain)
                    else:
                        # 用累计平均绝对值作为 baseline，warmup 结束后固定不再更新
                        n = self._distill_info_gain_warmup
                        self._distill_info_gain_baseline = (
                            self._distill_info_gain_baseline * (n - 1) + abs(info_gain)
                        ) / n

    def _compute_distill_weight(self) -> float:
        """纯函数:根据当前EMA状态计算蒸馏权重,不修改任何状态。

        Args:
            timestep: 当前训练步数

        Returns:
            蒸馏权重
        """
        if self._distill_weight_init <= 0:
            return 0.0

        if self._distill_anneal_type == "kl_adaptive":
            if self._distill_kl_ema is None:
                return self._distill_weight_init
            if self._distill_kl_ema > self._distill_target_kl:
                return self._distill_weight_init
            else:
                progress = (self._distill_target_kl - self._distill_kl_ema) / self._distill_target_kl
                progress = max(0.0, min(1.0, progress))
                return self._distill_weight_init * max(self._distill_weight_min, 1.0 - progress)
        elif self._distill_anneal_type == "itad":
            success_ema_threshold = self.cfg.get("distill_info_gain_success_ema_threshold", 0.0)
            success_ema_not_ready = (
                success_ema_threshold > 0.0 and (
                    self._distill_success_rate_ema is None or
                    self._distill_success_rate_ema < success_ema_threshold
                )
            )
            if (self._distill_avg_visible_gate is None or
                    self._distill_avg_top1_agreement is None or
                    success_ema_not_ready):
                return self._distill_weight_init
            return self._compute_itad_weight()
        elif self._distill_anneal_type == "itad_v2":
            # Phase 1: ramp up from weight_min to weight_init
            warmup_steps = self.cfg.get("distill_v2_warmup_steps", 1000)
            if self._distill_total_timestep < warmup_steps:
                progress = self._distill_total_timestep / max(warmup_steps, 1)
                return self._distill_weight_min + (
                    self._distill_weight_init - self._distill_weight_min
                ) * progress

            # Phase 2: wait until ITAD statistics are ready
            success_ema_threshold = self.cfg.get("distill_info_gain_success_ema_threshold", 0.0)
            success_ema_not_ready = (
                success_ema_threshold > 0.0 and (
                    self._distill_success_rate_ema is None or
                    self._distill_success_rate_ema < success_ema_threshold
                )
            )
            if (self._distill_avg_visible_gate is None or
                    self._distill_avg_top1_agreement is None or
                    success_ema_not_ready):
                return self._distill_weight_init
            if self._distill_kl_ema is None:
                return self._distill_weight_init

            plateau_steps = self.cfg.get("distill_v2_plateau_steps", 10000)
            if self._distill_total_timestep < (plateau_steps + warmup_steps):
                return self._distill_weight_init
            # perf_ready = (
            #     self._distill_intercept_rate_ema is not None and
            #     self._distill_intercept_rate_ema >= 0.15
            # )

            # if not perf_ready:
            #     return self._distill_weight_init

            # if self._distill_kl_ema > self._distill_target_kl or not perf_ready:
            #     return self._distill_weight_init

            # Phase 4: once KL is low enough, hand over to ITAD
            return self._compute_itad_weight()
        elif self._distill_anneal_type == "hybrid":
            time_factor = max(0.0, 1.0 - self._distill_total_timestep / self._distill_anneal_steps)
            if self._distill_kl_ema is not None and self._distill_peak_kl is not None:
                kl_factor = min(1.0, self._distill_kl_ema / max(self._distill_peak_kl, 1e-8))
                kl_factor = max(self._distill_weight_min, kl_factor)
            else:
                kl_factor = 1.0
            return self._distill_weight_init * min(time_factor, kl_factor)
        elif self._distill_anneal_type == "linear":
            progress = min(self._distill_total_timestep / self._distill_anneal_steps, 1.0)
            return self._distill_weight_init * max(0.0, 1.0 - progress)
        elif self._distill_anneal_type == "fixed":
            return self._distill_weight_init
        else:
            raise ValueError(f"Unknown distill_anneal_type: {self._distill_anneal_type}")

    def _compute_itad_weight(self) -> float:
        """itad 核心退火逻辑（不含预热/warmup 判断，由调用方负责）。"""
        # 永久锁死：agreement 达到阈值后不再上涨
        if self._distill_permanently_off:
            return self._distill_weight_min

        # 信号未就绪时保持初始权重
        if self._distill_avg_visible_gate is None or self._distill_avg_top1_agreement is None:
            return self._distill_weight_init

        # ITAD 核心公式：w = w_init × (1 - agreement) × visible_gate
        #   visible_gate：teacher 信号在可见目标上的概率质量（teacher 信号可执行性）
        #   1 - agreement：student 的学习缺口（agreement 越高说明学生越学会了，蒸馏价值越低）
        visible_gate = max(0.0, min(1.0, self._distill_avg_visible_gate))
        agreement    = max(0.0, min(1.0, self._distill_avg_top1_agreement))
        incompetence = 1.0 - agreement
        # w_anneal = self._distill_weight_init * incompetence * visible_gate
        w_anneal = self._distill_weight_init * incompetence

        # Cutoff：agreement 连续 N 次超过阈值则永久关闭蒸馏
        agreement_cutoff = self.cfg.get("distill_itad_agreement_cutoff", 0.95)
        streak_n         = self.cfg.get("distill_itad_zero_streak_n", 10)
        if agreement >= agreement_cutoff:
            self._distill_zero_streak += 1
        else:
            self._distill_zero_streak = 0
        # if self._distill_zero_streak >= streak_n:
        #     self._distill_permanently_off = True
        #     print(f"ITAD: 蒸馏权重永久关闭 (agreement={agreement:.4f} >= {agreement_cutoff} 连续 {streak_n} 次)")
        #     return self._distill_weight_min

        w_base = w_anneal

        prev = self._cached_distill_weight
        # target_weight = min(self._distill_weight_init, w_base + restore_strength)
        target_weight = min(self._distill_weight_init, w_base)
        # print(f"ITAD final weight: w_base={w_base:.6f}, restore_strength={restore_strength:.6f}, target_weight={target_weight:.6f}, prev_weight={prev:.6f}")
        max_down = self.cfg.get("distill_itad_max_down_delta", 0.001) * self._distill_weight_init
        max_up   = self.cfg.get("distill_itad_max_up_delta",   0.002) * self._distill_weight_init

        delta = target_weight - prev
        # 只限制下降速度，上升不限制（快速响应 teacher 信号变好或 student 退步）
        delta = max(-max_down, delta)
        result = min(self._distill_weight_init, max(self._distill_weight_min, prev + delta))
        # 权重降到 min 后永久锁死
        # if result <= self._distill_weight_min:
        #     self._distill_permanently_off = True
        return result

    def _update_release_schedule(self) -> float:
        """单向 release 退火调度（每次 update 调用一次）。

        三阶段：
        1. warmup：beta = beta_init，固定不变
        2. trigger 后：按 update 线性退火到 0
        3. 退火结束：permanently_off=True，永久关闭

        Returns:
            当前 beta 值
        """
        if self._distill_weight_init <= 0:
            return 0.0

        # 已永久关闭
        if self._distill_permanently_off:
            return 0.0

        # 已触发 release → 线性退火
        if self._release_triggered:
            self._release_update_count += 1
            progress = min(1.0, self._release_update_count / max(self._release_anneal_updates, 1))
            beta = self._distill_weight_init * (1.0 - progress)
            if progress >= 1.0:
                self._distill_permanently_off = True
                beta = 0.0
            return beta
        
        if self._release_mode is not None and self._release_mode == "success":
            # 检查 trigger 条件（基于 success_rate_ema）
            success_rate = self._distill_success_rate_ema
            if success_rate is not None and success_rate >= self._release_success_threshold:
                self._release_patience_counter += 1
            else:
                self._release_patience_counter = 0

            if self._release_patience_counter >= self._release_patience_updates:
                self._release_triggered = True
                self._release_update_count = 0
                logger.info(
                    f"[DualTeacher] Release triggered! success_rate_ema={success_rate:.4f} >= "
                    f"{self._release_success_threshold} for {self._release_patience_updates} updates. "
                    f"Starting linear anneal over {self._release_anneal_updates} updates."
                )
        elif self._release_mode is not None and self._release_mode == "interception":
            # 检查 trigger 条件（基于 interception_rate_ema）
            interception_rate = self._distill_intercept_rate_ema
            if interception_rate is not None and interception_rate >= self._release_interception_threshold:
                self._release_patience_counter += 1
            else:
                self._release_patience_counter = 0

            if self._release_patience_counter >= self._release_patience_updates:
                self._release_triggered = True
                self._release_update_count = 0
                logger.info(
                    f"[DualTeacher] Release triggered! interception_rate_ema={interception_rate:.4f} >= "
                    f"{self._release_success_threshold} for {self._release_patience_updates} updates. "
                    f"Starting linear anneal over {self._release_anneal_updates} updates."
                )

        return self._distill_weight_init

    def _extract_visibility_mask(self, policy, raw_states, target_dim):
        """从 raw_states 提取 visibility_mask [B, target_dim]。

        复用蒸馏 loss 和 margin reg 中的可见性提取逻辑。

        Args:
            policy: 策略模型（需要 obs_k_target, obs_k_friend_targetpos 属性）
            raw_states: [B, obs_dim] 未归一化的观测
            target_dim: 期望的目标维度（用于校验 lock_indices 数量）

        Returns:
            visibility_mask: [B, target_dim] bool tensor, True = 可见目标。None if 无法提取。
        """
        if not (hasattr(policy, 'obs_k_target') and hasattr(policy, 'obs_k_friend_targetpos')):
            return None

        k_target = policy.obs_k_target
        k_friend_targetpos = policy.obs_k_friend_targetpos
        obs_dim = raw_states.shape[-1]
        base_dim = 7 + 7 * k_target
        rem = obs_dim - base_dim
        denom_obs = 6 + 3 * k_friend_targetpos

        if rem >= 0 and denom_obs > 0 and rem % denom_obs == 0:
            k_friends_actual = rem // denom_obs
            target_start = 6 * k_friends_actual + 7
            lock_indices = list(range(target_start + 6, target_start + 7 * k_target, 7))
            if len(lock_indices) == target_dim:
                lock_flags = raw_states[:, lock_indices]  # [B, target_dim]
                return (lock_flags > 0.5)  # [B, target_dim] bool
        return None

    def _compute_visible_mass_gate(self, teacher_probs, visibility_mask):
        """根据 teacher 在可见集合上的概率质量计算蒸馏门控。

        Args:
            teacher_probs: [B, K]，已经归一化后的 teacher 分布
            visibility_mask: [B, K] bool / 0-1 mask，True 表示目标可见

        Returns:
            gate: [B]，取值在 [0, 1]，teacher 在可见集合上的总概率质量
            has_visible: [B] bool，当前样本是否存在 teacher 有质量的可见目标
        """
        if visibility_mask is None:
            gate = torch.ones(teacher_probs.shape[0], device=teacher_probs.device, dtype=teacher_probs.dtype)
            has_visible = torch.ones(teacher_probs.shape[0], device=teacher_probs.device, dtype=torch.bool)
            return gate, has_visible

        vis_f = visibility_mask.to(teacher_probs.dtype)
        # has_visible: 该样本是否有至少一个可见目标（基于 mask 本身，不依赖 teacher 概率质量）
        # 这样即使 teacher 全零（训练初期 Sinkhorn 未收敛），有可见目标的样本仍参与蒸馏
        has_visible = visibility_mask.any(dim=-1)  # [B] bool
        visible_mass = (teacher_probs * vis_f).sum(dim=-1).clamp(min=0.0, max=1.0)  # [B] gate
        return visible_mass.detach(), has_visible

    def act(self, states: Mapping[str, torch.Tensor], timestep: int, timesteps: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Process the environment's states to make a decision (actions) using the main policies

        :param states: Environment's states
        :type states: dictionary of torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        """
        # # sample random actions
        # # TODO: fix for stochasticity, rnn and log_prob
        # if timestep < self._random_timesteps:
        #     return self.policy.random_act({"states": states}, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            data = []
            for uid in self.possible_agents:
                pre_states = self._state_preprocessor[uid](states[uid])
                data.append(
                    self.policies[uid].act(
                        {"states": pre_states, "raw_states": states[uid]},
                        role="policy",
                    )
                )

            actions = {uid: d[0] for uid, d in zip(self.possible_agents, data)}
            log_prob = {uid: d[1] for uid, d in zip(self.possible_agents, data)}
            outputs = {uid: d[2] for uid, d in zip(self.possible_agents, data)}

            self._current_log_prob = log_prob

            # Cache assignment_probs for reward computation
            # Collect assignment_probs from all agents and stack into [N, M, E] tensor
            self._current_assignment_probs = {}
            assignment_probs_list = []
            for uid in self.possible_agents:
                if 'assignment_probs' in outputs[uid]:
                    self._current_assignment_probs[uid] = outputs[uid]['assignment_probs']
                    assignment_probs_list.append(outputs[uid]['assignment_probs'])

            # Stack into [N, M, E] if available
            if assignment_probs_list:
                # assignment_probs_list: list of [N, E] tensors, one per agent
                # Stack along dim=1 to get [N, M, E]
                self._stacked_assignment_probs = torch.stack(assignment_probs_list, dim=1)
            else:
                self._stacked_assignment_probs = None

        return actions, log_prob, outputs

    def record_transition(
        self,
        states: Mapping[str, torch.Tensor],
        actions: Mapping[str, torch.Tensor],
        rewards: Mapping[str, torch.Tensor],
        next_states: Mapping[str, torch.Tensor],
        terminated: Mapping[str, torch.Tensor],
        truncated: Mapping[str, torch.Tensor],
        infos: Mapping[str, Any],
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: dictionary of torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: dictionary of torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: dictionary of torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: dictionary of torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: dictionary of torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: dictionary of torch.Tensor
        :param infos: Additional information about the environment
        :type infos: dictionary of any supported type
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        # 从infos中提取任务指标（只在episode reset时有值）
        _metric_key = self._distill_perf_metric_key
        if _metric_key in infos:
            _log = infos[_metric_key]
            if "Episode_Metric/Interception_Rate" in _log:
                v = _log["Episode_Metric/Interception_Rate"]
                self._distill_latest_intercept_rate = v.item() if isinstance(v, torch.Tensor) else float(v)
            if "Episode_Metric/Success_Rate" in _log:
                v = _log["Episode_Metric/Success_Rate"]
                self._distill_latest_success_rate = v.item() if isinstance(v, torch.Tensor) else float(v)

        if self.memories:
            self._current_next_states = next_states

            for uid in self.possible_agents:
                # reward shaping
                if self._rewards_shaper is not None:
                    rewards[uid] = self._rewards_shaper(rewards[uid], timestep, timesteps)

                # compute values
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    values, _, _ = self.values[uid].act(
                        {"states": self._state_preprocessor[uid](states[uid])}, role="value"
                    )
                    values = self._value_preprocessor[uid](values, inverse=True)

                # time-limit (truncation) bootstrapping
                if self._time_limit_bootstrap[uid]:
                    rewards[uid] += self._discount_factor[uid] * values * truncated[uid]

                # storage transition in memory
                memory_samples = {
                    "states": states[uid],
                    "actions": actions[uid],
                    "rewards": rewards[uid],
                    "next_states": next_states[uid],
                    "terminated": terminated[uid],
                    "truncated": truncated[uid],
                    "log_prob": self._current_log_prob[uid],
                    "values": values,
                }

                # distillation: store optimal assignment per agent
                if self._distill_weight_init > 0:
                    agent_idx = self.possible_agents.index(uid)
                    if self._dual_teacher_enable:
                        # 始终写入（即使 teacher 是 None 也写全零），避免 memory 初始 NaN 残留
                        N = list(states.values())[0].shape[0]
                        K = self._distill_num_targets
                        dev = list(states.values())[0].device
                        g_slice = (self._global_assignment_sorted[:, agent_idx, :]
                                   if self._global_assignment_sorted is not None
                                   else torch.zeros(N, K, device=dev))
                        l_slice = (self._local_assignment_sorted[:, agent_idx, :]
                                   if self._local_assignment_sorted is not None
                                   else torch.zeros(N, K, device=dev))
                        memory_samples["global_assignment"] = g_slice
                        memory_samples["local_assignment"] = l_slice
                    elif self._optimal_assignment_sorted is not None:
                        memory_samples["optimal_assignment"] = self._optimal_assignment_sorted[:, agent_idx, :]

                self.memories[uid].add_samples(**memory_samples)

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        sampled_batches = {}
        value_list, return_list = [], []
        for uid in self.possible_agents:
            value = self.values[uid]
            memory = self.memories[uid]

            # compute returns and advantages
            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                value.train(False)
                last_values, _, _ = value.act(
                    {"states": self._state_preprocessor[uid](self._current_next_states[uid].float())}, role="value"
                )
                value.train(True)
            last_values = self._value_preprocessor[uid](last_values, inverse=True)

            values = memory.get_tensor_by_name("values")
            returns, advantages = compute_gae(
                rewards=memory.get_tensor_by_name("rewards"),
                dones=memory.get_tensor_by_name("terminated") | memory.get_tensor_by_name("truncated"),
                values=values,
                discount_factor=self._discount_factor[uid],
                lambda_coefficient=self._lambda[uid],
            )

            # collect values and returns before preprocessing and value clipping for explained variance
            value_list.append(values.detach().reshape(-1))
            return_list.append(returns.detach().reshape(-1))

            memory.set_tensor_by_name("values", self._value_preprocessor[uid](values, train=True))
            memory.set_tensor_by_name("returns", self._value_preprocessor[uid](returns, train=True))
            memory.set_tensor_by_name("advantages", advantages)

            # sample mini-batches from memory
            sampled_batches[uid] = memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches[uid])

        # compute explained variance
        # adapted from https://github.com/DLR-RM/stable-baselines3
        v_pred = torch.cat(value_list).reshape(-1)      # critic 预测的 value
        v_true = torch.cat(return_list).reshape(-1)     # GAE 算出来的 returns
        var_v = torch.var(v_true, unbiased=False)
        explained_var = float("nan") if var_v == 0 else (1 - torch.var(v_true - v_pred, unbiased=False) / var_v).item() # critic 对 returns 的解释程度

        # merge mini-batches from all agents by concatenating along batch dimension
        # adapted from https://github.com/jackzeng-robotics/skrl
        agent_0 = self.possible_agents[0]
        merged_batches = [[] for _ in range(self._mini_batches[agent_0])]
        for uid in self.possible_agents:
            for batch_idx, batch in enumerate(sampled_batches[uid]):
                if not merged_batches[batch_idx]:  
                    merged_batches[batch_idx] = list(batch)  # shape of batch: [batch_size, ...]
                else:
                    for tensor_idx, tensor in enumerate(batch):
                        merged_batches[batch_idx][tensor_idx] = torch.cat((merged_batches[batch_idx][tensor_idx], tensor), dim=0)   # batch dimension: dim=0
        sampled_batches_all = [tuple(batch) for batch in merged_batches]

        policy = self.policies[agent_0]
        value = self.values[agent_0]
        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0
        cumulative_value_loss = 0.0
        cumulative_kl_divergence = 0.0
        cumulative_assignment_reg_loss = 0.0
        cumulative_distill_loss = 0.0
        cumulative_raw_kl = 0.0               # 原始KL散度（未加权）
        cumulative_visible_gate = 0.0         # 可见性门控均值（teacher 在可见集合上的概率质量）
        cumulative_top1_agreement = 0.0       # teacher 和 student 的 top1 一致率
        cumulative_js_teacher = 0.0           # teacher-teacher JS divergence（仅日志，不反传）
        distill_count = 0                   # 实际计算了KL的batch数
        _entropy_accum = None
        _entropy_accum_count = 0
        clip_fractions = []
        has_assignment_reg = hasattr(policy, "get_margin_regularization")

        # learning epochs
        for epoch in range(self._learning_epochs[agent_0]):
            kl_divergences = []

            # mini-batches loop
            for batch in sampled_batches_all:

                sampled_optimal_assignment = None
                sampled_global_assignment = None
                sampled_local_assignment = None
                if self._distill_weight_init > 0 and self._dual_teacher_enable and len(batch) == 8:
                    (
                        sampled_states,
                        sampled_actions,
                        sampled_log_prob,
                        sampled_values,
                        sampled_returns,
                        sampled_advantages,
                        sampled_global_assignment,
                        sampled_local_assignment,
                    ) = batch
                elif self._distill_weight_init > 0 and len(batch) == 7:
                    (
                        sampled_states,
                        sampled_actions,
                        sampled_log_prob,
                        sampled_values,
                        sampled_returns,
                        sampled_advantages,
                        sampled_optimal_assignment,
                    ) = batch
                else:
                    (
                        sampled_states,
                        sampled_actions,
                        sampled_log_prob,
                        sampled_values,
                        sampled_returns,
                        sampled_advantages,
                    ) = batch

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    raw_states = sampled_states     #存储未归一化的观测
                    sampled_states = self._state_preprocessor[agent_0](sampled_states, train=not epoch)

                    policy_inputs = {"states": sampled_states, "raw_states": raw_states, "taken_actions": sampled_actions}
                    _, next_log_prob, _ = policy.act(policy_inputs, role="policy")

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self._kl_threshold[agent_0] and kl_divergence > self._kl_threshold[agent_0]:
                        break

                    # compute entropy loss
                    if self._entropy_loss_scale[agent_0]:
                        entropy_loss = -self._entropy_loss_scale[agent_0] * policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute assignment margin regularization
                    assignment_reg_loss = 0
                    if has_assignment_reg:
                        try:
                            vis_mask = self._extract_visibility_mask(
                                policy, raw_states,
                                getattr(policy, 'num_targets', 5)
                            )

                            # margin 正则：独立于蒸馏权重
                            margin_m = self.cfg.get("assignment_margin", 0.2)
                            margin_w = self.cfg.get("assignment_margin_weight", None)  # None = 用 entropy_reg_weight

                            # if sampled_optimal_assignment is not None and hasattr(policy, 'get_teacher_guided_margin_reg'):
                            #     # teacher-guided：推老师认为该追的目标在学生分布中领先
                            #     assignment_reg_loss = policy.get_teacher_guided_margin_reg(
                            #         teacher_probs=sampled_optimal_assignment,
                            #         visibility_mask=vis_mask,
                            #         margin=margin_m,
                            #         weight=margin_w,
                            #     )
                            # else:
                            #     # fallback：无 teacher 时用原始 top1 vs top2
                            #     assignment_reg_loss = policy.get_margin_regularization(
                            #         visibility_mask=vis_mask,
                            #         margin=margin_m,
                            #         weight=margin_w,
                            #     )
                            assignment_reg_loss = policy.get_margin_regularization(
                                visibility_mask=vis_mask,
                                margin=margin_m,
                                weight=margin_w,
                            )

                            # 独立 warmup schedule：前 N 次 update 不施加 reg，之后线性 ramp up
                            # FIXME：到什么时候才开始施加正则比较好？一开始学生都不会控制自己。
                            reg_warmup = self.cfg.get("assignment_reg_warmup_updates", 0)
                            if reg_warmup > 0 and self._margin_reg_update_count < reg_warmup:
                                reg_scale = self._margin_reg_update_count / reg_warmup
                                assignment_reg_loss = reg_scale * assignment_reg_loss
                        except Exception:
                            assignment_reg_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip[agent_0], 1.0 + self._ratio_clip[agent_0]
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute clip fraction
                    batch_clip_fraction = (torch.abs(ratio - 1.0) > self._ratio_clip[agent_0]).float().mean()
                    clip_fractions.append(batch_clip_fraction.detach())

                    # compute value loss
                    predicted_values, _, _ = value.act({"states": sampled_states}, role="value")

                    if self._clip_predicted_values[agent_0]:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip[agent_0], max=self._value_clip[agent_0]
                        )
                    value_loss = self._value_loss_scale[agent_0] * F.mse_loss(sampled_returns, predicted_values)

                    # ---- distillation loss ----
                    distill_loss = 0
                    raw_kl_val = torch.tensor(0.0)
                    batch_visible_gate = None
                    batch_top1_agreement = 0.0
                    valid_samples = None

                    if self._distill_weight_init > 0 and self._cached_distill_weight > 0:
                        assignment_probs = getattr(policy, '_assignment_probs_cache', None)
                        if assignment_probs is not None:
                            pred = assignment_probs  # [B, E]

                            if self._dual_teacher_enable and sampled_global_assignment is not None and sampled_local_assignment is not None:
                                # ---- dual-teacher distillation ----
                                def _align_and_norm(t, E):
                                    """Align teacher tensor to student dim E and normalize.
                                    Zero rows (no valid targets) stay zero — not normalized to uniform."""
                                    K = t.shape[-1]
                                    if K < E:
                                        pad = torch.zeros(t.shape[0], E - K, device=t.device, dtype=t.dtype)
                                        t = torch.cat([t, pad], dim=-1)
                                    elif K > E:
                                        t = t[:, :E]
                                    t = torch.nan_to_num(t, nan=0.0)  # 清除 NaN
                                    s = t.sum(dim=-1, keepdim=True)
                                    has_mass = (s > 1e-8)             # [B, 1] bool：有有效概率质量
                                    t = torch.where(has_mass, t / s.clamp(min=1e-8), torch.zeros_like(t))
                                    return t

                                def _kl_div(p, q):
                                    """KL(p || q) where p is teacher, q is student. Both [B, E].
                                    Zero rows in p contribute 0 (no teacher signal for that sample)."""
                                    log_p = torch.log(p + 1e-8)
                                    log_q = torch.log(q + 1e-8)
                                    return (p * (log_p - log_q)).sum(dim=-1)  # [B]

                                E = pred.shape[-1]
                                # guard: student probs may contain NaN at training start
                                pred_safe = torch.nan_to_num(pred, nan=0.0)
                                pred_safe = pred_safe.clamp(min=0.0)
                                pred_sum = pred_safe.sum(dim=-1, keepdim=True)
                                pred_safe = torch.where(
                                    pred_sum > 1e-8,
                                    pred_safe / pred_sum.clamp(min=1e-8),
                                    torch.ones_like(pred_safe) / E,
                                )

                                g_raw = _align_and_norm(sampled_global_assignment, E)  # [B, E]
                                l_raw = _align_and_norm(sampled_local_assignment, E)   # [B, E]

                                # visibility gate on global teacher (same logic as single-teacher path)
                                visibility_mask = self._extract_visibility_mask(policy, raw_states, E)
                                if visibility_mask is not None:
                                    vis_f = visibility_mask.to(pred_safe.dtype)
                                    visible_gate, valid_samples = self._compute_visible_mass_gate(g_raw, visibility_mask)
                                    valid_f = valid_samples.to(pred_safe.dtype)
                                    denom = valid_f.sum().clamp(min=1.0)

                                    # skip if no valid samples (no visible targets in entire batch)
                                    if valid_f.sum() < 1.0:
                                        distill_loss = torch.tensor(0.0, device=pred.device)
                                    else:
                                        # G_vis: global teacher masked to visible targets
                                        g_vis = g_raw * vis_f
                                        g_vis_sum = g_vis.sum(dim=-1, keepdim=True)
                                        g_vis = torch.where(g_vis_sum > 1e-8, g_vis / g_vis_sum.clamp(min=1e-8), torch.zeros_like(g_vis))
                                        g_vis = g_vis * valid_f.unsqueeze(-1)

                                        # L: local teacher (already visibility-aware by construction)
                                        l_vis = l_raw * valid_f.unsqueeze(-1)

                                        # consensus teacher C = normalize(eta * G_vis + (1-eta) * L)
                                            # eta 大，说明更相信 global teacher
                                            # eta 小，说明更相信 local teacher
                                        eta = self._distill_eta_consensus
                                        c_mix = eta * g_vis + (1.0 - eta) * l_vis
                                        c_sum = c_mix.sum(dim=-1, keepdim=True)
                                        c_vis = torch.where(c_sum > 1e-8, c_mix / c_sum.clamp(min=1e-8), torch.zeros_like(c_mix))
                                        c_vis = c_vis * valid_f.unsqueeze(-1)

                                        # student gated to visible
                                        pred_vis = pred_safe * vis_f
                                        pred_vis_sum = pred_vis.sum(dim=-1, keepdim=True)
                                        pred_vis = torch.where(
                                            pred_vis_sum > 1e-8,
                                            pred_vis / pred_vis_sum.clamp(min=1e-8),
                                            torch.ones_like(pred_vis) / E,
                                        )
                                        pred_vis = pred_vis * valid_f.unsqueeze(-1)

                                        alpha = self._distill_alpha_global
                                        lam_c = self._distill_lambda_consistency

                                        kl_g = _kl_div(g_vis, pred_vis)   # KL(G_vis || S)让学生像 global teacher
                                        kl_l = _kl_div(l_vis, pred_vis)   # KL(L || S)让学生像 local teacher
                                        kl_c = _kl_div(c_vis, pred_vis)   # KL(C || S)让学生像“这两个老师的折中意见”

                                        per_sample_dual = alpha * kl_g + (1.0 - alpha) * kl_l + lam_c * kl_c
                                        raw_kl_val = (per_sample_dual * valid_f).sum() / denom
                                        # visible_gate 只作监控，不乘进 loss
                                        weighted_kl_val = raw_kl_val
                                        batch_visible_gate = (visible_gate * valid_f).sum() / denom

                                        with torch.no_grad():
                                            m_tt = 0.5 * (g_vis + l_vis)
                                            m_tt_sum = m_tt.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                                            m_tt = m_tt / m_tt_sum * valid_f.unsqueeze(-1)
                                            # js_tt = 0.5 * (_kl_div(m_tt, g_vis) + _kl_div(m_tt, l_vis))
                                            js_tt = 0.5 * (_kl_div(g_vis, m_tt) + _kl_div(l_vis, m_tt))
                                            batch_js_teacher = (js_tt * valid_f).sum() / denom

                                            teacher_top1 = (g_vis * vis_f).argmax(dim=-1)
                                            student_top1 = (pred_vis * vis_f).argmax(dim=-1)
                                            top1_match = (teacher_top1 == student_top1).to(pred.dtype)
                                            batch_top1_agreement = (top1_match * valid_f).sum() / denom

                                        distill_loss = self._cached_distill_weight * weighted_kl_val
                                else:
                                    # fallback: no visibility mask
                                    eta = self._distill_eta_consensus
                                    c_mix = eta * g_raw + (1.0 - eta) * l_raw
                                    c_raw = c_mix / c_mix.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                                    alpha = self._distill_alpha_global
                                    lam_c = self._distill_lambda_consistency
                                    kl_g = _kl_div(g_raw, pred_safe)
                                    kl_l = _kl_div(l_raw, pred_safe)
                                    kl_c = _kl_div(c_raw, pred_safe)
                                    per_sample_dual = alpha * kl_g + (1.0 - alpha) * kl_l + lam_c * kl_c
                                    raw_kl_val = per_sample_dual.mean()
                                    weighted_kl_val = raw_kl_val
                                    batch_visible_gate = torch.tensor(1.0, device=pred.device)
                                    valid_samples = torch.ones(pred.shape[0], device=pred.device, dtype=torch.bool)
                                    with torch.no_grad():
                                        batch_js_teacher = torch.tensor(0.0, device=pred.device)
                                        batch_top1_agreement = (g_raw.argmax(-1) == pred_safe.argmax(-1)).float().mean()
                                    distill_loss = self._cached_distill_weight * weighted_kl_val

                            elif not self._dual_teacher_enable and sampled_optimal_assignment is not None:
                                # ---- single-teacher path (original logic preserved) ----
                                target = sampled_optimal_assignment
                                K_target, E = target.shape[-1], pred.shape[-1]
                                if K_target != E:
                                    if K_target < E:
                                        pad = torch.zeros(target.shape[0], E - K_target, device=target.device, dtype=target.dtype)
                                        target = torch.cat([target, pad], dim=-1)
                                    else:
                                        target = target[:, :E]
                                target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-8)

                                visibility_mask = self._extract_visibility_mask(policy, raw_states, target.shape[-1])
                                batch_js_teacher = torch.tensor(0.0)

                                if visibility_mask is not None:
                                    vis_f = visibility_mask.to(target.dtype)
                                    visible_gate, valid_samples = self._compute_visible_mass_gate(target, visibility_mask)
                                    valid_f = valid_samples.to(target.dtype)
                                    target_visible = target * vis_f
                                    tg_sum = target_visible.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                                    target_gated = target_visible / tg_sum * valid_f.unsqueeze(-1)
                                    pred_visible = pred * vis_f
                                    pg_sum = pred_visible.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                                    pred_gated = pred_visible / pg_sum * valid_f.unsqueeze(-1)
                                    log_pred_g = torch.log(pred_gated + 1e-8)
                                    log_target_g = torch.log(target_gated + 1e-8)
                                    per_sample_kl = (target_gated * (log_target_g - log_pred_g)).sum(dim=-1)
                                    denom = valid_f.sum().clamp(min=1.0)
                                    raw_kl_val = (per_sample_kl * valid_f).sum() / denom
                                    weighted_kl_val = raw_kl_val  # visible_gate 只作监控，不乘进 loss
                                    batch_visible_gate = (visible_gate * valid_f).sum() / denom
                                    with torch.no_grad():
                                        teacher_top1 = (target_visible * vis_f).argmax(dim=-1)
                                        student_top1 = (pred_visible * vis_f).argmax(dim=-1)
                                        top1_match = (teacher_top1 == student_top1).to(target.dtype)
                                        batch_top1_agreement = (top1_match * valid_f).sum() / denom
                                else:
                                    log_pred = torch.log(pred + 1e-8)
                                    log_target = torch.log(target + 1e-8)
                                    raw_kl_val = (target * (log_target - log_pred)).sum(dim=-1).mean()
                                    weighted_kl_val = raw_kl_val
                                    batch_visible_gate = torch.tensor(1.0, device=pred.device)
                                    valid_samples = torch.ones(pred.shape[0], device=pred.device, dtype=torch.bool)
                                    with torch.no_grad():
                                        batch_top1_agreement = (target.argmax(-1) == pred.argmax(-1)).float().mean()

                                distill_loss = self._cached_distill_weight * weighted_kl_val

                                # entropy accumulation for ITAD
                                if self._distill_anneal_type in ("itad", "itad_v2"):
                                    _cache = getattr(policy, '_assignment_probs_cache', None)
                                    if _cache is not None:
                                        with torch.no_grad():
                                            p = _cache.detach()
                                            if visibility_mask is not None:
                                                vis = visibility_mask.to(p.dtype)
                                                p = p * vis
                                                p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                                                k_vis = vis.sum(dim=-1).clamp(min=1.0)
                                                norm = torch.log(k_vis).clamp(min=1e-6)
                                                valid_f_e = valid_samples.to(p.dtype)
                                                e_sample = -(p * torch.log(p.clamp(min=1e-8))).sum(dim=-1) / norm
                                                e = (e_sample * valid_f_e).sum() / valid_f_e.sum().clamp(min=1.0)
                                            else:
                                                e = -(p * torch.log(p.clamp(min=1e-8))).sum(dim=-1).mean()
                                            _entropy_accum = e if _entropy_accum is None else (_entropy_accum + e)
                                            _entropy_accum_count += 1
                    # ---- end distillation loss ----
                # optimization step
                self.optimizers[agent_0].zero_grad()
                total_loss = policy_loss + entropy_loss + value_loss + assignment_reg_loss + distill_loss
                self.scaler.scale(total_loss).backward()

                if config.torch.is_distributed:
                    policy.reduce_parameters()
                    if policy is not value:
                        value.reduce_parameters()

                if self._grad_norm_clip[agent_0] > 0:
                    self.scaler.unscale_(self.optimizers[agent_0])
                    if policy is value:
                        nn.utils.clip_grad_norm_(policy.parameters(), self._grad_norm_clip[agent_0])
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(policy.parameters(), value.parameters()), self._grad_norm_clip[agent_0]
                        )

                self.scaler.step(self.optimizers[agent_0])
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale[agent_0]:
                    cumulative_entropy_loss += entropy_loss.item()
                if has_assignment_reg:
                    if isinstance(assignment_reg_loss, torch.Tensor):
                        cumulative_assignment_reg_loss += assignment_reg_loss.item()
                    else:
                        cumulative_assignment_reg_loss += float(assignment_reg_loss)
                if isinstance(distill_loss, torch.Tensor):
                    cumulative_distill_loss += distill_loss.item()
                    cumulative_raw_kl += raw_kl_val.item() if isinstance(raw_kl_val, torch.Tensor) else float(raw_kl_val)
                    if batch_visible_gate is not None:
                        cumulative_visible_gate += batch_visible_gate.item() if isinstance(batch_visible_gate, torch.Tensor) else float(batch_visible_gate)
                    if isinstance(batch_top1_agreement, torch.Tensor):
                        cumulative_top1_agreement += batch_top1_agreement.item()
                    else:
                        cumulative_top1_agreement += float(batch_top1_agreement)
                    if self._dual_teacher_enable and 'batch_js_teacher' in locals():
                        if isinstance(batch_js_teacher, torch.Tensor):
                            cumulative_js_teacher += batch_js_teacher.item()
                        else:
                            cumulative_js_teacher += float(batch_js_teacher)
                    distill_count += 1

            # update learning rate
            if self._learning_rate_scheduler[agent_0]:
                if isinstance(self.schedulers[agent_0], KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.schedulers[agent_0].step(kl.item())
                    cumulative_kl_divergence += kl.item()
                else:
                    self.schedulers[agent_0].step()

        # record data
        self.track_data(
            f"Loss / Policy loss",
            cumulative_policy_loss / (self._learning_epochs[agent_0] * self._mini_batches[agent_0]),
        )
        self.track_data(
            f"Loss / Value loss",
            cumulative_value_loss / (self._learning_epochs[agent_0] * self._mini_batches[agent_0]),
        )
        if self._entropy_loss_scale[agent_0]:
            self.track_data(
                f"Loss / Entropy loss",
                cumulative_entropy_loss / (self._learning_epochs[agent_0] * self._mini_batches[agent_0]),
            )
        if has_assignment_reg:
            self.track_data(
                f"Distillation / Assignment reg loss",
                cumulative_assignment_reg_loss / (self._learning_epochs[agent_0] * self._mini_batches[agent_0]),
            ) # 越小说明top1和top2的分配概率差距越大（更"果断"），但过大会导致过拟合 teacher 的分配（不够灵活）。这个正则项的引入是为了让学生在学习 teacher 的分配时更果断一些，避免过于平均地分配注意力，从而更好地模仿 teacher 的行为。
        if self._distill_weight_init > 0:
            avg_raw_kl = cumulative_raw_kl / max(distill_count, 1)
            avg_weighted_loss = cumulative_distill_loss / max(distill_count, 1)

            avg_entropy = None
            if self._distill_anneal_type in ("itad", "itad_v2") and _entropy_accum is not None and _entropy_accum_count > 0:
                avg_entropy = (_entropy_accum / _entropy_accum_count).item()

            avg_visible_gate = cumulative_visible_gate / max(distill_count, 1)
            avg_top1_agreement = cumulative_top1_agreement / max(distill_count, 1)
            avg_js_teacher = cumulative_js_teacher / max(distill_count, 1)

            if distill_count > 0:
                self._update_distill_ema(avg_raw_kl, avg_entropy,
                                         avg_visible_gate=avg_visible_gate,
                                         avg_top1_agreement=avg_top1_agreement)

            delta = timestep - self._distill_last_trainer_timestep
            self._distill_total_timestep += max(delta, 0)
            self._distill_last_trainer_timestep = timestep

            # dual-teacher: use release schedule; single-teacher: use existing _compute_distill_weight
            if self._dual_teacher_enable:
                self._cached_distill_weight = self._update_release_schedule()
            else:
                self._cached_distill_weight = self._compute_distill_weight()

            self.track_data(f"Distillation / Distillation loss (raw KL)", avg_raw_kl)
            self.track_data(f"Distillation / Distillation loss (weighted)", avg_weighted_loss)
            self.track_data(f"Distillation / Visible mass gate", avg_visible_gate)
            self.track_data(f"Distillation / Top1 agreement", avg_top1_agreement)
            self.track_data(f"Distillation / Distillation weight", self._cached_distill_weight)

            if self._dual_teacher_enable:
                # teacher-teacher JS divergence: monitoring metric only, no gradient
                    # 值小：两个 teacher 给的分配很接近
                    # 值大：两个 teacher 分歧比较大
                self.track_data(f"Distillation / Teacher-Teacher JS", avg_js_teacher)

            if self._distill_kl_ema is not None:
                self.track_data(f"Distillation / KL EMA", self._distill_kl_ema)             # KL EMA大说明学生和老师的差距大

            # ITAD 额外指标
            # Distillation / Info Gain EMA
                # 大且为正：teacher 还在持续带来有效指导（学生在"变像 + 变确定"）
                # 接近 0：teacher 边际贡献变小（学生基本收敛到 teacher 或不再从它学到新东西）
                # 为负：最近在"变不像/更犹豫"（但注意：负不必然是坏探索，只是"对齐 teacher 的进展"在倒退)
            # Distillation / Entropy EMA
                # 高：分配更均匀、更犹豫（不确定选谁）
                # 低：分配更尖锐、更果断（基本锁定一个目标）
                # 长期下降：学生在变得更"确定"
            # Distillation / Info Gain Baseline
                # baseline 越大：说明早期/当前 info_gain 的波动幅度大（teacher 边际贡献曾经/仍然明显）
                # baseline 越小：说明 info_gain 的幅度整体变小（训练更接近收敛）
            # Distillation / Perf Fast EMA, Perf Slow EMA
                # fast > slow：短期性能高于长期趋势（正在上升或稳定）
                # fast < slow：短期性能低于长期趋势（正在下降）
            # Distillation / Perf Slow Drop
                # 0.00：慢EMA没有退步（长期趋势处于历史最好附近）
                # 0.05：慢EMA相对最佳水平退步了 5%
            # Distillation / Intercept Rate EMA
                # 越大表示“最近一段时间平均拦截效果越好”
            # Distillation / Success Rate EMA
                # 越大表示“最近一段时间平均任务成功效果越好”
            if self._distill_anneal_type in ("itad", "itad_v2"):
                if self._distill_info_gain_ema is not None:
                    self.track_data(f"Distillation / Info Gain EMA", self._distill_info_gain_ema)
                if self._distill_entropy_ema is not None:
                    self.track_data(f"Distillation / Entropy EMA", self._distill_entropy_ema)
                if self._distill_info_gain_baseline is not None:
                    self.track_data(f"Distillation / Info Gain Baseline", self._distill_info_gain_baseline)
                if self._distill_perf_fast is not None:
                    self.track_data(f"Distillation / Perf Fast EMA", self._distill_perf_fast)
                if self._distill_perf_slow is not None:
                    self.track_data(f"Distillation / Perf Slow EMA", self._distill_perf_slow)
                if self._distill_perf_best is not None:
                    self.track_data(f"Distillation / Perf Best", self._distill_perf_best)
                    if abs(self._distill_perf_best) > 1e-6 and self._distill_perf_slow is not None:
                        slow_drop = max(0.0, (self._distill_perf_best - self._distill_perf_slow) / abs(self._distill_perf_best))
                        self.track_data(f"Distillation / Perf Slow Drop", slow_drop)
                if self._distill_intercept_rate_ema is not None:
                    self.track_data(f"Distillation / Intercept Rate EMA", self._distill_intercept_rate_ema)
                if self._distill_success_rate_ema is not None:
                    self.track_data(f"Distillation / Success Rate EMA", self._distill_success_rate_ema)

            # dual-teacher: also log success_rate_ema for release trigger monitoring
            if self._dual_teacher_enable and self._distill_success_rate_ema is not None:
                self.track_data(f"Distillation / Success Rate EMA", self._distill_success_rate_ema)
            if self._dual_teacher_enable and self._distill_intercept_rate_ema is not None:
                self.track_data(f"Distillation / Intercept Rate EMA", self._distill_intercept_rate_ema)

        self.track_data(f"Learning / Standard deviation", policy.distribution(role="policy").stddev.mean().item())
        self.track_data(f"Learning / Explained variance", explained_var)
        self.track_data(f"Learning / Clip fraction", torch.stack(clip_fractions).mean().item() if clip_fractions else 0.0)
        if self._learning_rate_scheduler[agent_0]:
            self.track_data(f"Learning / Learning rate", self.schedulers[agent_0].get_last_lr()[0])
            if isinstance(self.schedulers[agent_0], KLAdaptiveLR):
                self.track_data(f"Learning / KL divergence", cumulative_kl_divergence / self._learning_epochs[agent_0])

        if hasattr(policy, 'get_assignment_entropy'):
            try:
                assignment_entropy = policy.get_assignment_entropy()
                # 太高：分配犹豫，可能 entropy_reg_weight 太小/冲突惩罚太大/输入信息不足。太低：过早"硬分配"，可能 entropy_reg_weight 太大导致过拟合到单目标（或抢目标）
                self.track_data(f"Assignment / Entropy", assignment_entropy.item())

                # 记录top-1分配的平均概率（越高说明越确定）
                if hasattr(policy, '_assignment_probs_cache') and policy._assignment_probs_cache is not None:
                    probs = policy._assignment_probs_cache
                    top1_prob = probs.max(dim=-1)[0].mean()
                    self.track_data(f"Assignment / Top1 Probability", top1_prob.item())
            except Exception as e:
                pass

        # 每次 update 结束后递增 margin reg update 计数器
        self._margin_reg_update_count += 1