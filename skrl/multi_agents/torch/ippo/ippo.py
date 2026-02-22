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
        self._distill_weight_init = self.cfg.get("distill_weight_init", 0.0)
        self._distill_anneal_type = self.cfg.get("distill_anneal_type", "linear")

        # Time-based annealing parameters
        self._distill_anneal_steps = self.cfg.get("distill_anneal_steps", 200000)
        self._distill_weight_min = self.cfg.get("distill_weight_min", 0.0)

        # KL-adaptive annealing parameters
        self._distill_target_kl = self.cfg.get("distill_target_kl", 0.1)
        self._distill_kl_ema_momentum = self.cfg.get("distill_kl_ema_momentum", 0.99)
        self._distill_kl_ema = None  # 指数移动平均的KL（每个update只更新一次）
        self._distill_initial_kl = None  # 初始KL（用于kl_ratio）
        self._cached_distill_weight = self._distill_weight_init  # 缓存的权重，mini-batch中使用

        # routing regularization configuration
        self._routing_reg_weight = self.cfg.get("routing_reg_weight", 0.01)

        self._optimal_assignment_sorted = None  # [N, M, K_target] from trainer

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

    def _update_distill_ema(self, avg_raw_kl: float):
        """每个update结束时调用一次,用本次update的平均raw KL来更新KL的EMA(指数滑动平均)。

        Args:
            avg_raw_kl: 本次update中所有mini-batch的原始KL平均值
        """
        if self._distill_weight_init <= 0:
            return

        # 记录初始KL（用于kl_ratio）
        if self._distill_initial_kl is None:
            self._distill_initial_kl = avg_raw_kl

        # 更新KL的指数移动平均（每个update只更新一次）
        # ema = momentum * ema + (1 - momentum) * KL
        if self._distill_kl_ema is None:
            self._distill_kl_ema = avg_raw_kl
        else:
            self._distill_kl_ema = (
                self._distill_kl_ema_momentum * self._distill_kl_ema +
                (1 - self._distill_kl_ema_momentum) * avg_raw_kl
            )

    def _compute_distill_weight(self, timestep: int) -> float:
        """纯函数:根据当前EMA状态计算蒸馏权重,不修改任何状态。

        Args:
            timestep: 当前训练步数

        Returns:
            蒸馏权重
        """
        # TODO：这里的自适应退火是否有更加合理的设计？或者权重降低更加合理？
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
        elif self._distill_anneal_type == "hybrid":
            time_factor = max(0.0, 1.0 - timestep / self._distill_anneal_steps)
            if self._distill_kl_ema is not None and self._distill_initial_kl is not None:
                kl_factor = min(1.0, self._distill_kl_ema / max(self._distill_initial_kl, 1e-8))
                kl_factor = max(self._distill_weight_min, kl_factor)
            else:
                kl_factor = 1.0
            return self._distill_weight_init * min(time_factor, kl_factor)
        elif self._distill_anneal_type == "linear":
            progress = min(timestep / self._distill_anneal_steps, 1.0)
            return self._distill_weight_init * max(0.0, 1.0 - progress)
        else:
            raise ValueError(f"Unknown distill_anneal_type: {self._distill_anneal_type}")

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
                if self._distill_weight_init > 0 and self._optimal_assignment_sorted is not None:
                    agent_idx = self.possible_agents.index(uid)
                    # _optimal_assignment_sorted: [N, M, K_target] → take agent_idx slice → [N, K_target]
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
        v_pred = torch.cat(value_list).reshape(-1)
        v_true = torch.cat(return_list).reshape(-1)
        var_v = torch.var(v_true, unbiased=False)
        explained_var = float("nan") if var_v == 0 else (1 - torch.var(v_true - v_pred, unbiased=False) / var_v).item()

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
        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_kl_divergence = 0
        cumulative_assignment_reg_loss = 0
        cumulative_distill_loss = 0
        cumulative_raw_kl = 0               # 原始KL散度（未加权）
        cumulative_routing_loss = 0         # 路由正则化损失
        distill_count = 0                   # 实际计算了KL的batch数
        clip_fractions = []
        has_assignment_reg = hasattr(policy, "get_entropy_regularization")

        # learning epochs
        for epoch in range(self._learning_epochs[agent_0]):
            kl_divergences = []

            # mini-batches loop
            for batch in sampled_batches_all:

                sampled_optimal_assignment = None
                if self._distill_weight_init > 0 and len(batch) == 7:
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

                    raw_states = sampled_states
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

                    # compute assignment entropy regularization
                    # 让 assignment_probs 更“硬”、更果断地选某个目标
                    assignment_reg_loss = 0
                    if has_assignment_reg:
                        try:
                            assignment_reg_loss = policy.get_entropy_regularization()
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

                    # compute distillation loss (Sinkhorn optimal assignment → attention assignment)
                    # 用 teacher（Sinkhorn 算出来的"最优目标分配分布"）去监督 student（你注意力模块输出的 assignment_probs），让 student 在训练早期更快学会"怎么分配目标"。
                    distill_loss = 0
                    if self._distill_weight_init > 0 and sampled_optimal_assignment is not None:
                        # get current assignment_probs from cached policy output
                        assignment_probs = getattr(policy, '_assignment_probs_cache', None)
                        if assignment_probs is not None:
                            # KL divergence: target distribution (Sinkhorn) vs learned distribution
                            target = sampled_optimal_assignment  # [B, K_target]
                            pred = assignment_probs              # [B, E]

                            # Handle dimension mismatch (K_target != E)
                            K_target = target.shape[-1]
                            E = pred.shape[-1]

                            if K_target != E:
                                logger.warning(
                                    f"Dimension mismatch in distillation: K_target={K_target}, E={E}. "
                                    f"This should not happen if config is correct (num_targets should equal obs_k_target). "
                                    f"Applying fallback padding/truncation, but this may cause information loss."
                                )
                                if K_target < E:
                                    # Pad target with zeros
                                    pad_size = E - K_target
                                    pad = torch.zeros(target.shape[0], pad_size, device=target.device, dtype=target.dtype)
                                    target = torch.cat([target, pad], dim=-1)
                                else:
                                    # Truncate target to match E
                                    target = target[:, :E]

                            # normalize target to valid distribution (may not sum to 1 due to masking)
                            target_sum = target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                            target = target / target_sum

                            # KL(target || learned) = Σ target * log(target / learned)
                            log_pred = torch.log(pred + 1e-8)
                            log_target = torch.log(target + 1e-8)
                            raw_kl_val = (target * (log_target - log_pred)).sum(dim=-1).mean()

                            # 使用缓存的权重（上一次update结束时计算的，不更新EMA）
                            distill_loss = self._cached_distill_weight * raw_kl_val

                    # compute routing specialization loss (encourage peaked routing weights)
                    routing_loss = 0
                    routing_weights = getattr(policy, '_routing_weights_cache', None)
                    if routing_weights is not None and hasattr(self, '_routing_reg_weight'):
                        # Encourage routing weights to be peaked (not uniform)
                        # Use negative entropy: lower entropy = more peaked distribution
                        routing_entropy = -(routing_weights * torch.log(routing_weights + 1e-8)).sum(dim=-1).mean()
                        routing_loss = self._routing_reg_weight * routing_entropy

                # optimization step
                # NOTE：这新加的两个loss项是否会起到共同推进的作用，还是相互干扰的作用？
                self.optimizers[agent_0].zero_grad()
                total_loss = policy_loss + entropy_loss + value_loss + assignment_reg_loss + distill_loss + routing_loss
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
                    cumulative_raw_kl += raw_kl_val.item()
                    distill_count += 1
                if isinstance(routing_loss, torch.Tensor):
                    cumulative_routing_loss += routing_loss.item()

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
                f"Distillation / Loss_Assignment reg loss",
                cumulative_assignment_reg_loss / (self._learning_epochs[agent_0] * self._mini_batches[agent_0]),
            )
        if self._distill_weight_init > 0:
            avg_raw_kl = cumulative_raw_kl / max(distill_count, 1)
            avg_weighted_loss = cumulative_distill_loss / max(distill_count, 1)

            if distill_count > 0:
                self._update_distill_ema(avg_raw_kl)

            self._cached_distill_weight = self._compute_distill_weight(timestep)

            self.track_data(f"Distillation / Loss_Distillation loss (raw KL)", avg_raw_kl) # teacher 和 student 分配分布的 真实差距（纯 KL），不乘权重。student 当前跟 teacher 像不像（越小越像）。
            self.track_data(f"Distillation / Loss_Distillation loss (weighted)", avg_weighted_loss) # 真正加进你 total_loss 里参与反向传播的蒸馏损失项。蒸馏这一项在总 loss 里到底贡献了多大（梯度强不强）
            self.track_data(f"Distillation / Loss_Distillation weight", self._cached_distill_weight)

            if self._distill_kl_ema is not None:
                self.track_data(f"Distillation / KL EMA", self._distill_kl_ema)
            if self._distill_target_kl > 0:
                self.track_data(f"Distillation / Target KL", self._distill_target_kl)

        # Track routing loss
        if cumulative_routing_loss > 0:
            avg_routing_loss = cumulative_routing_loss / (self._learning_epochs[agent_0] * self._mini_batches[agent_0])
            self.track_data(f"Routing / Loss_Routing specialization loss", avg_routing_loss)

        # Track routing weights (if available)
        if hasattr(policy, '_routing_weights_cache') and policy._routing_weights_cache is not None:
            routing_weights = policy._routing_weights_cache
            self.track_data(f"Routing / Team Assignment Weight", routing_weights[:, 0].mean().item())
            self.track_data(f"Routing / Collision Avoidance Weight", routing_weights[:, 1].mean().item())
            self.track_data(f"Routing / Uncertainty Weight", routing_weights[:, 2].mean().item())

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
                # 太高：分配犹豫，可能 entropy_reg_weight 太小/冲突惩罚太大/输入信息不足。太低：过早“硬分配”，可能 entropy_reg_weight 太大导致过拟合到单目标（或抢目标）
                self.track_data(f"Assignment / Entropy", assignment_entropy.item())

                # 记录top-1分配的平均概率（越高说明越确定）
                if hasattr(policy, '_assignment_probs_cache') and policy._assignment_probs_cache is not None:
                    probs = policy._assignment_probs_cache
                    top1_prob = probs.max(dim=-1)[0].mean()
                    self.track_data(f"Assignment / Top1 Probability", top1_prob.item())
            except Exception as e:
                pass
