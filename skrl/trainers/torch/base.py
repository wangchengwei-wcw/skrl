from typing import List, Optional, Union

import atexit
import sys
import time
import tqdm
from loguru import logger as logger_

import torch

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper


def generate_equally_spaced_scopes(num_envs: int, num_simultaneous_agents: int) -> List[int]:
    """Generate a list of equally spaced scopes for the agents

    :param num_envs: Number of environments
    :type num_envs: int
    :param num_simultaneous_agents: Number of simultaneous agents
    :type num_simultaneous_agents: int

    :raises ValueError: If the number of simultaneous agents is greater than the number of environments

    :return: List of equally spaced scopes
    :rtype: List[int]
    """
    scopes = [int(num_envs / num_simultaneous_agents)] * num_simultaneous_agents
    if sum(scopes):
        scopes[-1] += num_envs - sum(scopes)
    else:
        raise ValueError(
            f"The number of simultaneous agents ({num_simultaneous_agents}) is greater than the number of environments ({num_envs})"
        )
    return scopes


class Trainer:
    def __init__(
        self,
        env: Wrapper,
        agents: Union[Agent, List[Agent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Base class for trainers

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``)
        :type cfg: dict, optional
        """
        self.cfg = cfg if cfg is not None else {}
        self.env = env
        self.agents = agents
        self.agents_scope = agents_scope if agents_scope is not None else []

        # get configuration
        self.timesteps = self.cfg.get("timesteps", 0)
        self.headless = self.cfg.get("headless", False)
        self.disable_progressbar = self.cfg.get("disable_progressbar", False)
        self.close_environment_at_exit = self.cfg.get("close_environment_at_exit", True)
        self.environment_info = self.cfg.get("environment_info", "episode")
        self.stochastic_evaluation = self.cfg.get("stochastic_evaluation", False)

        self.initial_timestep = 0

        # setup agents
        self.num_simultaneous_agents = 0
        self._setup_agents()

        # register environment closing if configured
        if self.close_environment_at_exit:

            @atexit.register
            def close_env():
                logger.info("Closing environment")
                self.env.close()
                logger.info("Environment closed")

        # update trainer configuration to avoid duplicated info/data in distributed runs
        if config.torch.is_distributed:
            if config.torch.rank:
                self.disable_progressbar = True

    def __str__(self) -> str:
        """Generate a string representation of the trainer

        :return: Representation of the trainer as string
        :rtype: str
        """
        string = f"Trainer: {self}"
        string += f"\n  |-- Number of parallelizable environments: {self.env.num_envs}"
        string += f"\n  |-- Number of simultaneous agents: {self.num_simultaneous_agents}"
        string += "\n  |-- Agents and scopes:"
        if self.num_simultaneous_agents > 1:
            for agent, scope in zip(self.agents, self.agents_scope):
                string += f"\n  |     |-- agent: {type(agent)}"
                string += f"\n  |     |     |-- scope: {scope[1] - scope[0]} environments ({scope[0]}:{scope[1]})"
        else:
            string += f"\n  |     |-- agent: {type(self.agents)}"
            string += f"\n  |     |     |-- scope: {self.env.num_envs} environment(s)"
        return string

    def _setup_agents(self) -> None:
        """Setup agents for training

        :raises ValueError: Invalid setup
        """
        # validate agents and their scopes
        if type(self.agents) in [tuple, list]:
            # single agent
            if len(self.agents) == 1:
                self.num_simultaneous_agents = 1
                self.agents = self.agents[0]
                self.agents_scope = [1]
            # parallel agents
            elif len(self.agents) > 1:
                self.num_simultaneous_agents = len(self.agents)
                # check scopes
                if not len(self.agents_scope):
                    logger.warning("The agents' scopes are empty, they will be generated as equal as possible")
                    self.agents_scope = [int(self.env.num_envs / len(self.agents))] * len(self.agents)
                    if sum(self.agents_scope):
                        self.agents_scope[-1] += self.env.num_envs - sum(self.agents_scope)
                    else:
                        raise ValueError(
                            f"The number of agents ({len(self.agents)}) is greater than the number of parallelizable environments ({self.env.num_envs})"
                        )
                elif len(self.agents_scope) != len(self.agents):
                    raise ValueError(
                        f"The number of agents ({len(self.agents)}) doesn't match the number of scopes ({len(self.agents_scope)})"
                    )
                elif sum(self.agents_scope) != self.env.num_envs:
                    raise ValueError(
                        f"The scopes ({sum(self.agents_scope)}) don't cover the number of parallelizable environments ({self.env.num_envs})"
                    )
                # generate agents' scopes
                index = 0
                for i in range(len(self.agents_scope)):
                    index += self.agents_scope[i]
                    self.agents_scope[i] = (index - self.agents_scope[i], index)
            else:
                raise ValueError("A list of agents is expected")
        else:
            self.num_simultaneous_agents = 1

    def train(self) -> None:
        """Train the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def eval(self) -> None:
        """Evaluate the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def single_agent_train(self) -> None:
        """Train agent

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # reset env
        states, infos = self.env.reset()

        sim_real_time_ratio_avg = None
        t_compute_actions, t_step_env, t_render, t_record_transition, t_iters_per_update = 0.0, 0.0, 0.0, 0.0, 0.0
        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):
            start_iteration = time.perf_counter()

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                start_compute_actions = time.perf_counter()
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]
                end_compute_actions = time.perf_counter()
                t_compute_actions += end_compute_actions - start_compute_actions

                # step the environments
                start_step_env = time.perf_counter()
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                end_step_env = time.perf_counter()
                t_step_env += end_step_env - start_step_env

                # render scene
                if not self.headless:
                    start_render = time.perf_counter()
                    self.env.render()
                    end_render = time.perf_counter()
                    t_render += end_render - start_render

                # record the environments' transitions
                start_record_transition = time.perf_counter()
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )
                end_record_transition = time.perf_counter()
                t_record_transition += end_record_transition - start_record_transition

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            start_update = time.perf_counter()
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)
            end_update = time.perf_counter()
            t_update = end_update - start_update

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

            # runtime report
            end_iteration = time.perf_counter()
            t_iteration = end_iteration - start_iteration
            t_iters_per_update += t_iteration
            t_iteration_sim = self.env.step_dt
            if sim_real_time_ratio_avg is None:
                sim_real_time_ratio_avg = t_iteration_sim / t_iteration
            else:
                sim_real_time_ratio = t_iteration_sim / t_iteration
                count = timestep - self.initial_timestep + 1
                sim_real_time_ratio_avg = (sim_real_time_ratio_avg * (count - 1) + sim_real_time_ratio) / count

            if not self.agents._rollout % self.agents._rollouts and (timestep - 1) >= self.agents._learning_starts:
                logger_.info("")
                logger_.info(f"Trainer iteration takes {t_iters_per_update:.5f}s")
                logger_.info(f">>> Compute actions takes up {t_compute_actions / t_iters_per_update * 100:.2f}%")
                logger_.info(f">>> Step envs takes up {t_step_env / t_iters_per_update * 100:.2f}%, {t_step_env / self.agents._rollouts:.5f}s / step")
                logger_.info(f">>> Render takes up {t_render / t_iters_per_update * 100:.2f}%")
                logger_.info(f">>> Record transitions takes up {t_record_transition / t_iters_per_update * 100:.2f}%")
                logger_.info(f">>> Update takes up {t_update / t_iters_per_update * 100:.2f}%, {t_update:.5f}s")
                logger_.info(f"Sim time passes {sim_real_time_ratio_avg:.2f} times faster than real time")
                logger_.info("\n\n\n\n=========================================================================================================================================\n===================================================== Training Iteration Split Line =====================================================\n=========================================================================================================================================")
                t_compute_actions = t_step_env = t_render = t_record_transition = t_iters_per_update = 0.0

    def single_agent_eval(self) -> None:
        """Evaluate agent

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                outputs = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)
                actions = outputs[0] if self.stochastic_evaluation else outputs[-1].get("mean_actions", outputs[0])

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

    def multi_agent_train(self) -> None:
        """Train multi-agents

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"

        # reset env
        states, infos = self.env.reset()
        shared_states = self.env.state()

        sim_real_time_ratio_avg = None
        t_compute_actions, t_step_env, t_render, t_record_transition, t_iters_per_update = 0.0, 0.0, 0.0, 0.0, 0.0
        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):
            start_iteration = time.perf_counter()

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                start_compute_actions = time.perf_counter()
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]
                end_compute_actions = time.perf_counter()
                t_compute_actions += end_compute_actions - start_compute_actions

                # Inject assignment_probs into environment before step
                if hasattr(self.agents, '_stacked_assignment_probs') and self.agents._stacked_assignment_probs is not None:
                    # Access unwrapped environment to set attribute (wrapper __setattr__ doesn't forward)
                    target_env = self.env._unwrapped if hasattr(self.env, '_unwrapped') else self.env
                    if hasattr(target_env, '_assignment_probs'):
                        # Validate assignment_probs before passing to environment
                        assignment_probs = self.agents._stacked_assignment_probs

                        # Check for NaN/Inf
                        if not torch.isfinite(assignment_probs).all():
                            # If NaN/Inf detected, replace with uniform distribution
                            N, M, E = assignment_probs.shape
                            assignment_probs = torch.ones_like(assignment_probs) / E
                            logger.warning("Assignment probs contain NaN/Inf, replaced with uniform distribution")

                        # Check dimension match
                        expected_shape = (target_env.num_envs, target_env.M, target_env.E)
                        if assignment_probs.shape != expected_shape:
                            logger.error(
                                f"Assignment probs shape mismatch: got {assignment_probs.shape}, "
                                f"expected {expected_shape}. Skipping injection."
                            )
                            target_env._assignment_probs = None
                        else:
                            target_env._assignment_probs = assignment_probs

                # Sync distill_enabled flag to env (skip Sinkhorn when weight has annealed to 0)
                if hasattr(self.agents, '_cached_distill_weight') and hasattr(self.agents, '_distill_weight_init'):
                    _target_env = self.env._unwrapped if hasattr(self.env, '_unwrapped') else self.env
                    if hasattr(_target_env, '_distill_enabled'):
                        _target_env._distill_enabled = self.agents._cached_distill_weight > 0

                # Sync hard assignment reward flag to env
                if hasattr(self.agents, '_margin_reg_update_count'):
                    _target_env = self.env._unwrapped if hasattr(self.env, '_unwrapped') else self.env
                    if hasattr(_target_env, '_use_hard_assignment_reward'):
                        reg_warmup = self.agents.cfg.get("assignment_reg_warmup_updates", 0)
                        _target_env._use_hard_assignment_reward = (
                            reg_warmup > 0 and self.agents._margin_reg_update_count >= reg_warmup
                        )

                # step the environments
                start_step_env = time.perf_counter()
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states
                end_step_env = time.perf_counter()
                t_step_env += end_step_env - start_step_env

                # Read optimal assignment from environment for distillation
                target_env = self.env._unwrapped if hasattr(self.env, '_unwrapped') else self.env
                if hasattr(target_env, '_optimal_assignment_sorted') and hasattr(self.agents, '_optimal_assignment_sorted'):
                    self.agents._optimal_assignment_sorted = target_env._optimal_assignment_sorted
                # Sync dual teacher tensors (global + local) for dual-teacher distillation
                if hasattr(target_env, '_global_assignment_sorted') and hasattr(self.agents, '_global_assignment_sorted'):
                    self.agents._global_assignment_sorted = target_env._global_assignment_sorted
                if hasattr(target_env, '_local_assignment_sorted') and hasattr(self.agents, '_local_assignment_sorted'):
                    self.agents._local_assignment_sorted = target_env._local_assignment_sorted

                # render scene
                if not self.headless:
                    start_render = time.perf_counter()
                    self.env.render()
                    end_render = time.perf_counter()
                    t_render += end_render - start_render

                # record the environments' transitions
                start_record_transition = time.perf_counter()
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )
                end_record_transition = time.perf_counter()
                t_record_transition += end_record_transition - start_record_transition

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            start_update = time.perf_counter()
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)
            end_update = time.perf_counter()
            t_update = end_update - start_update

            # reset environments
            if not self.env.agents:
                with torch.no_grad():
                    states, infos = self.env.reset()
                    shared_states = self.env.state()
            else:
                states = next_states
                shared_states = shared_next_states

            # runtime report
            end_iteration = time.perf_counter()
            t_iteration = end_iteration - start_iteration
            t_iters_per_update += t_iteration
            t_iteration_sim = self.env.step_dt
            if sim_real_time_ratio_avg is None:
                sim_real_time_ratio_avg = t_iteration_sim / t_iteration
            else:
                sim_real_time_ratio = t_iteration_sim / t_iteration
                count = timestep - self.initial_timestep + 1
                sim_real_time_ratio_avg = (sim_real_time_ratio_avg * (count - 1) + sim_real_time_ratio) / count

            if not self.agents._rollout % self.agents._rollouts and (timestep - 1) >= self.agents._learning_starts:
                logger_.info("")
                logger_.info(f"Trainer iteration takes {t_iters_per_update:.5f}s")
                logger_.info(f">>> Compute actions takes up {t_compute_actions / t_iters_per_update * 100:.2f}%")
                logger_.info(f">>> Step envs takes up {t_step_env / t_iters_per_update * 100:.2f}%, {t_step_env / self.agents._rollouts:.5f}s / step")
                logger_.info(f">>> Render takes up {t_render / t_iters_per_update * 100:.2f}%")
                logger_.info(f">>> Record transitions takes up {t_record_transition / t_iters_per_update * 100:.2f}%")
                logger_.info(f">>> Update takes up {t_update / t_iters_per_update * 100:.2f}%, {t_update:.5f}s")
                logger_.info(f"Sim time passes {sim_real_time_ratio_avg:.2f} times faster than real time")
                logger_.info("\n\n\n\n=========================================================================================================================================\n===================================================== Training Iteration Split Line =====================================================\n=========================================================================================================================================")
                t_compute_actions = t_step_env = t_render = t_record_transition = t_iters_per_update = 0.0

    def multi_agent_eval(self) -> None:
        """Evaluate multi-agents

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"

        # reset env
        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                outputs = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)
                actions = (
                    outputs[0]
                    if self.stochastic_evaluation
                    else {k: outputs[-1][k].get("mean_actions", outputs[0][k]) for k in outputs[-1]}
                )

                # Inject assignment_probs into environment before step
                if hasattr(self.agents, '_stacked_assignment_probs') and self.agents._stacked_assignment_probs is not None:
                    # Access unwrapped environment to set attribute (wrapper __setattr__ doesn't forward)
                    target_env = self.env._unwrapped if hasattr(self.env, '_unwrapped') else self.env
                    if hasattr(target_env, '_assignment_probs'):
                        # Validate assignment_probs before passing to environment
                        assignment_probs = self.agents._stacked_assignment_probs

                        # Check for NaN/Inf
                        if not torch.isfinite(assignment_probs).all():
                            # If NaN/Inf detected, replace with uniform distribution
                            N, M, E = assignment_probs.shape
                            assignment_probs = torch.ones_like(assignment_probs) / E
                            logger.warning("Assignment probs contain NaN/Inf, replaced with uniform distribution")

                        # Check dimension match
                        expected_shape = (target_env.num_envs, target_env.M, target_env.E)
                        if assignment_probs.shape != expected_shape:
                            logger.error(
                                f"Assignment probs shape mismatch: got {assignment_probs.shape}, "
                                f"expected {expected_shape}. Skipping injection."
                            )
                            target_env._assignment_probs = None
                        else:
                            target_env._assignment_probs = assignment_probs

                # Sync hard assignment reward flag to env
                if hasattr(self.agents, '_margin_reg_update_count'):
                    _target_env = self.env._unwrapped if hasattr(self.env, '_unwrapped') else self.env
                    if hasattr(_target_env, '_use_hard_assignment_reward'):
                        reg_warmup = self.agents.cfg.get("assignment_reg_warmup_updates", 0)
                        _target_env._use_hard_assignment_reward = (
                            reg_warmup > 0 and self.agents._margin_reg_update_count >= reg_warmup
                        )

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if not self.env.agents:
                with torch.no_grad():
                    states, infos = self.env.reset()
                    shared_states = self.env.state()
            else:
                states = next_states
                shared_states = shared_next_states
