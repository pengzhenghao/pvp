import copy
import io
import logging
import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Union, Optional

import numpy as np
import torch as th
import torch
from torch.nn import functional as F
from pvp.sb3.common.noise import ActionNoise, VectorizedActionNoise
from pvp.sb3.common.callbacks import BaseCallback
from pvp.sb3.common.buffers import ReplayBuffer
from pvp.sb3.common.save_util import load_from_pkl, save_to_pkl
from pvp.sb3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, \
    TrainFrequencyUnit
from pvp.sb3.common.utils import polyak_update
from pvp.sb3.haco.haco_buffer import HACOReplayBuffer, concat_samples
from pvp.sb3.td3.td3 import TD3
from pvp.sb3.td3.policies import TD3Policy
from pvp.sb3.common.utils import safe_mean, should_collect_more_steps
from pvp.sb3.common.vec_env import VecEnv
logger = logging.getLogger(__name__)


class PVPTD3(TD3):
    classifier: TD3Policy
    def __init__(self, use_balance_sample=True, q_value_bound=1., *args, **kwargs):
        """Please find the hyperparameters from original TD3"""
        if "cql_coefficient" in kwargs:
            self.cql_coefficient = kwargs["cql_coefficient"]
            kwargs.pop("cql_coefficient")
        else:
            self.cql_coefficient = 1
        if "replay_buffer_class" not in kwargs:
            kwargs["replay_buffer_class"] = HACOReplayBuffer

        if "intervention_start_stop_td" in kwargs:
            self.intervention_start_stop_td = kwargs["intervention_start_stop_td"]
            kwargs.pop("intervention_start_stop_td")
        else:
            # Default to set it True. We find this can improve the performance and user experience.
            self.intervention_start_stop_td = True

        self.extra_config = {}
        self.init_bc_steps = 0
        
        for k in ["no_done_for_positive", "no_done_for_negative", "reward_0_for_positive", "reward_0_for_negative",
                  "reward_n2_for_intervention", "reward_1_for_all", "use_weighted_reward", "remove_negative",
                  "adaptive_batch_size", "add_bc_loss", "only_bc_loss"]:
            if k in kwargs:
                v = kwargs.pop(k)
                assert v in ["True", "False"]
                v = v == "True"
                self.extra_config[k] = v
        for k in ["agent_data_ratio", "bc_loss_weight", "init_bc_steps", "thr_classifier"]:
            if k in kwargs:
                self.extra_config[k] = kwargs.pop(k)
                setattr(self, k, self.extra_config[k])
        self.switch2robot_thresh = 0
        self.switch2human_thresh = 0
        self.estimates = []
        self.q_value_bound = q_value_bound
        self.use_balance_sample = use_balance_sample
        super(PVPTD3, self).__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super(PVPTD3, self)._setup_model()
        if self.use_balance_sample:
            self.human_data_buffer = HACOReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs
            )
        else:
            self.human_data_buffer = self.replay_buffer
        from pvp.sb3.common.utils import get_schedule_fn
        self.classifier = TD3Policy(self.observation_space,
                    self.action_space,
                    get_schedule_fn(1e-4))
        self.classifier = self.classifier.to("cuda")
        self.classifier.set_training_mode(True)
            
    def compute_uncertainty(self, obs, actions):
        th_obs = th.from_numpy(np.expand_dims(obs, 0)).to(self.classifier.device)
        th_actions = th.from_numpy(np.expand_dims(actions, 0)).to(self.classifier.device)
        unc = self.classifier.critic(th_obs, th_actions)[0].item()
        return unc
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        stat_recorder = defaultdict(list)

        should_concat = False
        if self.replay_buffer.pos > 0 and self.human_data_buffer.pos > 0:
            replay_data_human = self.human_data_buffer.sample(
                int(batch_size), env=self._vec_normalize_env, return_all=True
            )
            human_data_size = len(replay_data_human.observations)
            human_data_size = max(1, self.extra_config["agent_data_ratio"] * human_data_size)
            human_data_size = int(human_data_size)
            should_concat = True

        elif self.human_data_buffer.pos > 0:
            replay_data = self.human_data_buffer.sample(
                batch_size, env=self._vec_normalize_env, return_all=True
            )
        elif self.replay_buffer.pos > 0:
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        else:
            gradient_steps = 0
        
        if (self.extra_config["only_bc_loss"]):
            gradient_steps_q = 0
        else:
            gradient_steps_q = gradient_steps
        
        for step in range(gradient_steps_q):
            self._n_updates += 1
            # Sample replay buffer

            if self.extra_config["adaptive_batch_size"]:
                if should_concat:
                    replay_data_agent = self.replay_buffer.sample(human_data_size, env=self._vec_normalize_env)
                    replay_data = concat_samples(replay_data_agent, replay_data_human)
            else:
                replay_data_agent = self.replay_buffer.sample(int(batch_size / 2), env=self._vec_normalize_env)
                replay_data_human = self.human_data_buffer.sample(int(batch_size / 2), env=self._vec_normalize_env)
                replay_data = concat_samples(replay_data_agent, replay_data_human)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions_behavior.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_behavior_values = self.critic(replay_data.observations, replay_data.actions_behavior)
            current_q_novice_values = self.critic(replay_data.observations, replay_data.actions_novice)

            stat_recorder["q_value_behavior"].append(current_q_behavior_values[0].mean().item())
            stat_recorder["q_value_novice"].append(current_q_novice_values[0].mean().item())

            # Compute critic loss
            critic_loss = []
            for (current_q_behavior, current_q_novice) in zip(current_q_behavior_values, current_q_novice_values):
                if self.intervention_start_stop_td:
                    l = 0.5 * F.mse_loss(
                        replay_data.stop_td * current_q_behavior, replay_data.stop_td * target_q_values
                    )
                else:
                    l = 0.5 * F.mse_loss(current_q_behavior, target_q_values)

                # ====== The key of Proxy Value Objective =====

                l += th.mean(
                    replay_data.interventions * self.cql_coefficient *
                    F.mse_loss(
                        current_q_behavior, self.q_value_bound * th.ones_like(current_q_behavior), reduction="none"
                    )
                )
                l += th.mean(
                    replay_data.interventions * self.cql_coefficient *
                    F.mse_loss(
                        current_q_novice, -self.q_value_bound * th.ones_like(current_q_behavior), reduction="none"
                    )
                )

                critic_loss.append(l)
            critic_loss = sum(critic_loss)

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            stat_recorder["critic_loss"] = critic_loss.item()
            
        if self.num_timesteps < self.init_bc_steps:
            gradient_steps_a = 0
        elif self.num_timesteps == self.init_bc_steps:
            gradient_steps_a = gradient_steps * self.init_bc_steps
        else:
            gradient_steps_a = gradient_steps * self.policy_delay * (int)(self.human_data_buffer.pos % self.policy_delay == 0)
        
        for step in range(gradient_steps_a):
                self.estimates = []
                self._n_updates += 1
                # Sample replay buffer
                replay_data = self.human_data_buffer.sample(batch_size, env=self._vec_normalize_env)

                # Compute actor loss
                new_action = self.actor(replay_data.observations)
                actor_loss = -self.critic.q1_forward(replay_data.observations, new_action).mean()

                # BC loss on human data
                bc_loss = F.mse_loss(replay_data.actions_behavior, new_action, reduction="none").mean(axis=-1)
                masked_bc_loss = (replay_data.interventions.flatten() * bc_loss).sum() / (
                    replay_data.interventions.flatten().sum() + 1e-5
                )
                # masked_bc_loss = masked_bc_loss.mean()

                if self.extra_config["only_bc_loss"]:
                    # actor_loss = masked_bc_loss  #.mean()
                    actor_loss = bc_loss.mean()  #.mean()

                else:
                    if self.extra_config["add_bc_loss"]:
                        actor_loss += masked_bc_loss * self.extra_config["bc_loss_weight"]
                        # actor_loss += bc_loss.mean() * self.extra_config["bc_loss_weight"]

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                stat_recorder["actor_loss"] = actor_loss.item()
                stat_recorder["masked_bc_loss"] = masked_bc_loss.item()
                stat_recorder["bc_loss"] = bc_loss.mean().item()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
        
        if self.num_timesteps < self.init_bc_steps:
            gradient_steps_c = 0
        elif self.num_timesteps == self.init_bc_steps:
            gradient_steps_c = self.init_bc_steps
        else:
            gradient_steps_c = self.policy_delay * 8 * (int)(self.human_data_buffer.pos % (8*self.policy_delay) == 0)
        for step in range(gradient_steps_c):
                    self._n_updates += 1
                    self.estimates = []
                    with th.no_grad():
                        replay_data_human = self.human_data_buffer.sample(int(batch_size), env=self._vec_normalize_env)
                        new_action, _ = self.predict(replay_data_human.observations.cpu().numpy())
                        new_action = th.Tensor(new_action).to(self.device)
                            
                    current_c_behavior = self.classifier.critic(replay_data_human.observations, replay_data_human.actions_behavior)[0]
                    
                    current_c_novice = self.classifier.critic(replay_data_human.observations, new_action)[0]
                    
                    no_overlap = (
                        ((replay_data_human.actions_behavior - new_action) ** 2).mean(dim=-1) > self.switch2robot_thresh * 1.5
                    ).float()
                    
                    loss_class = th.mean((current_c_behavior + 1) ** 2 + (current_c_novice * no_overlap - 1) ** 2)
                        
                    self.classifier.critic.optimizer.zero_grad()
                    loss_class.backward()
                    self.classifier.critic.optimizer.step()
                    stat_recorder["loss_class"] = loss_class.item()
        if self.num_timesteps == self.init_bc_steps:
            train_data = self.human_data_buffer._get_samples(np.arange(self.human_data_buffer.pos), env=self._vec_normalize_env)
            actions_train_data, _ = self.predict(observation=train_data.observations.cpu().numpy())
            actions_train_data = th.Tensor(actions_train_data).to(self.device)
            discre = th.mean((train_data.actions_behavior - actions_train_data) ** 2, dim = -1)
            self.switch2robot_thresh = th.mean(discre).item()
            with th.no_grad():
                replay_data_human = self.human_data_buffer._get_samples(np.arange(self.human_data_buffer.pos), env=self._vec_normalize_env)
                new_action, _ = self.predict(replay_data_human.observations.cpu().numpy())
                current_c_novice = self.classifier.critic(replay_data_human.observations, th.Tensor(new_action).to(self.device))[0]
            self.estimates = current_c_novice.squeeze().tolist()
            self.switch2human_thresh = th.quantile(current_c_novice, self.extra_config["thr_classifier"]).item()
            
            
        stat_recorder["switch2human_thresh"] = self.switch2human_thresh
        stat_recorder["switch2robot_thresh"] = self.switch2robot_thresh
        
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for key, values in stat_recorder.items():
            self.logger.record("train/{}".format(key), np.mean(values))

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        if infos[0]["takeover"] or infos[0]["takeover_start"]:
            replay_buffer = self.human_data_buffer
        super(PVPTD3, self)._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    def save_replay_buffer(
        self, path_human: Union[str, pathlib.Path, io.BufferedIOBase], path_replay: Union[str, pathlib.Path,
                                                                                          io.BufferedIOBase]
    ) -> None:
        save_to_pkl(path_human, self.human_data_buffer, self.verbose)
        super(PVPTD3, self).save_replay_buffer(path_replay)

    def load_replay_buffer(
        self,
        path_human: Union[str, pathlib.Path, io.BufferedIOBase],
        path_replay: Union[str, pathlib.Path, io.BufferedIOBase],
        truncate_last_traj: bool = True,
    ) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        :param truncate_last_traj: When using ``HerReplayBuffer`` with online sampling:
            If set to ``True``, we assume that the last trajectory in the replay buffer was finished
            (and truncate it).
            If set to ``False``, we assume that we continue the same trajectory (same episode).
        """
        self.human_data_buffer = load_from_pkl(path_human, self.verbose)
        assert isinstance(
            self.human_data_buffer, ReplayBuffer
        ), "The replay buffer must inherit from ReplayBuffer class"

        # Backward compatibility with SB3 < 2.1.0 replay buffer
        # Keep old behavior: do not handle timeout termination separately
        if not hasattr(self.human_data_buffer, "handle_timeout_termination"):  # pragma: no cover
            self.human_data_buffer.handle_timeout_termination = False
            self.human_data_buffer.timeouts = np.zeros_like(self.replay_buffer.dones)
        super(PVPTD3, self).load_replay_buffer(path_replay, truncate_last_traj)
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
        deterministic=None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        
        #for remote in self.remotes:
        #    remote.send(("set_training_mode", False))
        #for remote in self.remotes:
        #    remote.recv()

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            raise NotImplementedError
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(
                learning_starts, action_noise, env.num_envs, deterministic=deterministic #self.policy_choice
            )

            th_obs = th.from_numpy(self._last_obs).to(self.device)
            th_actions = th.from_numpy(actions).to(self.device)
            unc = self.classifier.critic(th_obs, th_actions)[0].item()

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)
            
            self.logger.record("train/human_involved_steps", infos[0]["total_human_involved_steps"])
            if not infos[0]["takeover"]:
                self.estimates.append(unc)

            self.num_timesteps += env.num_envs
            self.since_last_reset += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(
                    num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:

                    if len(self.estimates) > 25:
                        self.switch2human_thresh = th.quantile(th.Tensor(self.estimates).squeeze(), self.extra_config["thr_classifier"]).item()
                        
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)
                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

                    # PZH: We add a callback here to allow doing something after each episode is ended.
                    self._on_episode_end(infos[idx])

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        save_timesteps: int = 2000,
        buffer_save_timesteps: int = 2000,
        save_path_human: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        save_path_replay: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        save_buffer: bool = True,
        load_buffer: bool = False,
        load_path_human: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        load_path_replay: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        warmup: bool = False,
        warmup_steps: int = 5000,
    ) -> "OffPolicyAlgorithm":

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )
        if load_buffer:
            self.load_replay_buffer(load_path_human, load_path_replay)
        callback.on_training_start(locals(), globals())
        if warmup:
            assert load_buffer, "warmup is useful only when load buffer"
            print("Start warmup with steps: " + str(warmup_steps))
            self.train(batch_size=self.batch_size, gradient_steps=warmup_steps)

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break
            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
            if save_buffer and self.num_timesteps > 0 and self.num_timesteps % buffer_save_timesteps == 0:
                buffer_location_human = os.path.join(
                    save_path_human, "human_buffer_" + str(self.num_timesteps) + ".pkl"
                )
                buffer_location_replay = os.path.join(
                    save_path_replay, "replay_buffer_" + str(self.num_timesteps) + ".pkl"
                )
                logger.info("Saving..." + str(buffer_location_human))
                logger.info("Saving..." + str(buffer_location_replay))
                self.save_replay_buffer(buffer_location_human, buffer_location_replay)

        callback.on_training_end()

        return self
