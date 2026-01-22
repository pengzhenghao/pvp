"""
Conservative Q-Learning (CQL) implementation based on TD3.

Paper: "Conservative Q-Learning for Offline Reinforcement Learning"
https://arxiv.org/abs/2006.04779

CQL adds a conservative regularization term to the Q-function that minimizes
Q-values for out-of-distribution actions while maximizing Q-values for actions
in the dataset.

Key design (consistent with original paper):
- Critic: TD loss + CQL penalty (logsumexp(Q) - Q(s, a_data))
- Actor: Standard Q-value maximization (-Q(s, π(s)))
- The conservative penalty is ONLY on the critic, NOT on the actor

Optional: Set use_td3_bc=True to combine CQL with TD3+BC style actor loss.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from pvp.sb3.common.buffers import ReplayBuffer
from pvp.sb3.common.noise import ActionNoise
from pvp.sb3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from pvp.sb3.common.utils import polyak_update
from pvp.sb3.td3.policies import TD3Policy
from pvp.sb3.td3.td3 import TD3


class CQL(TD3):
    """
    Conservative Q-Learning (CQL) based on TD3.
    
    CQL adds a conservative penalty to prevent overestimation of Q-values
    for out-of-distribution actions, which is critical for offline RL.
    
    The CQL loss for the critic is:
        CQL_loss = cql_alpha * (logsumexp(Q(s, a_sampled)) - Q(s, a_data))
    
    Where a_sampled includes:
        - Random actions uniformly sampled from the action space
        - Actions from the current policy
    
    :param cql_alpha: Weight for the CQL conservative penalty (default: 1.0)
    :param num_random_actions: Number of random actions to sample for CQL loss (default: 10)
    :param cql_temp: Temperature for logsumexp in CQL loss (default: 1.0)
    :param with_lagrange: Whether to use Lagrange multiplier for automatic cql_alpha tuning (default: False)
    :param lagrange_threshold: Target value for CQL penalty when using Lagrange (default: 10.0)
    :param bc_loss_weight: Weight for BC loss in actor update (default: 1.0)
    :param use_td3_bc: Whether to combine Q-learning loss with BC loss for actor (default: False)
    """
    
    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        monitor_wrapper=False,
        bc_loss_weight: float = 1.0,
        use_td3_bc: bool = False,
        td3_bc_alpha: float = 2.5,
        # CQL specific parameters
        cql_alpha: float = 1.0,
        num_random_actions: int = 10,
        cql_temp: float = 1.0,
        with_lagrange: bool = False,
        lagrange_threshold: float = 10.0,
    ):
        super(CQL, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            monitor_wrapper=monitor_wrapper,
            bc_loss_weight=bc_loss_weight,
            use_td3_bc=use_td3_bc,
            td3_bc_alpha=td3_bc_alpha,
        )
        
        # CQL specific parameters
        self.cql_alpha = cql_alpha
        self.num_random_actions = num_random_actions
        self.cql_temp = cql_temp
        self.with_lagrange = with_lagrange
        self.lagrange_threshold = lagrange_threshold
        
        # For Lagrange multiplier
        if self.with_lagrange:
            self.log_alpha_cql = th.tensor(np.log(self.cql_alpha), requires_grad=True, device=self.device)
            self.alpha_cql_optimizer = th.optim.Adam([self.log_alpha_cql], lr=learning_rate)
    
    def _compute_cql_loss(self, observations, actions_data: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Compute the CQL conservative penalty.
        
        CQL loss = logsumexp(Q(s, a_sampled)) - Q(s, a_data)
        
        Where a_sampled includes random actions and policy actions.
        This implementation handles both tensor and dict observations (for CNN policies).
        
        Returns:
            cql_loss_q1: CQL loss for Q1
            cql_loss_q2: CQL loss for Q2
            cql_alpha: Current CQL alpha (may be learned if using Lagrange)
        """
        batch_size = actions_data.shape[0]
        action_dim = actions_data.shape[-1]
        
        # Get current alpha (learned or fixed)
        if self.with_lagrange:
            cql_alpha = th.exp(self.log_alpha_cql).clamp(min=0.0, max=1e6)
        else:
            cql_alpha = self.cql_alpha
        
        # Get current policy actions
        with th.no_grad():
            policy_actions = self.actor(observations)
        
        # Q-values for policy actions
        q1_policy, q2_policy = self.critic(observations, policy_actions)
        
        # Q-values for data actions
        q1_data, q2_data = self.critic(observations, actions_data)
        
        # Sample random actions and compute Q-values
        # For dict observations (CNN), we compute Q-values for each random action set separately
        q1_random_list = []
        q2_random_list = []
        
        for _ in range(self.num_random_actions):
            # Sample random actions uniformly from [-1, 1]
            random_actions = th.FloatTensor(batch_size, action_dim).uniform_(-1, 1).to(self.device)
            q1_rand, q2_rand = self.critic(observations, random_actions)
            q1_random_list.append(q1_rand)
            q2_random_list.append(q2_rand)
        
        # Stack random Q-values: (batch, num_random)
        q1_random = th.cat(q1_random_list, dim=1)
        q2_random = th.cat(q2_random_list, dim=1)
        
        # Concatenate all Q-values for logsumexp
        # Shape: (batch, num_random + 1)
        q1_cat = th.cat([q1_random, q1_policy], dim=1)
        q2_cat = th.cat([q2_random, q2_policy], dim=1)
        
        # Compute logsumexp with temperature
        # logsumexp(Q/temp) * temp
        q1_logsumexp = th.logsumexp(q1_cat / self.cql_temp, dim=1, keepdim=True) * self.cql_temp
        q2_logsumexp = th.logsumexp(q2_cat / self.cql_temp, dim=1, keepdim=True) * self.cql_temp
        
        # CQL loss = logsumexp(Q(s, a_sampled)) - Q(s, a_data)
        cql_loss_q1 = (q1_logsumexp - q1_data).mean()
        cql_loss_q2 = (q2_logsumexp - q2_data).mean()
        
        return cql_loss_q1, cql_loss_q2, cql_alpha
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Train the CQL agent.
        
        This overrides TD3's train method to add the CQL conservative penalty
        to the critic loss.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        
        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        
        actor_losses, critic_losses, cql_losses = [], [], []
        
        for _ in range(gradient_steps):
            self._n_updates += 1
            
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
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
            current_q_values = self.critic(replay_data.observations, replay_data.actions_behavior)
            q1, q2 = current_q_values
            
            # Compute standard TD3 critic loss (Bellman error)
            bellman_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            
            # Compute CQL conservative penalty
            cql_loss_q1, cql_loss_q2, cql_alpha = self._compute_cql_loss(
                replay_data.observations, 
                replay_data.actions_behavior
            )
            cql_loss = cql_loss_q1 + cql_loss_q2
            
            # Total critic loss = Bellman loss + CQL penalty
            critic_loss = bellman_loss + cql_alpha * cql_loss
            
            critic_losses.append(bellman_loss.item())
            cql_losses.append(cql_loss.item())
            
            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
            # Update Lagrange multiplier if using automatic alpha tuning
            if self.with_lagrange:
                alpha_loss = -self.log_alpha_cql.exp() * (cql_loss.detach() - self.lagrange_threshold)
                self.alpha_cql_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_cql_optimizer.step()
            
            # Delayed policy updates
            if (self._n_updates % self.policy_delay == 0):
                if self.num_timesteps > 0:
                    # Get policy actions
                    pi_actions = self.actor(replay_data.observations)
                    
                    # CQL Actor Loss: Standard Q-value maximization (as per original paper)
                    # actor_loss = -Q(s, π(s))
                    q_pi = self.critic.q1_forward(replay_data.observations, pi_actions)
                    actor_loss = -q_pi.mean()
                    actor_losses.append(actor_loss.item())
                    
                    # Compute BC loss for monitoring (not used in optimization by default)
                    bc_loss = F.mse_loss(pi_actions, replay_data.actions_behavior)
                    
                    # Optional: TD3+BC style - combine Q-learning with BC loss
                    if self.use_td3_bc:
                        # Normalize Q-loss as in TD3+BC paper
                        q_data = self.critic.q1_forward(replay_data.observations, replay_data.actions_behavior)
                        lmbda = self.td3_bc_alpha / th.abs(q_data).mean().detach()
                        actor_loss = -lmbda * q_pi.mean() + bc_loss
                    
                    # Optimize the actor
                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor.optimizer.step()
                
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
        
        # Logging
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/q_values_mean", th.mean(th.abs(q1) + th.abs(q2)).item() * 0.5)
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/bc_loss", bc_loss.item())
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/cql_loss", np.mean(cql_losses))
        if self.with_lagrange:
            self.logger.record("train/cql_alpha", th.exp(self.log_alpha_cql).item())
        else:
            self.logger.record("train/cql_alpha", self.cql_alpha)
        
        # ===== Additional training metrics (same as TD3) =====
        if len(actor_losses) > 0:
            with th.no_grad():
                data_actions = replay_data.actions_behavior
                policy_actions = self.actor(replay_data.observations)
                
                # Per-dimension BC loss
                bc_loss_steering = F.mse_loss(policy_actions[:, 0], data_actions[:, 0])
                bc_loss_accel = F.mse_loss(policy_actions[:, 1], data_actions[:, 1])
                self.logger.record("train/bc_loss_steering", bc_loss_steering.item())
                self.logger.record("train/bc_loss_accel", bc_loss_accel.item())
                
                # Mean and absolute mean of DATA actions
                self.logger.record("train/data_mean_steering", data_actions[:, 0].mean().item())
                self.logger.record("train/data_mean_steering_abs", th.abs(data_actions[:, 0]).mean().item())
                self.logger.record("train/data_mean_accel", data_actions[:, 1].mean().item())
                self.logger.record("train/data_mean_accel_abs", th.abs(data_actions[:, 1]).mean().item())
                
                # Mean and absolute mean of POLICY actions
                self.logger.record("train/policy_mean_steering", policy_actions[:, 0].mean().item())
                self.logger.record("train/policy_mean_steering_abs", th.abs(policy_actions[:, 0]).mean().item())
                self.logger.record("train/policy_mean_accel", policy_actions[:, 1].mean().item())
                self.logger.record("train/policy_mean_accel_abs", th.abs(policy_actions[:, 1]).mean().item())
                
                # Action difference (L2 norm)
                action_diff_l2 = th.norm(policy_actions - data_actions, dim=1).mean().item()
                self.logger.record("train/action_diff_l2", action_diff_l2)
        
        import wandb
        wandb.log(self.logger.name_to_value, step=self.num_timesteps)
    
    def _excluded_save_params(self) -> List[str]:
        excluded = super(CQL, self)._excluded_save_params()
        if self.with_lagrange:
            excluded.append("log_alpha_cql")
        return excluded
    
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts, _ = super(CQL, self)._get_torch_save_params()
        if self.with_lagrange:
            state_dicts.append("alpha_cql_optimizer")
        return state_dicts, []
