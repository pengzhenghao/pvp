"""
Implicit Q-Learning (IQL) implementation based on TD3.

Paper: "Offline Reinforcement Learning with Implicit Q-Learning"
https://arxiv.org/abs/2110.06169

IQL avoids querying Q-values for out-of-distribution actions by:
1. Using expectile regression to train V(s) to approximate max_a Q(s,a)
2. Training Q(s,a) with standard TD but using V(s') instead of max Q(s',a')
3. Extracting policy using advantage-weighted regression (AWR)

Key hyperparameters:
- tau (expectile): Controls how V approximates max Q (0.5=mean, 0.7-0.9 typical)
- beta (temperature): Controls policy extraction greediness (higher=more greedy)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F
import torch.nn as nn

from pvp.sb3.common.buffers import ReplayBuffer
from pvp.sb3.common.noise import ActionNoise
from pvp.sb3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from pvp.sb3.common.utils import polyak_update
from pvp.sb3.td3.policies import TD3Policy
from pvp.sb3.td3.td3 import TD3


class IQL(TD3):
    """
    Implicit Q-Learning (IQL) for offline RL.
    
    IQL key ideas:
    1. V(s) is trained with expectile regression on Q(s, a_data) to approximate max_a Q(s, a)
    2. Q(s, a) is trained with TD learning using V(s') instead of max_a' Q(s', a')
    3. Policy is extracted using advantage-weighted regression (AWR)
    
    :param iql_tau: Expectile parameter for value function (default: 0.7)
        - 0.5 = mean (no max approximation)
        - closer to 1.0 = better max approximation but higher variance
        - typical values: 0.7-0.9
    :param iql_beta: Temperature for advantage-weighted policy extraction (default: 3.0)
        - higher = more greedy (exploit high-advantage actions)
        - lower = more uniform (explore more)
    :param clip_score: Maximum value for advantage weights to prevent explosion (default: 100.0)
    :param max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
    """
    
    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 1,  # IQL typically updates all networks together
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
        # IQL specific parameters
        iql_tau: float = 0.7,
        iql_beta: float = 3.0,
        clip_score: float = 100.0,
        max_grad_norm: float = 1.0,
        reward_normalize: bool = False,  # Whether to normalize rewards
        normalize_advantage: bool = True,  # Whether to normalize advantage (disable for vanilla IQL)
    ):
        # Don't initialize model yet - we need to set up value network first
        super(IQL, self).__init__(
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
            _init_setup_model=False,  # We'll setup manually
            monitor_wrapper=monitor_wrapper,
            bc_loss_weight=bc_loss_weight,
            use_td3_bc=use_td3_bc,
            td3_bc_alpha=td3_bc_alpha,
        )
        
        # IQL specific parameters
        self.iql_tau = iql_tau
        self.iql_beta = iql_beta
        self.clip_score = clip_score
        self.max_grad_norm = max_grad_norm
        self.reward_normalize = reward_normalize
        self.normalize_advantage = normalize_advantage
        self._reward_mean = None
        self._reward_std = None
        
        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        super(IQL, self)._setup_model()
        
        # Create value network V(s) - use Q1 network architecture but output single value
        # We'll create a simple MLP that takes the same features as the critic
        # For simplicity, we use the critic's Q1 network as a template
        
        # Create value network with same structure as critic but single output
        # Use the policy's actor features extractor (which handles dict obs)
        self.value_features_extractor = self.policy.make_features_extractor()
        self.value_features_extractor = self.value_features_extractor.to(self.device)
        features_dim = self.value_features_extractor.features_dim
        
        # Simple MLP for value head
        self.value_mlp = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        
        # Initialize weights with small values for stability
        for layer in self.value_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0.0)
        
        # Create optimizer for value network (features extractor + mlp)
        value_params = list(self.value_features_extractor.parameters()) + list(self.value_mlp.parameters())
        self.value_optimizer = th.optim.Adam(value_params, lr=self.learning_rate)
    
    def _get_value(self, obs) -> th.Tensor:
        """Get V(s) from value network."""
        features = self.value_features_extractor(obs)
        return self.value_mlp(features)
    
    def _expectile_loss(self, diff: th.Tensor, expectile: float) -> th.Tensor:
        """
        Compute expectile loss (asymmetric L2 loss).
        
        L_tau(u) = |tau - 1(u < 0)| * u^2
        
        When tau > 0.5:
        - Underestimating (diff > 0, meaning Q > V) is penalized more
        - This pushes V towards max Q
        """
        weight = th.where(diff > 0, expectile, 1 - expectile)
        return (weight * (diff ** 2)).mean()
    
    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        """
        Train IQL agent.
        
        IQL training consists of:
        1. Train V(s) with expectile regression on Q(s, a_data)
        2. Train Q(s, a) with TD learning using V(s')
        3. Train policy with advantage-weighted regression
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        self.value_features_extractor.train()
        self.value_mlp.train()
        
        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        
        actor_losses, critic_losses, value_losses, bc_losses = [], [], [], []
        advantages_mean, advantages_std = [], []
        
        for _ in range(gradient_steps):
            self._n_updates += 1
            
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # ==================== Update Value Network ====================
            # V(s) is trained with expectile regression on Q(s, a_data)
            # This implicitly learns max_a Q(s, a) without querying OOD actions
            
            with th.no_grad():
                # Get Q-values for data actions (use min of two critics for stability)
                q1, q2 = self.critic(replay_data.observations, replay_data.actions_behavior)
                q_data = th.min(q1, q2)
            
            # Get V(s)
            v_pred = self._get_value(replay_data.observations)
            
            # Expectile loss: pushes V towards max Q when tau > 0.5
            # diff = Q - V, positive diff means Q > V (underestimating)
            value_loss = self._expectile_loss(q_data - v_pred, self.iql_tau)
            value_losses.append(value_loss.item())
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            # Gradient clipping for stability
            th.nn.utils.clip_grad_norm_(
                list(self.value_features_extractor.parameters()) + list(self.value_mlp.parameters()),
                self.max_grad_norm
            )
            self.value_optimizer.step()
            
            # ==================== Update Q Networks ====================
            # Q(s, a) is trained with TD learning using V(s') instead of max Q(s', a')
            
            with th.no_grad():
                # Use V(s') instead of max_a' Q(s', a')
                next_v = self._get_value(replay_data.next_observations)
                
                # Apply reward normalization if enabled
                rewards = replay_data.rewards
                if self.reward_normalize:
                    # Compute running stats on first call or use stored stats
                    if self._reward_mean is None:
                        # Compute stats from current batch (will be updated over time)
                        self._reward_mean = rewards.mean().item()
                        self._reward_std = rewards.std().item() + 1e-8
                    else:
                        # Exponential moving average for stability
                        alpha = 0.01
                        self._reward_mean = (1 - alpha) * self._reward_mean + alpha * rewards.mean().item()
                        self._reward_std = (1 - alpha) * self._reward_std + alpha * (rewards.std().item() + 1e-8)
                    
                    # Normalize rewards: zero mean, unit variance
                    rewards = (rewards - self._reward_mean) / self._reward_std
                
                target_q = rewards + (1 - replay_data.dones) * self.gamma * next_v
            
            # Get current Q estimates
            current_q_values = self.critic(replay_data.observations, replay_data.actions_behavior)
            q1, q2 = current_q_values
            
            # Compute critic loss
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            critic_losses.append(critic_loss.item())
            
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            # Gradient clipping for stability
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic.optimizer.step()
            
            # ==================== Update Policy (Actor) ====================
            # Policy is extracted using advantage-weighted regression (AWR)
            # π(a|s) ∝ exp(β * A(s, a)) where A(s, a) = Q(s, a) - V(s)
            
            if self._n_updates % self.policy_delay == 0:
                # Compute advantages
                with th.no_grad():
                    v_s = self._get_value(replay_data.observations)
                    q1, q2 = self.critic(replay_data.observations, replay_data.actions_behavior)
                    q_s_a = th.min(q1, q2)
                    advantage = q_s_a - v_s
                    
                    # Log advantage statistics
                    advantages_mean.append(advantage.mean().item())
                    advantages_std.append(advantage.std().item())
                    
                    # Normalize advantages for stability (optional)
                    if self.normalize_advantage:
                        adv_std = advantage.std()
                        if adv_std > 1e-8:
                            advantage_for_weights = advantage / adv_std
                        else:
                            advantage_for_weights = advantage
                    else:
                        # Vanilla IQL: use raw advantage
                        advantage_for_weights = advantage
                    
                    # Compute weights: exp(β * A) with clipping for stability
                    weights = th.exp(self.iql_beta * advantage_for_weights)
                    weights = th.clamp(weights, max=self.clip_score)
                    # Normalize weights to have mean 1 for stable gradients (always do this)
                    weights = weights / weights.mean()
                
                # Get policy actions
                pi_actions = self.actor(replay_data.observations)
                
                # Advantage-weighted BC loss
                # Minimize: E[w(s,a) * ||π(s) - a||^2]
                bc_diff = (pi_actions - replay_data.actions_behavior) ** 2
                actor_loss = (weights * bc_diff.sum(dim=1, keepdim=True)).mean()
                actor_losses.append(actor_loss.item())
                
                # Also compute pure BC loss for monitoring
                pure_bc_loss = bc_diff.mean()
                bc_losses.append(pure_bc_loss.item())
                
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                # Gradient clipping for stability
                th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()
                
                # Update target networks
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
        
        # Logging
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/bc_loss", np.mean(bc_losses))
            self.logger.record("train/advantage_mean", np.mean(advantages_mean))
            self.logger.record("train/advantage_std", np.mean(advantages_std))
        self.logger.record("train/iql_tau", self.iql_tau)
        self.logger.record("train/iql_beta", self.iql_beta)
        self.logger.record("train/reward_normalize", int(self.reward_normalize))
        if self.reward_normalize and self._reward_mean is not None:
            self.logger.record("train/reward_mean", self._reward_mean)
            self.logger.record("train/reward_std", self._reward_std)
        
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
                
                # IQL specific: V-value statistics
                v_pred = self._get_value(replay_data.observations)
                self.logger.record("train/v_value_mean", v_pred.mean().item())
                
                # ================================================================
                # IQL Diagnostic Metrics: Is IQL != BC?
                # ================================================================
                # Re-compute advantages and weights for detailed logging
                v_s = self._get_value(replay_data.observations)
                q1, q2 = self.critic(replay_data.observations, replay_data.actions_behavior)
                q_s_a = th.min(q1, q2)
                advantage_raw = q_s_a - v_s
                
                # 1. Raw advantage statistics (before normalization)
                adv_std_raw = advantage_raw.std().item()
                adv_min = advantage_raw.min().item()
                adv_max = advantage_raw.max().item()
                adv_range = adv_max - adv_min
                self.logger.record("train/iql_advantage_std_raw", adv_std_raw)
                self.logger.record("train/iql_advantage_min", adv_min)
                self.logger.record("train/iql_advantage_max", adv_max)
                self.logger.record("train/iql_advantage_range", adv_range)
                
                # Advantage percentiles
                adv_flat = advantage_raw.flatten()
                self.logger.record("train/iql_advantage_p10", th.quantile(adv_flat, 0.1).item())
                self.logger.record("train/iql_advantage_p25", th.quantile(adv_flat, 0.25).item())
                self.logger.record("train/iql_advantage_p50", th.quantile(adv_flat, 0.5).item())
                self.logger.record("train/iql_advantage_p75", th.quantile(adv_flat, 0.75).item())
                self.logger.record("train/iql_advantage_p90", th.quantile(adv_flat, 0.9).item())
                
                # 2. exp(beta * advantage) statistics - BEFORE normalization
                # This shows the raw variation that beta*advantage produces
                if adv_std_raw > 1e-8:
                    advantage_normalized = advantage_raw / adv_std_raw
                else:
                    advantage_normalized = advantage_raw
                
                exp_weights_raw = th.exp(self.iql_beta * advantage_normalized)
                exp_weights_clipped = th.clamp(exp_weights_raw, max=self.clip_score)
                
                self.logger.record("train/iql_exp_weight_mean", exp_weights_clipped.mean().item())
                self.logger.record("train/iql_exp_weight_std", exp_weights_clipped.std().item())
                self.logger.record("train/iql_exp_weight_min", exp_weights_clipped.min().item())
                self.logger.record("train/iql_exp_weight_max", exp_weights_clipped.max().item())
                self.logger.record("train/iql_exp_weight_ratio_max_min", 
                                   (exp_weights_clipped.max() / (exp_weights_clipped.min() + 1e-8)).item())
                
                # 3. Effective Sample Size (ESS): measures how much weighting differs from uniform
                # ESS = (sum(w))^2 / sum(w^2), normalized by n gives ESS_ratio in [0, 1]
                # ESS_ratio = 1.0 means uniform weights (IQL = BC)
                # ESS_ratio << 1.0 means weights are concentrated (IQL != BC)
                weights_normalized = exp_weights_clipped / exp_weights_clipped.sum()
                ess = 1.0 / (weights_normalized ** 2).sum()
                ess_ratio = ess.item() / len(weights_normalized)  # normalized to [0, 1]
                self.logger.record("train/iql_effective_sample_size", ess.item())
                self.logger.record("train/iql_ess_ratio", ess_ratio)  # 1.0 = uniform = BC, <1.0 = selective = IQL
                
                # 4. Weight entropy: another measure of uniformity
                # High entropy = uniform = BC, Low entropy = selective = IQL
                weight_entropy = -(weights_normalized * th.log(weights_normalized + 1e-10)).sum()
                max_entropy = np.log(len(weights_normalized))  # entropy of uniform distribution
                entropy_ratio = weight_entropy.item() / max_entropy  # normalized to [0, 1]
                self.logger.record("train/iql_weight_entropy", weight_entropy.item())
                self.logger.record("train/iql_entropy_ratio", entropy_ratio)  # 1.0 = uniform = BC
                
                # 5. Correlation between reward and advantage: "Do high-reward samples get higher advantage?"
                # This validates that IQL is correctly identifying good actions
                rewards_flat = replay_data.rewards.flatten()
                adv_flat = advantage_raw.flatten()
                
                # Pearson correlation
                rewards_centered = rewards_flat - rewards_flat.mean()
                adv_centered = adv_flat - adv_flat.mean()
                corr_num = (rewards_centered * adv_centered).sum()
                corr_denom = (th.sqrt((rewards_centered ** 2).sum()) * th.sqrt((adv_centered ** 2).sum()) + 1e-8)
                reward_advantage_corr = (corr_num / corr_denom).item()
                self.logger.record("train/iql_reward_advantage_corr", reward_advantage_corr)
                
                # 6. Advantage for high-reward vs low-reward samples
                # Split by median reward
                reward_median = th.median(rewards_flat)
                high_reward_mask = rewards_flat >= reward_median
                low_reward_mask = rewards_flat < reward_median
                
                if high_reward_mask.sum() > 0 and low_reward_mask.sum() > 0:
                    adv_high_reward = adv_flat[high_reward_mask].mean().item()
                    adv_low_reward = adv_flat[low_reward_mask].mean().item()
                    self.logger.record("train/iql_advantage_high_reward", adv_high_reward)
                    self.logger.record("train/iql_advantage_low_reward", adv_low_reward)
                    self.logger.record("train/iql_advantage_gap", adv_high_reward - adv_low_reward)
                    
                    # Weight for high-reward vs low-reward samples
                    weight_high_reward = exp_weights_clipped.flatten()[high_reward_mask].mean().item()
                    weight_low_reward = exp_weights_clipped.flatten()[low_reward_mask].mean().item()
                    self.logger.record("train/iql_weight_high_reward", weight_high_reward)
                    self.logger.record("train/iql_weight_low_reward", weight_low_reward)
                    self.logger.record("train/iql_weight_ratio_high_low", 
                                       weight_high_reward / (weight_low_reward + 1e-8))
                
                # 7. Q-value and V-value statistics
                self.logger.record("train/iql_q_mean", q_s_a.mean().item())
                self.logger.record("train/iql_q_std", q_s_a.std().item())
                self.logger.record("train/iql_v_mean", v_s.mean().item())
                self.logger.record("train/iql_v_std", v_s.std().item())
                
                # 8. Summary: Is IQL effectively different from BC?
                # IQL_BC_divergence: higher = more different from BC
                # Combines ESS ratio (inverted) and weight variance
                iql_bc_divergence = (1.0 - ess_ratio) + exp_weights_clipped.std().item()
                self.logger.record("train/iql_bc_divergence", iql_bc_divergence)
        
        import wandb
        wandb.log(self.logger.name_to_value, step=self.num_timesteps)
    
    def _excluded_save_params(self) -> List[str]:
        return super(IQL, self)._excluded_save_params() + [
            "value_features_extractor", "value_mlp", "value_optimizer"
        ]
    
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts, _ = super(IQL, self)._get_torch_save_params()
        state_dicts.extend(["value_features_extractor", "value_mlp", "value_optimizer"])
        return state_dicts, []
