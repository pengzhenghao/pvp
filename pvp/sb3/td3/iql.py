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


class ValueNetwork(nn.Module):
    """
    Value network V(s) for IQL.
    Uses the same architecture as the critic but outputs a single value.
    """
    def __init__(self, features_extractor, features_dim: int, net_arch: List[int] = [256, 256]):
        super(ValueNetwork, self).__init__()
        self.features_extractor = features_extractor
        
        # Build MLP layers
        layers = []
        last_dim = features_dim
        for hidden_dim in net_arch:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, obs):
        features = self.features_extractor(obs)
        return self.mlp(features)


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
        
        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        super(IQL, self)._setup_model()
        
        # Create value network V(s) using same features extractor architecture
        # Get features dimension from the critic
        features_dim = self.critic.features_extractor.features_dim
        
        # Create a new features extractor for value network (same architecture)
        self.value_net = ValueNetwork(
            features_extractor=self.policy.make_features_extractor(),
            features_dim=features_dim,
            net_arch=[256, 256]
        ).to(self.device)
        
        # Create optimizer for value network
        self.value_optimizer = th.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
    
    def _expectile_loss(self, diff: th.Tensor, expectile: float) -> th.Tensor:
        """
        Compute expectile loss (asymmetric L2 loss).
        
        L_tau(u) = |tau - 1(u < 0)| * u^2
        
        When tau > 0.5:
        - Underestimating (diff > 0) is penalized more
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
        
        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        
        actor_losses, critic_losses, value_losses = [], [], []
        
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
            v_pred = self.value_net(replay_data.observations)
            
            # Expectile loss: pushes V towards max Q when tau > 0.5
            value_loss = self._expectile_loss(q_data - v_pred, self.iql_tau)
            value_losses.append(value_loss.item())
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            # ==================== Update Q Networks ====================
            # Q(s, a) is trained with TD learning using V(s') instead of max Q(s', a')
            
            with th.no_grad():
                # Use V(s') instead of max_a' Q(s', a')
                next_v = self.value_net(replay_data.next_observations)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_v
            
            # Get current Q estimates
            current_q_values = self.critic(replay_data.observations, replay_data.actions_behavior)
            q1, q2 = current_q_values
            
            # Compute critic loss
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            critic_losses.append(critic_loss.item())
            
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
            # ==================== Update Policy (Actor) ====================
            # Policy is extracted using advantage-weighted regression (AWR)
            # π(a|s) ∝ exp(β * A(s, a)) where A(s, a) = Q(s, a) - V(s)
            
            if self._n_updates % self.policy_delay == 0:
                # Compute advantages
                with th.no_grad():
                    v_s = self.value_net(replay_data.observations)
                    q1, q2 = self.critic(replay_data.observations, replay_data.actions_behavior)
                    q_s_a = th.min(q1, q2)
                    advantage = q_s_a - v_s
                    
                    # Compute weights: exp(β * A) with clipping for stability
                    weights = th.exp(self.iql_beta * advantage)
                    weights = th.clamp(weights, max=self.clip_score)
                
                # Get policy actions
                pi_actions = self.actor(replay_data.observations)
                
                # Advantage-weighted BC loss
                # Minimize: -E[w(s,a) * log π(a|s)] ≈ E[w(s,a) * ||π(s) - a||^2]
                bc_diff = (pi_actions - replay_data.actions_behavior) ** 2
                actor_loss = (weights * bc_diff.mean(dim=1, keepdim=True)).mean()
                actor_losses.append(actor_loss.item())
                
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                
                # Update target networks
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
        
        # Logging
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/iql_tau", self.iql_tau)
        self.logger.record("train/iql_beta", self.iql_beta)
        
        import wandb
        wandb.log(self.logger.name_to_value, step=self.num_timesteps)
    
    def _excluded_save_params(self) -> List[str]:
        return super(IQL, self)._excluded_save_params() + ["value_net", "value_optimizer"]
    
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts, _ = super(IQL, self)._get_torch_save_params()
        state_dicts.extend(["value_net", "value_optimizer"])
        return state_dicts, []
