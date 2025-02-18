"""
Implemented based on SAC (as it uses stochastic policy)
"""
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from pvp.sb3.common.noise import ActionNoise
from pvp.sb3.common.type_aliases import GymEnv, Schedule
from pvp.sb3.common.utils import polyak_update
from pvp.sb3.haco.haco_buffer import HACOReplayBuffer
from pvp.sb3.haco.policies import HACOPolicy
from pvp.sb3.sac import SAC
from collections import defaultdict
import torch


def biased_bce_with_logits(adv1, adv2, y, bias=1.0, cbias = 0):
    # Apply the log-sum-exp trick.
    # y = 1 if we prefer x2 to x1
    # We need to implement the numerical stability trick.

    logit21 = adv2 - bias * adv1 - cbias
    logit12 = adv1 - bias * adv2
    max21 = torch.clamp(-logit21, min=0, max=None)
    max12 = torch.clamp(-logit12, min=0, max=None)
    nlp21 = torch.log(torch.exp(-max21) + torch.exp(-logit21 - max21)) + max21
    nlp12 = torch.log(torch.exp(-max12) + torch.exp(-logit12 - max12)) + max12
    loss = y * nlp21 + (1 - y) * nlp12
    loss = loss.mean()

    # Now compute the accuracy
    with torch.no_grad():
        accuracy = ((adv2 > adv1) == torch.round(y)).float().mean()

    return loss, accuracy


class PREF(SAC):
    def __init__(
        self,
        policy: Union[str, Type[HACOPolicy]],
        env: Union[GymEnv, str],
        learning_rate: dict = dict(actor=0.0, critic=0.0, entropy=0.0),
        buffer_size: int = 100,  # Shrink the size to reduce memory consumption when testing
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[HACOReplayBuffer] = HACOReplayBuffer,  # PZH: !! Use HACO Replay Buffer
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = True,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,

        # PZH: Our new introduce hyper-parameters
        cql_coefficient=1,
        monitor_wrapper=False,
        future_steps=5,
        bias=0,
        cbias=0,
        alpha=0.1,
        poso="pos_observations",
        posa="pos_actions",
        nego="neg_observations",
        nega="neg_actions",
        cpl_loss_weight=1.0,
        bc_loss_weight=1.0,
        use_bc_only=False,
        use_bcmse_only=False,
    ):

        assert replay_buffer_class == HACOReplayBuffer
        self.bias = bias
        self.cbias, self.alpha, self.bc_loss_weight, self.use_bc_only = cbias, alpha, bc_loss_weight, use_bc_only
        self.poso, self.posa, self.nego, self.nega = poso, posa, nego, nega
        self.cpl_loss_weight, self.use_bcmse_only = cpl_loss_weight, use_bcmse_only

        super().__init__(
            policy,
            env,
            HACOPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            monitor_wrapper=monitor_wrapper
        )
        # PZH: Define some new variables
        self.cql_coefficient = cql_coefficient
        from pvp.sb3.haco.haco_buffer import PrefReplayBuffer
        self.future_steps = future_steps
        self.prefreplay_buffer = PrefReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                future_steps=self.future_steps,
                **self.replay_buffer_kwargs,
        )

    # def _create_aliases(self) -> None:
    #     super()._create_aliases()
    #     self.cost_critic = self.policy.cost_critic
    #     self.cost_critic_target = self.policy.cost_critic_target
    def _store_transition(
        self,
        replay_buffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        if infos[0]["takeover"] or infos[0]["takeover_start"]:
            replay_buffer = self.human_data_buffer
        super(PREF, self)._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = {
            "actor": self.actor.optimizer,
            "critic": self.critic.optimizer,
            # "cost_critic": self.cost_critic.optimizer
        }
        if self.ent_coef_optimizer is not None:
            optimizers["entropy"] = self.ent_coef_optimizer

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        stat_recorder = defaultdict(list)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            # replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            #
            # # We need to sample because `log_std` may have changed between two gradient steps
            # if self.use_sde:
            #     self.actor.reset_noise()
            #
            # # Action by the current actor for the sampled state
            # actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            # log_prob = log_prob.reshape(-1, 1)

            # ========== Compute the CPL loss ==========
            bc_loss_weight = self.bc_loss_weight 

            # Sample replay buffer
            replay_data_human = None
            replay_data_agent = None
            if self.replay_buffer.pos > batch_size and self.human_data_buffer.pos > batch_size:
                replay_data_agent = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                replay_data_human = self.human_data_buffer.sample(batch_size, env=self._vec_normalize_env)
            elif self.human_data_buffer.pos > batch_size:
                replay_data_human = self.human_data_buffer.sample(batch_size, env=self._vec_normalize_env)
            elif self.replay_buffer.pos > batch_size:
                replay_data_agent = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            if self.prefreplay_buffer.pos > 0:
                replay_data = self.prefreplay_buffer.sample(batch_size, env=self._vec_normalize_env)
            else:
                loss = None
                break

            alpha = self.alpha
            accuracy = cpl_loss = bc_loss = None
            
            mask = replay_data.mask
            
            pos_obs, pos_action = getattr(replay_data, self.poso), getattr(replay_data, self.posa)
            pos_obs_reshape = th.reshape(pos_obs, (-1, pos_obs.shape[-1]))
            pos_action_reshape = th.reshape(pos_action, (-1, pos_action.shape[-1]))
            mean, log_std, _ = self.policy.actor.get_action_dist_params(pos_obs_reshape)
            dist = self.policy.actor.action_dist.proba_distribution(mean, log_std)
            log_prob_pos = dist.log_prob(pos_action_reshape)
            log_prob_pos = th.reshape(log_prob_pos, pos_obs.shape[:-1]) * mask
            if self.use_bcmse_only:
                policy_mean = th.reshape(mean, pos_action.shape)
                actdiff = th.mean((policy_mean - pos_action) ** 2, dim = -1)
                loss = th.mean(actdiff * mask)
            elif self.use_bc_only:
                loss = -log_prob_pos.mean()
            log_prob_pos = log_prob_pos.sum(dim = -1)
            
            neg_obs, neg_action = getattr(replay_data, self.nego), getattr(replay_data, self.nega)
            neg_obs_reshape = th.reshape(neg_obs, (-1, neg_obs.shape[-1]))
            neg_action_reshape = th.reshape(neg_action, (-1, neg_action.shape[-1]))
            mean, log_std, _ = self.policy.actor.get_action_dist_params(neg_obs_reshape)
            dist = self.policy.actor.action_dist.proba_distribution(mean, log_std)
            log_prob_neg = dist.log_prob(neg_action_reshape)
            log_prob_neg = th.reshape(log_prob_neg, neg_obs.shape[:-1]) * mask
            log_prob_neg = log_prob_neg.sum(dim = -1)
            
            adv_pos, adv_neg = alpha * log_prob_pos, alpha * log_prob_neg
            label = torch.ones_like(adv_pos)
            cpl_loss, accuracy = biased_bce_with_logits(adv_neg, adv_pos, label.float(), bias=self.bias, cbias=self.cbias)


            # if replay_data_human is not None:
            #     human_action = replay_data_human.actions_behavior
            #     agent_action = replay_data_human.actions_novice

            #     mean, log_std, _ = self.policy.actor.get_action_dist_params(replay_data_human.observations)
            #     dist = self.policy.actor.action_dist.proba_distribution(mean, log_std)

            #     log_prob_human = dist.log_prob(human_action)  #.sum(dim=-1)  # Don't do the sum...
            #     log_prob_agent = dist.log_prob(agent_action)  #.sum(dim=-1)
            #     adv_human = alpha * log_prob_human
            #     adv_agent = alpha * log_prob_agent
            #     # If label = 1, then adv_human > adv_agent
            #     label = torch.ones_like(adv_human)
            #     cpl_loss, accuracy = biased_bce_with_logits(adv_agent, adv_human, label.float(), bias=0.5)

            if replay_data_human is not None:
                mean, log_std, _ = self.policy.actor.get_action_dist_params(replay_data_human.observations)
                dist = self.policy.actor.action_dist.proba_distribution(mean, log_std)
                log_prob_bc = dist.log_prob(replay_data_human.actions_behavior)
                bc_loss = -log_prob_bc.mean()

            # Aggregate losses
            if bc_loss is None and cpl_loss is None:
                break
            
            if not self.use_bc_only and not self.use_bcmse_only:
                loss = bc_loss_weight * (bc_loss
                                     if bc_loss is not None else 0.0) + self.cpl_loss_weight * (cpl_loss if cpl_loss is not None else 0.0)

            self._optimize_actor(actor_loss=loss)

            # Stats
            stat_recorder["bc_loss"].append(bc_loss.item() if bc_loss is not None else float('nan'))
            stat_recorder["cpl_loss"].append(cpl_loss.item() if cpl_loss is not None else float('nan'))
            stat_recorder["cpl_accuracy"].append(accuracy.item() if accuracy is not None else float('nan'))
            stat_recorder["loss"].append(loss.item() if loss is not None else float('nan'))

            # if self.policy_kwargs["share_features_extractor"] == "critic":
            #     self._optimize_actor(actor_loss=actor_loss)
            #     # self._optimize_critics(merged_critic_loss=merged_critic_loss)
            # elif self.policy_kwargs["share_features_extractor"] == "actor":
            #     raise ValueError()
            # else:
            #     self._optimize_actor(actor_loss=actor_loss)
            #     # self._optimize_critics(merged_critic_loss=merged_critic_loss)

            # Update target networks
            # if gradient_step % self.target_update_interval == 0:
            #     polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            #     polyak_update(self.cost_critic.parameters(), self.cost_critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/num_traj", self.prefreplay_buffer.pos)
        self.logger.record("train/n_updates", self._n_updates)
        for key, values in stat_recorder.items():
            self.logger.record("train/{}".format(key), np.mean(values))

    def _optimize_actor(self, actor_loss):
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

    # def _optimize_critics(self, merged_critic_loss):
    #     self.critic.optimizer.zero_grad()
    #     merged_critic_loss.backward()
    #     self.critic.optimizer.step()

    # def _excluded_save_params(self) -> List[str]:
    #     return super()._excluded_save_params() + ["cost_critic", "cost_critic_target"]
    #
    # def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
    #     # state_dicts = ["policy", "actor.optimizer", "critic.optimizer", "cost_critic.optimizer"]
    #     state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
    #     if self.ent_coef_optimizer is not None:
    #         saved_pytorch_variables = ["log_ent_coef"]
    #         state_dicts.append("ent_coef_optimizer")
    #     else:
    #         saved_pytorch_variables = ["ent_coef_tensor"]
    #     return state_dicts, saved_pytorch_variables

    def _setup_model(self) -> None:
        super()._setup_model()
        self.human_data_buffer = HACOReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
            **self.replay_buffer_kwargs
        )
        # self.human_data_buffer
