import os
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np

from pvp.sb3.common import base_class  # pytype: disable=pyi-error
from pvp.sb3.common.evaluation import evaluate_policy
from pvp.sb3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization


class BaseCallback(ABC):
    """
    Base class for callback.

    :param verbose:
    """
    def __init__(self, verbose: int = 0):
        super(BaseCallback, self).__init__()
        # The RL model
        self.model = None  # type: Optional[base_class.BaseAlgorithm]
        # An alias for self.model.get_env(), the environment used for training
        self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        # n_envs * n times env.step() was called
        self.num_timesteps = 0  # type: int
        self.verbose = verbose
        self.locals: Dict[str, Any] = {}
        self.globals: Dict[str, Any] = {}
        self.logger = None
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        self.parent = None  # type: Optional[BaseCallback]

    # Type hint as string to avoid circular import
    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.model = model
        self.training_env = model.get_env()
        self.logger = model.logger
        self._init_callback()

    def _init_callback(self) -> None:
        pass

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        self._on_training_start()

    def _on_training_start(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def _on_rollout_start(self) -> None:
        pass

    @abstractmethod
    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to ``env.step()``.

        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        # timesteps start at zero
        self.num_timesteps = self.model.num_timesteps

        return self._on_step()

    def on_training_end(self) -> None:
        self._on_training_end()

    def _on_training_end(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:
        pass

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables on sub callbacks.

        :param locals_: the local variables during rollout collection
        """
        pass


class EventCallback(BaseCallback):
    """
    Base class for triggering callback on event.

    :param callback: Callback that will be called
        when an event is triggered.
    :param verbose:
    """
    def __init__(self, callback: Optional[BaseCallback] = None, verbose: int = 0):
        super(EventCallback, self).__init__(verbose=verbose)
        self.callback = callback
        # Give access to the parent
        if callback is not None:
            self.callback.parent = self

    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        super(EventCallback, self).init_callback(model)
        if self.callback is not None:
            self.callback.init_callback(self.model)

    def _on_training_start(self) -> None:
        if self.callback is not None:
            self.callback.on_training_start(self.locals, self.globals)

    def _on_event(self) -> bool:
        if self.callback is not None:
            return self.callback.on_step()
        return True

    def _on_step(self) -> bool:
        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback is not None:
            self.callback.update_locals(locals_)


class CallbackList(BaseCallback):
    """
    Class for chaining callbacks.

    :param callbacks: A list of callbacks that will be called
        sequentially.
    """
    def __init__(self, callbacks: List[BaseCallback]):
        super(CallbackList, self).__init__()
        assert isinstance(callbacks, list)
        self.callbacks = callbacks

    def _init_callback(self) -> None:
        for callback in self.callbacks:
            callback.init_callback(self.model)

    def _on_training_start(self) -> None:
        for callback in self.callbacks:
            callback.on_training_start(self.locals, self.globals)

    def _on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()

    def _on_step(self) -> bool:
        continue_training = True
        for callback in self.callbacks:
            # Return False (stop training) if at least one callback returns False
            continue_training = callback.on_step() and continue_training
        return continue_training

    def _on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        for callback in self.callbacks:
            callback.update_locals(locals_)


class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True


class ConvertCallback(BaseCallback):
    """
    Convert functional callback (old-style) to object.

    :param callback:
    :param verbose:
    """
    def __init__(self, callback: Callable[[Dict[str, Any], Dict[str, Any]], bool], verbose: int = 0):
        super(ConvertCallback, self).__init__(verbose)
        self.callback = callback

    def _on_step(self) -> bool:
        if self.callback is not None:
            return self.callback(self.locals, self.globals)
        return True


class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []
        self.evaluations_info_buffer = defaultdict(list)
        
        # ===== Load expert for agent-expert comparison =====
        self.expert = None
        self._expert_loaded = False

    def _load_expert(self) -> None:
        """Load the PPO expert model for agent-expert comparison during evaluation."""
        if self._expert_loaded:
            return
        
        print("[EvalCallback] Loading expert model for evaluation comparison...", flush=True)
        # Use the globally loaded expert from fakehuman_env (already loaded at module import)
        from pvp.experiments.metadrive.egpo.fakehuman_env import _expert
        self.expert = _expert
        self._expert_loaded = True
        if self.expert is not None:
            print(f"[EvalCallback] Expert model loaded successfully! Type: {type(self.expert)}", flush=True)
        else:
            print("[EvalCallback] WARNING: Expert is None!", flush=True)

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        # Load expert model for agent-expert comparison
        self._load_expert()

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        # Debug: confirm callback is being called
        if not hasattr(self, '_callback_called_count'):
            self._callback_called_count = 0
        self._callback_called_count += 1
        if self._callback_called_count == 1:
            print(f"[DEBUG] _log_success_callback called for the first time!", flush=True)
            print(f"[DEBUG] locals_ keys: {list(locals_.keys())}", flush=True)
        
        info = locals_["info"]
        # Note: evaluate_policy passes 'actions' (plural), not 'action' (singular)
        # For VecEnv, actions is shape (n_envs, action_dim), we need the current env's action
        actions = locals_.get("actions", None)
        i = locals_.get("i", 0)  # Current env index
        
        # Extract action for current environment
        if actions is not None:
            action = actions[i] if len(actions.shape) > 1 else actions
        else:
            # Fallback: try to get from info dict
            action = info.get("action", None)
        
        # Debug: print action info on first call
        if self._callback_called_count == 1:
            print(f"[DEBUG] actions shape: {actions.shape if actions is not None else None}", flush=True)
            print(f"[DEBUG] i (env index): {i}", flush=True)
            print(f"[DEBUG] action extracted: {action}", flush=True)
            print(f"[DEBUG] info keys: {list(info.keys())}", flush=True)
            print(f"[DEBUG] lidar_obs in info: {'lidar_obs' in info}", flush=True)
            if 'lidar_obs' in info:
                lidar = info['lidar_obs']
                print(f"[DEBUG] lidar_obs shape: {lidar.shape if hasattr(lidar, 'shape') else type(lidar)}", flush=True)
        
        # ===== Initialize episode-level tracking =====
        if not hasattr(self, '_episode_crash_flags'):
            self._episode_crash_flags = {
                'crash_vehicle': False,
                'crash_object': False,
                'crash_building': False,
                'crash_sidewalk': False,
                'crash_human': False,
                'out_of_road': False,
            }
        
        # Track crash COUNTS per episode (not just binary flags)
        if not hasattr(self, '_episode_crash_counts'):
            self._episode_crash_counts = {
                'crash_vehicle': 0,
                'crash_object': 0,
                'crash_building': 0,
                'crash_sidewalk': 0,
                'crash_human': 0,
                'out_of_road': 0,
            }
        
        if not hasattr(self, '_episode_actions'):
            self._episode_actions = []  # Track all actions in episode
            self._episode_crash_actions = []  # Actions at crash moments
            self._episode_crash_q_values = []  # Q-values at crash moments
            self._episode_expert_diffs = []  # Agent-expert action differences (L2)
            self._episode_crash_expert_diffs = []  # Agent-expert diffs at crash moments
            self._episode_expert_steering_diffs = []  # Steering diffs
            self._episode_expert_accel_diffs = []  # Acceleration diffs
        
        # ===== Track actions at every step =====
        if action is not None:
            action_np = np.array(action).flatten()
            if len(action_np) >= 2:
                self._episode_actions.append(action_np)
                
                # ===== Expert comparison =====
                # Expert uses lidar observation, not RGB - get it from info dict
                lidar_obs = info.get("lidar_obs", None)
                
                # Debug: print status on first step of first episode
                if not hasattr(self, '_debug_printed'):
                    self._debug_printed = True
                    print("=" * 60, flush=True)
                    print("[DEBUG EvalCallback] First step debug info:", flush=True)
                    print(f"  Expert loaded: {self.expert is not None}", flush=True)
                    print(f"  Expert type: {type(self.expert)}", flush=True)
                    print(f"  lidar_obs in info: {lidar_obs is not None}", flush=True)
                    if lidar_obs is not None:
                        print(f"  lidar_obs shape: {lidar_obs.shape if hasattr(lidar_obs, 'shape') else type(lidar_obs)}", flush=True)
                    print(f"  info keys: {list(info.keys())}", flush=True)
                    print("=" * 60, flush=True)
                
                # Use prev_obs (observation BEFORE step) for correct comparison
                # prev_obs is the observation that was used to generate the action
                prev_obs = locals_.get("prev_obs", None)
                
                # Fall back to lidar_obs if prev_obs not available (for backwards compatibility)
                expert_input_obs = prev_obs if prev_obs is not None else lidar_obs
                
                if self.expert is not None and expert_input_obs is not None:
                    # Get expert action using the same observation that agent used
                    expert_action, _ = self.expert.predict(expert_input_obs, deterministic=True)
                    expert_action_np = np.array(expert_action).flatten()
                    
                    # Action difference (L2 norm)
                    action_diff = np.linalg.norm(action_np - expert_action_np)
                    self._episode_expert_diffs.append(action_diff)
                    
                    # Per-dimension differences
                    steering_diff = abs(action_np[0] - expert_action_np[0])
                    accel_diff = abs(action_np[1] - expert_action_np[1])
                    self._episode_expert_steering_diffs.append(steering_diff)
                    self._episode_expert_accel_diffs.append(accel_diff)
                
                # If crash happens this step, record crash-specific data
                is_crashing = (info.get("crash_vehicle", False) or 
                              info.get("crash_object", False) or
                              info.get("crash_building", False) or
                              info.get("crash_sidewalk", False))
                if is_crashing:
                    self._episode_crash_actions.append(action_np)
                    
                    # Record agent-expert diff at crash moment
                    if len(self._episode_expert_diffs) > 0:
                        self._episode_crash_expert_diffs.append(self._episode_expert_diffs[-1])
                    
                    # Try to get Q-value from model if available (not for pure BC)
                    obs = locals_.get("obs", None)
                    if hasattr(self.model, 'critic') and obs is not None:
                        import torch as th
                        with th.no_grad():
                            obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
                            action_tensor = th.tensor(action_np).float().unsqueeze(0).to(self.model.device)
                            q_value = self.model.critic.q1_forward(obs_tensor, action_tensor)
                            self._episode_crash_q_values.append(q_value.item())
        
        # Accumulate crash events during this step
        # Flags: OR logic - once True, stays True (for "did it happen at all")
        # Counts: increment each time crash happens (for "how many times")
        if info.get("crash_vehicle", False):
            self._episode_crash_flags['crash_vehicle'] = True
            self._episode_crash_counts['crash_vehicle'] += 1
        if info.get("crash_object", False):
            self._episode_crash_flags['crash_object'] = True
            self._episode_crash_counts['crash_object'] += 1
        if info.get("crash_building", False):
            self._episode_crash_flags['crash_building'] = True
            self._episode_crash_counts['crash_building'] += 1
        if info.get("crash_sidewalk", False):
            self._episode_crash_flags['crash_sidewalk'] = True
            self._episode_crash_counts['crash_sidewalk'] += 1
        if info.get("crash_human", False):
            self._episode_crash_flags['crash_human'] = True
            self._episode_crash_counts['crash_human'] += 1
        if info.get("out_of_road", False):
            self._episode_crash_flags['out_of_road'] = True
            self._episode_crash_counts['out_of_road'] += 1

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

            maybe_is_success2 = info.get("arrive_dest", None)
            if maybe_is_success2 is not None:
                self._is_success_buffer.append(maybe_is_success2)

            assert (maybe_is_success is None) or (maybe_is_success2 is None), "We cannot have two success flags!"

            # Log standard metrics (these are episode-level metrics from MetaDrive)
            for k in ["episode_energy", "route_completion", "total_cost", "arrive_dest", "max_step", "cost"]:
                if k in info:
                    self.evaluations_info_buffer[k].append(info[k])
            
            # ===== Use accumulated crash flags (tracked across entire episode) =====
            arrive_dest = info.get("arrive_dest", False)
            crash_vehicle = self._episode_crash_flags['crash_vehicle']
            crash_object = self._episode_crash_flags['crash_object']
            crash_building = self._episode_crash_flags['crash_building']
            crash_sidewalk = self._episode_crash_flags['crash_sidewalk']
            crash_human = self._episode_crash_flags['crash_human']
            out_of_road = self._episode_crash_flags['out_of_road']
            
            # Compute crash-aware success rates using accumulated flags
            success_no_crash_vehicle = arrive_dest and not crash_vehicle
            success_no_crash_any = arrive_dest and not crash_vehicle and not crash_object
            
            self.evaluations_info_buffer["success_no_crash_vehicle"].append(float(success_no_crash_vehicle))
            self.evaluations_info_buffer["success_no_crash_any"].append(float(success_no_crash_any))
            
            # Track crash rates separately for analysis
            self.evaluations_info_buffer["crash_vehicle_rate"].append(float(crash_vehicle))
            self.evaluations_info_buffer["crash_object_rate"].append(float(crash_object))
            # crash_vehicle_or_object: only movable obstacles (vehicle + traffic cone/object)
            self.evaluations_info_buffer["crash_vehicle_or_object_rate"].append(float(crash_vehicle or crash_object))
            
            # Track additional failure types
            self.evaluations_info_buffer["crash_building_rate"].append(float(crash_building))
            self.evaluations_info_buffer["crash_sidewalk_rate"].append(float(crash_sidewalk))
            self.evaluations_info_buffer["crash_human_rate"].append(float(crash_human))
            self.evaluations_info_buffer["out_of_road_rate"].append(float(out_of_road))
            
            # Track any bad event (comprehensive failure rate)
            # crash_total: ANY type of crash (vehicle + object + building + sidewalk + human)
            crash_total = crash_vehicle or crash_object or crash_building or crash_sidewalk or crash_human
            any_bad_event = crash_total or out_of_road
            self.evaluations_info_buffer["crash_total_rate"].append(float(crash_total))
            self.evaluations_info_buffer["any_bad_event_rate"].append(float(any_bad_event))
            
            # Success without any bad event
            success_no_bad_event = arrive_dest and not any_bad_event
            self.evaluations_info_buffer["success_no_bad_event"].append(float(success_no_bad_event))
            
            # ===== Track crash COUNTS per episode (how many times, not just if happened) =====
            if hasattr(self, '_episode_crash_counts'):
                # Total crashes in this episode (sum of all types)
                total_crash_count = sum(self._episode_crash_counts.values())
                self.evaluations_info_buffer["crash_count_total"].append(total_crash_count)
                
                # Individual crash type counts
                self.evaluations_info_buffer["crash_count_vehicle"].append(self._episode_crash_counts['crash_vehicle'])
                self.evaluations_info_buffer["crash_count_object"].append(self._episode_crash_counts['crash_object'])
                self.evaluations_info_buffer["crash_count_building"].append(self._episode_crash_counts['crash_building'])
                self.evaluations_info_buffer["crash_count_sidewalk"].append(self._episode_crash_counts['crash_sidewalk'])
            
            # ===== Compute and store episode action statistics =====
            if hasattr(self, '_episode_actions') and len(self._episode_actions) > 0:
                episode_actions = np.array(self._episode_actions)
                
                # Mean and abs mean of steering (action dim 0)
                mean_steering = np.mean(episode_actions[:, 0])
                mean_steering_abs = np.mean(np.abs(episode_actions[:, 0]))
                self.evaluations_info_buffer["mean_steering"].append(mean_steering)
                self.evaluations_info_buffer["mean_steering_abs"].append(mean_steering_abs)
                
                # Mean and abs mean of acceleration (action dim 1)
                mean_accel = np.mean(episode_actions[:, 1])
                mean_accel_abs = np.mean(np.abs(episode_actions[:, 1]))
                self.evaluations_info_buffer["mean_accel"].append(mean_accel)
                self.evaluations_info_buffer["mean_accel_abs"].append(mean_accel_abs)
                
                # Steering variance (indicates how erratic the driving is)
                steering_variance = np.var(episode_actions[:, 0])
                self.evaluations_info_buffer["steering_variance"].append(steering_variance)
                
                # Count of hard braking (accel < -0.5)
                hard_brake_ratio = np.mean(episode_actions[:, 1] < -0.5)
                self.evaluations_info_buffer["hard_brake_ratio"].append(hard_brake_ratio)
                
                # Count of hard steering (|steering| > 0.5)
                hard_steer_ratio = np.mean(np.abs(episode_actions[:, 0]) > 0.5)
                self.evaluations_info_buffer["hard_steer_ratio"].append(hard_steer_ratio)
            
            # ===== Crash-specific statistics =====
            if hasattr(self, '_episode_crash_actions') and len(self._episode_crash_actions) > 0:
                crash_actions = np.array(self._episode_crash_actions)
                
                # Mean steering at crash moments
                crash_mean_steering = np.mean(crash_actions[:, 0])
                crash_mean_accel = np.mean(crash_actions[:, 1])
                self.evaluations_info_buffer["crash_mean_steering"].append(crash_mean_steering)
                self.evaluations_info_buffer["crash_mean_accel"].append(crash_mean_accel)
                
                # Number of crash steps in this episode
                crash_steps = len(crash_actions)
                self.evaluations_info_buffer["crash_steps"].append(crash_steps)
            
            # ===== Q-value at crash moments (only for algorithms with critic) =====
            if hasattr(self, '_episode_crash_q_values') and len(self._episode_crash_q_values) > 0:
                mean_crash_q = np.mean(self._episode_crash_q_values)
                self.evaluations_info_buffer["crash_q_value"].append(mean_crash_q)
            
            # ===== Agent-Expert comparison statistics =====
            if hasattr(self, '_episode_expert_diffs') and len(self._episode_expert_diffs) > 0:
                # Overall agent-expert action difference
                mean_expert_diff = np.mean(self._episode_expert_diffs)
                self.evaluations_info_buffer["expert_action_diff_l2"].append(mean_expert_diff)
                
                # Per-dimension differences
                if len(self._episode_expert_steering_diffs) > 0:
                    mean_steering_diff = np.mean(self._episode_expert_steering_diffs)
                    mean_accel_diff = np.mean(self._episode_expert_accel_diffs)
                    self.evaluations_info_buffer["expert_steering_diff"].append(mean_steering_diff)
                    self.evaluations_info_buffer["expert_accel_diff"].append(mean_accel_diff)
                
                # Agent-expert agreement ratio (action diff < 0.3 is considered "agree")
                agreement_ratio = np.mean(np.array(self._episode_expert_diffs) < 0.3)
                self.evaluations_info_buffer["expert_agreement_ratio"].append(agreement_ratio)
            
            # ===== Agent-Expert diff at crash moments =====
            if hasattr(self, '_episode_crash_expert_diffs') and len(self._episode_crash_expert_diffs) > 0:
                crash_expert_diff = np.mean(self._episode_crash_expert_diffs)
                self.evaluations_info_buffer["crash_expert_diff"].append(crash_expert_diff)
                
                # Check if agent was following expert when crashing
                # High crash_expert_diff means agent was NOT following expert when crashing
                crash_follow_expert = np.mean(np.array(self._episode_crash_expert_diffs) < 0.3)
                self.evaluations_info_buffer["crash_follow_expert_ratio"].append(crash_follow_expert)
            
            # Reset episode tracking for next episode
            self._episode_crash_flags = {
                'crash_vehicle': False,
                'crash_object': False,
                'crash_building': False,
                'crash_sidewalk': False,
                'crash_human': False,
                'out_of_road': False,
            }
            self._episode_crash_counts = {
                'crash_vehicle': 0,
                'crash_object': 0,
                'crash_building': 0,
                'crash_sidewalk': 0,
                'crash_human': 0,
                'out_of_road': 0,
            }
            self._episode_actions = []
            self._episode_crash_actions = []
            self._episode_crash_q_values = []
            self._episode_expert_diffs = []
            self._episode_crash_expert_diffs = []
            self._episode_expert_steering_diffs = []
            self._episode_expert_accel_diffs = []

        if "raw_action" in info:
            self.evaluations_info_buffer["raw_action"].append(info["raw_action"])

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []
            self.evaluations_info_buffer.clear()
            
            # Reset debug flags to print debug info for each evaluation
            if hasattr(self, '_debug_printed'):
                delattr(self, '_debug_printed')
            if hasattr(self, '_callback_called_count'):
                delattr(self, '_callback_called_count')
            
            # Ensure expert is loaded (in case _init_callback wasn't called)
            if not self._expert_loaded:
                print("[DEBUG] Expert not loaded yet, loading now...", flush=True)
                self._load_expert()
            
            print(f"[DEBUG] Before evaluation - expert loaded: {self.expert is not None}", flush=True)
            print("Start evaluating policy for {} episodes!".format(self.n_eval_episodes), flush=True)

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            print("Finish evaluating policy for {} episodes!".format(self.n_eval_episodes))

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate (arrive_dest): {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Log crash-aware success rates with clear names
            if "success_no_crash_vehicle" in self.evaluations_info_buffer:
                success_no_crash_vehicle = np.mean(self.evaluations_info_buffer["success_no_crash_vehicle"])
                success_no_crash_any = np.mean(self.evaluations_info_buffer["success_no_crash_any"])
                crash_vehicle_rate = np.mean(self.evaluations_info_buffer["crash_vehicle_rate"])
                crash_object_rate = np.mean(self.evaluations_info_buffer["crash_object_rate"])
                crash_vehicle_or_object_rate = np.mean(self.evaluations_info_buffer["crash_vehicle_or_object_rate"])
                
                if self.verbose > 0:
                    print(f"Success rate (no crash_vehicle): {100 * success_no_crash_vehicle:.2f}%")
                    print(f"Success rate (no crash_any): {100 * success_no_crash_any:.2f}%")
                    print(f"Crash vehicle rate: {100 * crash_vehicle_rate:.2f}%")
                    print(f"Crash object rate: {100 * crash_object_rate:.2f}%")
                    print(f"Crash vehicle/object rate: {100 * crash_vehicle_or_object_rate:.2f}%")
                
                self.logger.record("eval/success_no_crash_vehicle", success_no_crash_vehicle)
                self.logger.record("eval/success_no_crash_any", success_no_crash_any)
                self.logger.record("eval/crash_vehicle_rate", crash_vehicle_rate)
                self.logger.record("eval/crash_object_rate", crash_object_rate)
                self.logger.record("eval/crash_vehicle_or_object_rate", crash_vehicle_or_object_rate)
                
                # Log additional failure rates
                if "crash_building_rate" in self.evaluations_info_buffer:
                    crash_building_rate = np.mean(self.evaluations_info_buffer["crash_building_rate"])
                    crash_sidewalk_rate = np.mean(self.evaluations_info_buffer["crash_sidewalk_rate"])
                    crash_human_rate = np.mean(self.evaluations_info_buffer["crash_human_rate"])
                    out_of_road_rate = np.mean(self.evaluations_info_buffer["out_of_road_rate"])
                    crash_total_rate = np.mean(self.evaluations_info_buffer["crash_total_rate"])
                    any_bad_event_rate = np.mean(self.evaluations_info_buffer["any_bad_event_rate"])
                    success_no_bad_event = np.mean(self.evaluations_info_buffer["success_no_bad_event"])
                    
                    if self.verbose > 0:
                        print(f"Crash building rate: {100 * crash_building_rate:.2f}%")
                        print(f"Crash sidewalk rate: {100 * crash_sidewalk_rate:.2f}%")
                        print(f"Out of road rate: {100 * out_of_road_rate:.2f}%")
                        print(f"Crash total rate: {100 * crash_total_rate:.2f}%")
                        print(f"Any bad event rate: {100 * any_bad_event_rate:.2f}%")
                        print(f"Success (no bad events): {100 * success_no_bad_event:.2f}%")
                    
                    self.logger.record("eval/crash_building_rate", crash_building_rate)
                    self.logger.record("eval/crash_sidewalk_rate", crash_sidewalk_rate)
                    self.logger.record("eval/crash_human_rate", crash_human_rate)
                    self.logger.record("eval/out_of_road_rate", out_of_road_rate)
                    self.logger.record("eval/crash_total_rate", crash_total_rate)
                    self.logger.record("eval/any_bad_event_rate", any_bad_event_rate)
                
                # Log average crash COUNTS per episode
                if "crash_count_total" in self.evaluations_info_buffer:
                    avg_crash_count = np.mean(self.evaluations_info_buffer["crash_count_total"])
                    avg_crash_count_vehicle = np.mean(self.evaluations_info_buffer["crash_count_vehicle"])
                    avg_crash_count_object = np.mean(self.evaluations_info_buffer["crash_count_object"])
                    avg_crash_count_building = np.mean(self.evaluations_info_buffer["crash_count_building"])
                    avg_crash_count_sidewalk = np.mean(self.evaluations_info_buffer["crash_count_sidewalk"])
                    
                    if self.verbose > 0:
                        print(f"Avg crashes per episode: {avg_crash_count:.2f} (vehicle: {avg_crash_count_vehicle:.2f}, object: {avg_crash_count_object:.2f}, building: {avg_crash_count_building:.2f}, sidewalk: {avg_crash_count_sidewalk:.2f})")
                    
                    self.logger.record("eval/crash_count_total", avg_crash_count)
                    self.logger.record("eval/crash_count_vehicle", avg_crash_count_vehicle)
                    self.logger.record("eval/crash_count_object", avg_crash_count_object)
                    self.logger.record("eval/crash_count_building", avg_crash_count_building)
                    self.logger.record("eval/crash_count_sidewalk", avg_crash_count_sidewalk)
                    self.logger.record("eval/success_no_bad_event", success_no_bad_event)

            # ===== Log action statistics =====
            if "mean_steering" in self.evaluations_info_buffer and len(self.evaluations_info_buffer["mean_steering"]) > 0:
                mean_steering = np.mean(self.evaluations_info_buffer["mean_steering"])
                mean_steering_abs = np.mean(self.evaluations_info_buffer["mean_steering_abs"])
                mean_accel = np.mean(self.evaluations_info_buffer["mean_accel"])
                mean_accel_abs = np.mean(self.evaluations_info_buffer["mean_accel_abs"])
                steering_variance = np.mean(self.evaluations_info_buffer["steering_variance"])
                hard_brake_ratio = np.mean(self.evaluations_info_buffer["hard_brake_ratio"])
                hard_steer_ratio = np.mean(self.evaluations_info_buffer["hard_steer_ratio"])
                
                if self.verbose > 0:
                    print(f"Mean steering: {mean_steering:.4f}, |steering|: {mean_steering_abs:.4f}")
                    print(f"Mean accel: {mean_accel:.4f}, |accel|: {mean_accel_abs:.4f}")
                    print(f"Hard brake ratio: {100 * hard_brake_ratio:.2f}%, Hard steer ratio: {100 * hard_steer_ratio:.2f}%")
                
                self.logger.record("eval/mean_steering", mean_steering)
                self.logger.record("eval/mean_steering_abs", mean_steering_abs)
                self.logger.record("eval/mean_accel", mean_accel)
                self.logger.record("eval/mean_accel_abs", mean_accel_abs)
                self.logger.record("eval/steering_variance", steering_variance)
                self.logger.record("eval/hard_brake_ratio", hard_brake_ratio)
                self.logger.record("eval/hard_steer_ratio", hard_steer_ratio)
            
            # ===== Log crash-specific statistics =====
            if "crash_mean_steering" in self.evaluations_info_buffer and len(self.evaluations_info_buffer["crash_mean_steering"]) > 0:
                crash_mean_steering = np.mean(self.evaluations_info_buffer["crash_mean_steering"])
                crash_mean_accel = np.mean(self.evaluations_info_buffer["crash_mean_accel"])
                crash_steps = np.mean(self.evaluations_info_buffer["crash_steps"])
                
                if self.verbose > 0:
                    print(f"Crash moments - steering: {crash_mean_steering:.4f}, accel: {crash_mean_accel:.4f}")
                    print(f"Avg crash steps per episode: {crash_steps:.2f}")
                
                self.logger.record("eval/crash_mean_steering", crash_mean_steering)
                self.logger.record("eval/crash_mean_accel", crash_mean_accel)
                self.logger.record("eval/crash_steps", crash_steps)
            
            # ===== Log Q-value at crash moments (only for algorithms with critic) =====
            if "crash_q_value" in self.evaluations_info_buffer and len(self.evaluations_info_buffer["crash_q_value"]) > 0:
                crash_q_value = np.mean(self.evaluations_info_buffer["crash_q_value"])
                if self.verbose > 0:
                    print(f"Q-value at crash moments: {crash_q_value:.4f}")
                self.logger.record("eval/crash_q_value", crash_q_value)
            
            # ===== Log agent-expert comparison statistics =====
            if "expert_action_diff_l2" in self.evaluations_info_buffer and len(self.evaluations_info_buffer["expert_action_diff_l2"]) > 0:
                expert_diff = np.mean(self.evaluations_info_buffer["expert_action_diff_l2"])
                expert_steering_diff = np.mean(self.evaluations_info_buffer["expert_steering_diff"])
                expert_accel_diff = np.mean(self.evaluations_info_buffer["expert_accel_diff"])
                expert_agreement = np.mean(self.evaluations_info_buffer["expert_agreement_ratio"])
                
                if self.verbose > 0:
                    print(f"Agent-Expert action diff (L2): {expert_diff:.4f}")
                    print(f"Agent-Expert steering diff: {expert_steering_diff:.4f}, accel diff: {expert_accel_diff:.4f}")
                    print(f"Agent-Expert agreement ratio: {100 * expert_agreement:.2f}%")
                
                self.logger.record("eval/expert_action_diff", expert_diff)
                self.logger.record("eval/expert_steering_diff", expert_steering_diff)
                self.logger.record("eval/expert_accel_diff", expert_accel_diff)
                self.logger.record("eval/expert_agreement", expert_agreement)
            
            # ===== Log agent-expert comparison at crash moments =====
            if "crash_expert_diff" in self.evaluations_info_buffer and len(self.evaluations_info_buffer["crash_expert_diff"]) > 0:
                crash_expert_diff = np.mean(self.evaluations_info_buffer["crash_expert_diff"])
                crash_follow_expert = np.mean(self.evaluations_info_buffer["crash_follow_expert_ratio"])
                
                if self.verbose > 0:
                    print(f"Agent-Expert diff at crash: {crash_expert_diff:.4f}")
                    print(f"Following expert when crash: {100 * crash_follow_expert:.2f}%")
                
                self.logger.record("eval/crash_expert_diff", crash_expert_diff)
                self.logger.record("eval/crash_follow_expert", crash_follow_expert)
            
            # Log other metrics (skip the ones we already logged above)
            skip_keys = {"success_no_crash_vehicle", "success_no_crash_any", 
                        "crash_vehicle_rate", "crash_object_rate", "crash_vehicle_or_object_rate",
                        "crash_building_rate", "crash_sidewalk_rate", "crash_human_rate",
                        "out_of_road_rate", "crash_total_rate", "any_bad_event_rate", "success_no_bad_event",
                        "crash_count_total", "crash_count_vehicle", "crash_count_object", 
                        "crash_count_building", "crash_count_sidewalk",
                        "mean_steering", "mean_steering_abs", "mean_accel", "mean_accel_abs",
                        "steering_variance", "hard_brake_ratio", "hard_steer_ratio",
                        "crash_mean_steering", "crash_mean_accel", "crash_steps", "crash_q_value",
                        "expert_action_diff_l2", "expert_steering_diff", "expert_accel_diff",
                        "expert_agreement_ratio", "crash_expert_diff", "crash_follow_expert_ratio"}
            for k, v in self.evaluations_info_buffer.items():
                if k not in skip_keys and len(v) > 0:
                    self.logger.record("eval/{}".format(k), np.mean(np.asarray(v)))

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps)
            self.logger.dump(self.num_timesteps)
            
            # Explicitly sync to wandb (ensure all eval metrics are logged)
            import wandb
            if wandb.run is not None:
                wandb.log(self.logger.name_to_value, step=self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


class StopTrainingOnRewardThreshold(BaseCallback):
    """
    Stop the training once a threshold in episodic reward
    has been reached (i.e. when the model is good enough).

    It must be used with the ``EvalCallback``.

    :param reward_threshold:  Minimum expected reward per episode
        to stop training.
    :param verbose:
    """
    def __init__(self, reward_threshold: float, verbose: int = 0):
        super(StopTrainingOnRewardThreshold, self).__init__(verbose=verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnMinimumReward`` callback must be used " "with an ``EvalCallback``"
        # Convert np.bool_ to bool, otherwise callback() is False won't work
        continue_training = bool(self.parent.best_mean_reward < self.reward_threshold)
        if self.verbose > 0 and not continue_training:
            print(
                f"Stopping training because the mean reward {self.parent.best_mean_reward:.2f} "
                f" is above the threshold {self.reward_threshold}"
            )
        return continue_training


class EveryNTimesteps(EventCallback):
    """
    Trigger a callback every ``n_steps`` timesteps

    :param n_steps: Number of timesteps between two trigger.
    :param callback: Callback that will be called
        when the event is triggered.
    """
    def __init__(self, n_steps: int, callback: BaseCallback):
        super(EveryNTimesteps, self).__init__(callback)
        self.n_steps = n_steps
        self.last_time_trigger = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps
            return self._on_event()
        return True


class StopTrainingOnMaxEpisodes(BaseCallback):
    """
    Stop the training once a maximum number of episodes are played.

    For multiple environments presumes that, the desired behavior is that the agent trains on each env for ``max_episodes``
    and in total for ``max_episodes * n_envs`` episodes.

    :param max_episodes: Maximum number of episodes to stop training.
    :param verbose: Select whether to print information about when training ended by reaching ``max_episodes``
    """
    def __init__(self, max_episodes: int, verbose: int = 0):
        super(StopTrainingOnMaxEpisodes, self).__init__(verbose=verbose)
        self.max_episodes = max_episodes
        self._total_max_episodes = max_episodes
        self.n_episodes = 0

    def _init_callback(self) -> None:
        # At start set total max according to number of envirnments
        self._total_max_episodes = self.max_episodes * self.training_env.num_envs

    def _on_step(self) -> bool:
        # Check that the `dones` local variable is defined
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        self.n_episodes += np.sum(self.locals["dones"]).item()

        continue_training = self.n_episodes < self._total_max_episodes

        if self.verbose > 0 and not continue_training:
            mean_episodes_per_env = self.n_episodes / self.training_env.num_envs
            mean_ep_str = (
                f"with an average of {mean_episodes_per_env:.2f} episodes per env"
                if self.training_env.num_envs > 1 else ""
            )

            print(
                f"Stopping training with a total of {self.num_timesteps} steps because the "
                f"{self.locals.get('tb_log_name')} model reached max_episodes={self.max_episodes}, "
                f"by playing for {self.n_episodes} episodes "
                f"{mean_ep_str}"
            )
        return continue_training
