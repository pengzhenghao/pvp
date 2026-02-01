"""
Parallel evaluation of models on hard scenarios (headless mode).
Uses SubprocVecEnv for parallel evaluation - much faster than sequential.

Usage:
    python eval_hard_scenarios_parallel.py --model iql --num_envs 10
    python eval_hard_scenarios_parallel.py --model both --num_seeds 20  # For testing
"""

import os
# CRITICAL: Set GPU environment BEFORE any other imports
# This ensures SubprocVecEnv child processes inherit the correct GPU setting
_gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_id
os.environ["SDL_VIDEODRIVER"] = "offscreen"
os.environ["PYOPENGL_PLATFORM"] = "egl"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]
print(f"[GPU Setup] CUDA_VISIBLE_DEVICES={_gpu_id}")

import argparse
import json
import numpy as np
import sys
import time
import gymnasium
import gymnasium.spaces
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
gymnasium.spaces.space = gymnasium.spaces

from pathlib import Path
import collections

# Suppress metadrive logging
import logging
logging.getLogger('metadrive').setLevel(logging.WARNING)
logging.getLogger('panda3d').setLevel(logging.WARNING)


class SeedPoolEnv:
    """
    Wrapper that manages a queue of seeds. When env is done, 
    auto-reset pulls the next seed from the queue.
    This enables continuous evaluation without waiting for slow envs.
    """
    def __init__(self, env, seeds):
        self.env = env
        self._original_seeds = list(seeds)  # Keep original for reset_pool
        self.seed_queue = collections.deque(seeds)
        self.current_seed = None
        self.finished = False  # True when no more seeds in queue
        self._last_info = {}  # Store info for retrieval
    
    def reset_pool(self):
        """Reset the seed pool to original state for reuse with new checkpoint."""
        self.seed_queue = collections.deque(self._original_seeds)
        self.current_seed = None
        self.finished = False
        self._last_info = {}
        return self.reset()
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def unwrapped(self):
        return self.env.unwrapped
    
    def __getattr__(self, name):
        """Proxy any unknown attributes to the wrapped env."""
        return getattr(self.env, name)
        
    def reset(self, **kwargs):
        if not self.seed_queue:
            self.finished = True
            # Return dummy observation - matching the Dict observation space
            if hasattr(self.observation_space, 'sample'):
                dummy_obs = self.observation_space.sample()
            else:
                dummy_obs = {}
            self._last_info = {'finished': True, 'seed': None}
            # Return ONLY obs - SubprocVecEnv expects this
            return dummy_obs
        
        self.current_seed = self.seed_queue.popleft()
        self.finished = False
        result = self.env.reset(seed=self.current_seed)
        
        # Handle both (obs, info) tuple and just obs return
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            if not isinstance(info, dict):
                info = {}
        else:
            obs = result
            info = {}
        
        info['seed'] = self.current_seed
        info['finished'] = False
        info['remaining_seeds'] = len(self.seed_queue)
        self._last_info = info
        # Return ONLY obs - SubprocVecEnv expects this
        return obs
    
    def step(self, action):
        if self.finished:
            # Return dummy done state (4 values for SB3 VecEnv compatibility)
            if hasattr(self.observation_space, 'sample'):
                dummy_obs = self.observation_space.sample()
            else:
                dummy_obs = {}
            self._last_info = {'finished': True, 'seed': None}
            return dummy_obs, 0, True, self._last_info
        
        result = self.env.step(action)
        # Handle different return formats
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        elif len(result) == 4:
            obs, reward, done, info = result
        else:
            raise ValueError(f"Unexpected step return: {len(result)} values")
        
        if not isinstance(info, dict):
            info = {}
        info['seed'] = self.current_seed
        info['finished'] = False
        self._last_info = info
        # Return 4 values for SB3 VecEnv compatibility
        return obs, reward, done, info
    
    @property
    def seeds_remaining(self):
        return len(self.seed_queue)
    
    def add_seeds(self, new_seeds):
        """Add more seeds to the queue (for reusing env across models)."""
        self.seed_queue.extend(new_seeds)
        self.finished = False
    
    def close(self):
        return self.env.close()
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


# Top 200 hardest scenarios from combined difficulty ranking
TOP_200_SEEDS = [
    1832, 1683, 1786, 1081, 1175, 1111, 1946, 1839, 1821, 1802,
    1213, 1466, 1604, 1911, 1612, 1650, 1831, 1497, 1364, 1886,
    1543, 1970, 1787, 1389, 1118, 1516, 1847, 1425, 1151, 1208,
    1735, 1637, 1228, 1229, 1789, 1405, 1179, 1943, 1508, 1509,
    1790, 1137, 1538, 1837, 1764, 1935, 1335, 1424, 1026, 1272,
    1901, 1898, 1817, 1751, 1499, 1147, 1241, 1338, 1200, 1865,
    1605, 1252, 1855, 1824, 1300, 1994, 1926, 1000, 1448, 1636,
    1432, 1452, 1197, 1377, 1627, 1904, 1571, 1397, 1491, 1454,
    1583, 1380, 1535, 1004, 1705, 1283, 1840, 1225, 1292, 1697,
    1143, 1261, 1579, 1060, 1988, 1967, 1842, 1374, 1413, 1415,
    1644, 1869, 1609, 1584, 1871, 1671, 1457, 1834, 1629, 1056,
    1103, 1973, 1954, 1124, 1409, 1812, 1546, 1934, 1673, 1002,
    1479, 1418, 1198, 1203, 1725, 1384, 1905, 1864, 1747, 1715,
    1072, 1784, 1564, 1979, 1135, 1142, 1792, 1617, 1841, 1486,
    1160, 1367, 1755, 1868, 1136, 1342, 1500, 1307, 1116, 1520,
    1953, 1015, 1279, 1665, 1101, 1270, 1366, 1129, 1046, 1851,
    1426, 1148, 1510, 1848, 1572, 1980, 1916, 1305, 1575, 1237,
    1667, 1280, 1201, 1775, 1010, 1827, 1903, 1719, 1243, 1590,
    1569, 1047, 1897, 1467, 1504, 1399, 1303, 1793, 1258, 1189,
    1324, 1306, 1248, 1341, 1720, 1568, 1394, 1244, 1711, 1222,
]


class SeedQueueWrapper:
    """
    Wrapper that automatically resets to the next seed in a queue.
    When done=True, the SubprocVecEnv calls env.reset(), and this wrapper
    intercepts it to use the next seed from the queue.
    """
    def __init__(self, env, seeds):
        self.env = env
        self.seed_queue = list(seeds)
        self.current_seed_idx = 0
        self.episode_data = {}  # Store episode results
        
        # Forward all attributes to wrapped env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    def reset(self, **kwargs):
        # Use next seed from queue
        if self.current_seed_idx < len(self.seed_queue):
            seed = self.seed_queue[self.current_seed_idx]
            self.current_seed_idx += 1
            obs = self.env.reset(seed=seed, **kwargs)
            # Store current seed for tracking
            self.current_seed = seed
            # VERIFY: Check actual env seed after reset
            actual_seed = getattr(self.env, 'current_seed', None)
            if actual_seed is None and hasattr(self.env, 'engine'):
                actual_seed = getattr(self.env.engine, 'current_seed', None)
            if actual_seed is not None and actual_seed != seed:
                print(f"[WARNING] SeedQueueWrapper: requested seed={seed}, but env has seed={actual_seed}")
            return obs
        else:
            # All seeds done - just reset normally
            return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        return self.env.close()
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    
    def is_done(self):
        """Check if all seeds have been evaluated"""
        return self.current_seed_idx >= len(self.seed_queue)
    
    def get_current_seed(self):
        """Get the seed of the current episode"""
        return getattr(self, 'current_seed', None)
    
    def __getattr__(self, name):
        # Forward any other attribute access to wrapped env
        return getattr(self.env, name)


def make_shared_env_config(use_image=True, start_seed=1000, num_scenarios=1000, daytime="06:10"):
    """Create config for a shared environment that supports multiple seeds via reset(seed=...)."""
    from metadrive.component.sensors.rgb_camera import RGBCamera
    sensor_size = (84, 84)
    
    config = dict(
        use_render=False,  # No on-screen rendering
        manual_control=False,
        start_seed=start_seed,
        num_scenarios=num_scenarios,  # Support multiple seeds
        horizon=1500,
        crash_vehicle_done=False,
        crash_object_done=False,
        cost_to_reward=False,
        crash_vehicle_penalty=5.0,
        crash_object_penalty=5.0,
        out_of_road_penalty=5.0,
    )
    
    if use_image:
        config.update(dict(
            image_observation=True,
            vehicle_config=dict(image_source="rgb_camera"),
            sensors={"rgb_camera": (RGBCamera, *sensor_size)},
            stack_size=3,
            interface_panel=["rgb_camera", "dashboard"],
            daytime=daytime,
        ))
    else:
        config['image_observation'] = False
    
    return config


def load_model(model_type, checkpoint_path, temp_env):
    """Load IQL or TD3 model from checkpoint."""
    from pvp.sb3.td3.policies import TD3Policy
    from pvp.sb3.sac.our_features_extractor import OurFeaturesExtractorCNN as OurFeaturesExtractor
    from pvp.sb3.common.save_util import load_from_zip_file
    
    policy_kwargs = dict(
        features_extractor_class=OurFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=275),
        share_features_extractor=False,
        net_arch=[256,]
    )
    
    if model_type == "iql":
        from pvp.sb3.td3.iql import IQL
        model = IQL(
            policy=TD3Policy,
            env=temp_env,
            learning_rate=1e-4,
            policy_kwargs=policy_kwargs,
            buffer_size=1000,
            verbose=0,
            device="auto",
        )
    else:  # td3 or td3bc
        from pvp.sb3.td3.td3 import TD3
        model = TD3(
            policy=TD3Policy,
            env=temp_env,
            learning_rate=1e-4,
            policy_kwargs=policy_kwargs,
            buffer_size=1000,
            verbose=0,
            device="auto",
        )
    
    data, params, pytorch_variables = load_from_zip_file(
        checkpoint_path, device=model.device, print_system_info=False
    )
    model.set_parameters(params, exact_match=False, device=model.device)
    
    return model


def evaluate_seeds_sequential(model, seeds, shared_env, expert_model=None):
    """
    Evaluate model on multiple seeds sequentially using a shared environment.
    Uses reset(seed=seed) for fast scenario switching (no env recreation).
    
    Args:
        model: The model to evaluate
        seeds: List of scenario seeds to evaluate
        shared_env: Shared environment (uses reset(seed=seed) for fast switching)
        expert_model: Optional expert model for comparison
    
    Returns:
        List of result dictionaries, one per seed
    """
    all_results = []
    
    for seed_idx, seed in enumerate(seeds):
        
        print(f"  Evaluating seed {seed} ({seed_idx+1}/{len(seeds)})...", end=" ", flush=True)
        
        # Run episode - use reset(seed=seed) for fast scenario switching
        obs = shared_env.reset(seed=seed)
        if isinstance(obs, tuple):
            obs = obs[0]
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        # Cumulative tracking for bad events
        had_crash_vehicle = False
        had_crash_object = False
        had_out_of_road = False
        had_bad_event = False
        crash_count = 0
        arrive_dest = False
        route_completion = 0.0
        
        # Tracking data
        actions_list = []
        expert_diffs = []
        steering_diffs = []
        accel_diffs = []
        
        # Traffic proximity tracking
        min_vehicle_distance = float('inf')
        total_close_encounters = 0
        safe_pass_count = 0
        dangerous_close_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions_list.append(action.copy())
            
            obs, reward, done, info = shared_env.step(action)
            if isinstance(obs, tuple):
                obs = obs[0]
            
            episode_reward += reward
            episode_length += 1
            
            # Track crashes per step
            step_crash = False
            if info.get('crash_vehicle', False):
                had_crash_vehicle = True
                crash_count += 1
                step_crash = True
            if info.get('crash_object', False):
                had_crash_object = True
                crash_count += 1
                step_crash = True
            if info.get('out_of_road', False):
                had_out_of_road = True
            if had_crash_vehicle or had_crash_object or had_out_of_road:
                had_bad_event = True
            
            if info.get('arrive_dest', False):
                arrive_dest = True
            route_completion = max(route_completion, info.get('route_completion', 0.0))
            
            # Traffic proximity tracking
            vehicle_dist = info.get('min_vehicle_distance', -1)
            if vehicle_dist > 0 and vehicle_dist < min_vehicle_distance:
                min_vehicle_distance = vehicle_dist
            
            close_count = info.get('close_vehicle_count', 0)
            if close_count > 0:
                total_close_encounters += close_count
                if step_crash:
                    dangerous_close_count += 1
                else:
                    safe_pass_count += 1
            
            # Expert comparison (get expert action from lidar obs in info)
            if expert_model is not None and 'lidar_obs' in info:
                lidar_obs = info['lidar_obs']
                expert_action, _ = expert_model.predict(lidar_obs, deterministic=True)
                expert_diffs.append(np.linalg.norm(action - expert_action))
                steering_diffs.append(abs(action[0] - expert_action[0]))
                accel_diffs.append(abs(action[1] - expert_action[1]))
        
        # Compute action statistics
        actions_arr = np.array(actions_list)
        
        # Route completion weighted by no bad events
        rc_no_bad = route_completion if not had_bad_event else route_completion * 0.5
        
        # Close encounter crash rate
        close_encounter_crash_rate = 0.0
        if total_close_encounters > 0:
            close_encounter_crash_rate = dangerous_close_count / total_close_encounters
        
        result = {
            'seed': seed,
            'reward': float(episode_reward),
            'length': int(episode_length),
            'route_completion': float(route_completion),
            'route_completion_no_bad_event': float(rc_no_bad),
            'crash_vehicle_rate': float(had_crash_vehicle),
            'crash_object_rate': float(had_crash_object),
            'out_of_road_rate': float(had_out_of_road),
            'any_bad_event_rate': float(had_bad_event),
            'success_rate': float(arrive_dest),
            'success_no_bad_event_rate': float(arrive_dest and not had_bad_event),
            'crash_count_mean': float(crash_count),
            'mean_steering_abs': float(np.mean(np.abs(actions_arr[:, 0]))) if len(actions_arr) > 0 else 0.0,
            'mean_accel_abs': float(np.mean(np.abs(actions_arr[:, 1]))) if len(actions_arr) > 0 else 0.0,
            'hard_brake_ratio': float(np.mean(actions_arr[:, 1] < -0.5)) if len(actions_arr) > 0 else 0.0,
            'hard_steer_ratio': float(np.mean(np.abs(actions_arr[:, 0]) > 0.5)) if len(actions_arr) > 0 else 0.0,
            # Traffic proximity metrics
            'min_vehicle_distance': float(min_vehicle_distance) if min_vehicle_distance != float('inf') else -1,
            'total_close_encounters': float(total_close_encounters),
            'close_encounter_rate': float(total_close_encounters / episode_length) if episode_length > 0 else 0.0,
            'safe_pass_count': float(safe_pass_count),
            'dangerous_close_count': float(dangerous_close_count),
            'close_encounter_crash_rate': float(close_encounter_crash_rate),
        }
        
        # Expert comparison metrics
        if len(expert_diffs) > 0:
            result['expert_action_diff_l2'] = float(np.mean(expert_diffs))
            result['expert_steering_diff'] = float(np.mean(steering_diffs))
            result['expert_accel_diff'] = float(np.mean(accel_diffs))
            result['expert_agreement_ratio'] = float(np.mean(np.array(expert_diffs) < 0.3))
        
        all_results.append(result)
        print(f"reward={episode_reward:.1f}, route={route_completion:.2%}, success={arrive_dest}")
    
    # Don't close shared_env here - it will be reused
    return all_results


def evaluate_seeds_parallel(model, seeds, num_envs, shared_env_config, expert_model=None):
    """
    Evaluate model on multiple seeds in parallel using SubprocVecEnv.
    Uses DYNAMIC SEED ASSIGNMENT: when an env finishes, it immediately starts the next seed.
    No waiting for slow envs!
    
    Args:
        model: The model to evaluate
        seeds: List of scenario seeds to evaluate
        num_envs: Number of parallel environments
        shared_env_config: Config dict for environments
        expert_model: Optional expert model for comparison
    
    Returns:
        List of result dictionaries, one per seed
    """
    from pvp.sb3.common.vec_env import SubprocVecEnv
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    
    # Distribute seeds evenly across environments
    seeds_per_env = [[] for _ in range(num_envs)]
    for i, seed in enumerate(seeds):
        seeds_per_env[i % num_envs].append(seed)
    
    print(f"  Distributing {len(seeds)} seeds across {num_envs} envs")
    for i in range(min(3, num_envs)):
        print(f"    Env {i}: {len(seeds_per_env[i])} seeds - {seeds_per_env[i][:3]}...")
    
    # Find min/max seeds to set proper range for all envs
    min_seed = min(seeds)
    max_seed = max(seeds)
    num_scenarios = max_seed - min_seed + 100  # Extra buffer
    
    print(f"  Seed range: {min_seed}-{max_seed} (num_scenarios={num_scenarios})")
    
    # Create env factories with SeedPoolEnv wrapper
    # All envs use same start_seed and num_scenarios to allow any seed
    import sys as _sys
    _sys.stdout.flush()  # Ensure previous output is flushed
    
    def make_env_with_pool(env_idx):
        env_seeds = seeds_per_env[env_idx]
        # Capture GPU ID from parent process
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        def _init():
            import time as _init_time
            _start = _init_time.time()
            print(f"[GPU Setup] CUDA_VISIBLE_DEVICES={gpu_id}", flush=True)
            
            # Re-set GPU in subprocess (important for spawn/forkserver)
            import os as subprocess_os
            subprocess_os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            subprocess_os.environ["SDL_VIDEODRIVER"] = "offscreen"
            subprocess_os.environ["PYOPENGL_PLATFORM"] = "egl"
            if "DISPLAY" in subprocess_os.environ:
                del subprocess_os.environ["DISPLAY"]
            
            config = shared_env_config.copy()
            config["start_seed"] = min_seed  # Use min seed for all envs
            config["num_scenarios"] = num_scenarios  # Cover all possible seeds
            env = HumanInTheLoopEnv(config=config)
            print(f"  [Env {env_idx}] Created in {_init_time.time()-_start:.1f}s", flush=True)
            return SeedPoolEnv(env, env_seeds)
        return _init
    
    print(f"  Creating {num_envs} subprocess environments...", flush=True)
    import time as _create_time
    _create_start = _create_time.time()
    
    env_fns = [make_env_with_pool(i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    
    print(f"  VecEnv created with {num_envs} envs in {_create_time.time()-_create_start:.1f}s (dynamic seed assignment)", flush=True)
    
    # Reset all (each env starts with its first seed)
    obs = vec_env.reset()
        
    # Per-env episode tracking
    episode_rewards = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs, dtype=int)
    route_completions = np.zeros(num_envs)
    had_crash_vehicle = np.zeros(num_envs, dtype=bool)
    had_crash_object = np.zeros(num_envs, dtype=bool)
    had_out_of_road = np.zeros(num_envs, dtype=bool)
    had_bad_event = np.zeros(num_envs, dtype=bool)
    arrive_dests = np.zeros(num_envs, dtype=bool)
    crash_counts = np.zeros(num_envs, dtype=int)
    min_vehicle_distances = np.full(num_envs, float('inf'))
    total_close_encounters = np.zeros(num_envs)
    safe_pass_counts = np.zeros(num_envs)
    dangerous_close_counts = np.zeros(num_envs)
    
    all_actions = [[] for _ in range(num_envs)]
    all_speeds = [[] for _ in range(num_envs)]  # Track speed per env
    expert_diffs = [[] for _ in range(num_envs)]  # Track expert action diffs per env
    steering_diffs = [[] for _ in range(num_envs)]
    accel_diffs = [[] for _ in range(num_envs)]
    current_seeds = [seeds_per_env[i][0] if seeds_per_env[i] else None for i in range(num_envs)]
    env_finished = np.zeros(num_envs, dtype=bool)  # True when env has no more seeds
    
    all_results = []
    import time as eval_time
    start_time = eval_time.time()
    total_steps = 0
    last_print_count = 0
    
    # Run until all envs have processed all their seeds
    while not np.all(env_finished):
        actions, _ = model.predict(obs, deterministic=True)
        
        # Track actions and expert actions
        for i in range(num_envs):
            if not env_finished[i]:
                all_actions[i].append(actions[i].copy())
        
        
        obs, rewards, dones, infos = vec_env.step(actions)
        total_steps += 1
        
        for i in range(num_envs):
            if env_finished[i]:
                continue
            
            info = infos[i] if isinstance(infos[i], dict) else {}
            
            # Check if this env is finished (no more seeds)
            if info.get('finished', False):
                env_finished[i] = True
                continue
            
            episode_rewards[i] += rewards[i]
            episode_lengths[i] += 1
            route_completions[i] = max(route_completions[i], info.get('route_completion', 0.0))
            
            # Track crashes
            step_crash = False
            if info.get('crash_vehicle', False):
                had_crash_vehicle[i] = True
                crash_counts[i] += 1
                step_crash = True
            if info.get('crash_object', False):
                had_crash_object[i] = True
                crash_counts[i] += 1
                step_crash = True
            if info.get('out_of_road', False):
                had_out_of_road[i] = True
            if had_crash_vehicle[i] or had_crash_object[i] or had_out_of_road[i]:
                had_bad_event[i] = True
            if info.get('arrive_dest', False):
                arrive_dests[i] = True
            
            # Track speed from velocity info
            if 'velocity' in info:
                speed = np.linalg.norm(info['velocity'])
                all_speeds[i].append(speed)
            
            # Traffic proximity - use close_vehicle_count from env
            vehicle_dist = info.get('min_vehicle_distance', -1)
            if vehicle_dist > 0 and vehicle_dist < min_vehicle_distances[i]:
                min_vehicle_distances[i] = vehicle_dist
            
            close_count = info.get('close_vehicle_count', 0)
            if close_count > 0:
                total_close_encounters[i] += close_count
                if step_crash:
                    dangerous_close_counts[i] += 1
                else:
                    safe_pass_counts[i] += 1
            
            # Expert comparison (get expert action from lidar_obs in info)
            if expert_model is not None and 'lidar_obs' in info:
                lidar_obs = info['lidar_obs']
                expert_action, _ = expert_model.predict(lidar_obs, deterministic=True)
                expert_diffs[i].append(np.linalg.norm(actions[i] - expert_action))
                steering_diffs[i].append(abs(actions[i][0] - expert_action[0]))
                accel_diffs[i].append(abs(actions[i][1] - expert_action[1]))
            
            # Episode done - save result and reset tracking
            if dones[i]:
                # Save result for completed episode
                seed = current_seeds[i]
                actions_arr = np.array(all_actions[i]) if all_actions[i] else np.array([[0, 0]])
                rc_no_bad = route_completions[i] if not had_bad_event[i] else route_completions[i] * 0.5
                
                close_encounter_crash_rate = 0.0
                if total_close_encounters[i] > 0:
                    close_encounter_crash_rate = dangerous_close_counts[i] / total_close_encounters[i]
                
                close_encounter_crash_rate = 0.0
                if total_close_encounters[i] > 0:
                    close_encounter_crash_rate = dangerous_close_counts[i] / total_close_encounters[i]
                
                # Compute speed metrics
                speeds_arr = np.array(all_speeds[i]) if all_speeds[i] else np.array([0.0])
                avg_speed = float(np.mean(speeds_arr))
                speed_std = float(np.std(speeds_arr))
                
                # Compute behavioral metrics
                steering_std = float(np.std(actions_arr[:, 0])) if len(actions_arr) > 0 else 0.0
                accel_std = float(np.std(actions_arr[:, 1])) if len(actions_arr) > 0 else 0.0
                
                result = {
                    'seed': seed,
                    'reward': float(episode_rewards[i]),
                    'length': int(episode_lengths[i]),
                    'route_completion': float(route_completions[i]),
                    'route_completion_no_bad_event': float(rc_no_bad),
                    'crash_vehicle_rate': float(had_crash_vehicle[i]),
                    'crash_object_rate': float(had_crash_object[i]),
                    'out_of_road_rate': float(had_out_of_road[i]),
                    'any_bad_event_rate': float(had_bad_event[i]),
                    'success_rate': float(arrive_dests[i]),
                    'success_no_bad_event_rate': float(arrive_dests[i] and not had_bad_event[i]),
                    'crash_count_mean': float(crash_counts[i]),
                    # Behavioral metrics
                    'avg_speed': avg_speed,
                    'speed_std': speed_std,
                    'steering_std': steering_std,
                    'accel_std': accel_std,
                    'mean_steering_abs': float(np.mean(np.abs(actions_arr[:, 0]))) if len(actions_arr) > 0 else 0.0,
                    'mean_accel_abs': float(np.mean(np.abs(actions_arr[:, 1]))) if len(actions_arr) > 0 else 0.0,
                    'hard_brake_ratio': float(np.mean(actions_arr[:, 1] < -0.5)) if len(actions_arr) > 0 else 0.0,
                    'hard_steer_ratio': float(np.mean(np.abs(actions_arr[:, 0]) > 0.5)) if len(actions_arr) > 0 else 0.0,
                    # Traffic metrics
                    'min_vehicle_distance': float(min_vehicle_distances[i]) if min_vehicle_distances[i] != float('inf') else -1,
                    'total_close_encounters': float(total_close_encounters[i]),
                    'close_encounter_rate': float(total_close_encounters[i] / episode_lengths[i]) if episode_lengths[i] > 0 else 0.0,
                    'safe_pass_count': float(safe_pass_counts[i]),
                    'dangerous_close_count': float(dangerous_close_counts[i]),
                    'close_encounter_crash_rate': float(close_encounter_crash_rate),
                }
                
                # Add expert comparison metrics if available
                if len(expert_diffs[i]) > 0:
                    result['expert_action_diff_l2'] = float(np.mean(expert_diffs[i]))
                    result['expert_steering_diff'] = float(np.mean(steering_diffs[i]))
                    result['expert_accel_diff'] = float(np.mean(accel_diffs[i]))
                    result['expert_agreement_ratio'] = float(np.mean(np.array(expert_diffs[i]) < 0.3))
                
                all_results.append(result)
                
                # Reset tracking for next episode (auto-reset already happened)
                episode_rewards[i] = 0
                episode_lengths[i] = 0
                route_completions[i] = 0
                had_crash_vehicle[i] = False
                had_crash_object[i] = False
                had_out_of_road[i] = False
                had_bad_event[i] = False
                arrive_dests[i] = False
                crash_counts[i] = 0
                all_speeds[i] = []  # Reset speed tracking
                min_vehicle_distances[i] = float('inf')
                total_close_encounters[i] = 0
                safe_pass_counts[i] = 0
                dangerous_close_counts[i] = 0
                all_actions[i] = []
                expert_diffs[i] = []  # Reset expert tracking
                steering_diffs[i] = []
                accel_diffs[i] = []
                
                # Update current seed from info (set by SeedPoolEnv)
                current_seeds[i] = info.get('seed', None)
        
        # Print progress every 10 seconds OR every 20 completed episodes
        elapsed = eval_time.time() - start_time
        time_based_print = (elapsed - getattr(evaluate_seeds_parallel, '_last_print_time', 0)) >= 10
        count_based_print = len(all_results) >= last_print_count + 20
        
        if time_based_print or count_based_print:
            if count_based_print:
                last_print_count = (len(all_results) // 20) * 20
            evaluate_seeds_parallel._last_print_time = elapsed
            
            active_envs = num_envs - np.sum(env_finished)
            avg_reward = np.mean([r['reward'] for r in all_results[-20:]]) if len(all_results) >= 20 else (np.mean([r['reward'] for r in all_results]) if all_results else 0)
            avg_success = np.mean([r['success_rate'] for r in all_results[-20:]]) if len(all_results) >= 20 else (np.mean([r['success_rate'] for r in all_results]) if all_results else 0)
            seeds_per_sec = len(all_results) / elapsed if elapsed > 0 else 0
            remaining = len(seeds) - len(all_results)
            eta = remaining / seeds_per_sec if seeds_per_sec > 0 else 0
            
            # Also show step progress (how many steps done in active envs)
            total_episode_steps = sum(episode_lengths)
            print(f"  [{len(all_results)}/{len(seeds)}] {elapsed:.1f}s | "
                  f"{active_envs} active | steps={total_episode_steps} | "
                  f"Reward={avg_reward:.1f} Succ={avg_success:.1%} | "
                  f"ETA={eta:.0f}s", flush=True)
    
    vec_env.close()
    
    print(f"  Done! {len(all_results)} seeds in {eval_time.time() - start_time:.1f}s")
    return all_results


def load_lidar_expert():
    """Load the lidar-based PPO expert for comparison."""
    from pvp.sb3.ppo import PPO
    from pvp.sb3.ppo.policies import ActorCriticPolicy
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    from pvp.sb3.common.save_util import load_from_zip_file
    
    temp_env = HumanInTheLoopEnv(config={'manual_control': False, 'use_render': False})
    
    expert = PPO(
        policy=ActorCriticPolicy,
        env=temp_env,
        n_steps=1024,
        verbose=0,
        device="auto",
    )
    
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    ckpt = script_dir / "pvp" / "experiments" / "metadrive" / "egpo" / "metadrive_pvp_20m_steps"
    
    if ckpt.exists():
        data, params, pytorch_variables = load_from_zip_file(ckpt, device=expert.device, print_system_info=False)
        expert.set_parameters(params, exact_match=True, device=expert.device)
    
    temp_env.close()
    return expert


def main():
    parser = argparse.ArgumentParser(description="Parallel evaluation on hard scenarios")
    parser.add_argument("--model", type=str, required=True, choices=["iql", "td3", "td3bc2", "both", "all", "pretrained"],
                        help="Which model to evaluate (pretrained for testing)")
    parser.add_argument("--num_seeds", type=int, default=200,
                        help="Number of seeds to evaluate (from top 200)")
    parser.add_argument("--num_envs", type=int, default=10,
                        help="Number of parallel environments")
    parser.add_argument("--output", type=str, default="./results/hard_scenario_eval",
                        help="Output directory")
    parser.add_argument("--no_expert", action="store_true",
                        help="Skip expert comparison (faster)")
    parser.add_argument("--iql_checkpoint", type=str, default="IQLBEST1.zip",
                        help="IQL checkpoint path")
    parser.add_argument("--td3_checkpoint", type=str, default="TD3BCBEST1.zip",
                        help="TD3 (BC) checkpoint path")
    parser.add_argument("--td3bc2_checkpoint", type=str, default="TD3BCBEST2.zip",
                        help="TD3BC2 checkpoint path")
    parser.add_argument("--pretrained_checkpoint", type=str, default="pretrained.zip",
                        help="Pretrained checkpoint path (for testing)")
    parser.add_argument("--model_type", type=str, default="td3", choices=["td3", "iql", "td3bc"],
                        help="Model type for loading pretrained checkpoint (td3, iql, or td3bc)")
    # Distributed evaluation arguments
    parser.add_argument("--job_id", type=int, default=0,
                        help="Job ID for distributed evaluation (0 to num_jobs-1)")
    parser.add_argument("--num_jobs", type=int, default=1,
                        help="Total number of distributed jobs")
    parser.add_argument("--save_json", action="store_true",
                        help="Save results to JSON for later merging")
    parser.add_argument("--daytime", type=str, default="06:10",
                        help="Daytime setting for environment (e.g., '06:10', '08:30')")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_seeds = TOP_200_SEEDS[:args.num_seeds]
    
    # Distributed evaluation: split seeds across jobs
    if args.num_jobs > 1:
        seeds_per_job = len(all_seeds) // args.num_jobs
        start_idx = args.job_id * seeds_per_job
        if args.job_id == args.num_jobs - 1:
            # Last job gets remaining seeds
            end_idx = len(all_seeds)
        else:
            end_idx = start_idx + seeds_per_job
        seeds = all_seeds[start_idx:end_idx]
        print(f"[Distributed Job {args.job_id+1}/{args.num_jobs}] Evaluating seeds {start_idx}-{end_idx-1} ({len(seeds)} seeds)")
    else:
        seeds = all_seeds
    
    print(f"Parallel evaluation on {len(seeds)} hardest scenarios")
    print(f"Using {args.num_envs} parallel environments per GPU")
    
    # Load lidar expert for comparison
    expert_model = None
    if not args.no_expert:
        print("Loading lidar expert for comparison...")
        try:
            expert_model = load_lidar_expert()
            print("Expert loaded!")
        except Exception as e:
            print(f"Warning: Could not load expert: {e}")
            expert_model = None
    
    # Create shared env for model loading AND evaluation (uses reset(seed=seed) for fast switching)
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    min_seed = min(seeds)
    max_seed = max(seeds)
    num_scenarios = max_seed - min_seed + 100  # Extra buffer
    shared_env = HumanInTheLoopEnv(config=make_shared_env_config(
        use_image=True, 
        start_seed=min_seed,
        num_scenarios=num_scenarios,
        daytime=args.daytime
    ))
    print(f"Created shared env: start_seed={min_seed}, num_scenarios={num_scenarios}, daytime={args.daytime}")
    
    # Load models
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    models_to_eval = {}
    
    if args.model in ["iql", "both", "all"]:
        iql_path = script_dir / args.iql_checkpoint
        if iql_path.exists():
            print(f"Loading IQL model from {iql_path}...")
            models_to_eval["iql"] = load_model("iql", iql_path, shared_env)
            print("IQL model loaded!")
        else:
            print(f"ERROR: IQL checkpoint not found at {iql_path}")
            shared_env.close()
            return
    
    if args.model in ["td3", "both", "all"]:
        td3_path = script_dir / args.td3_checkpoint
        if td3_path.exists():
            print(f"Loading TD3BC2 model from {td3_path}...")
            models_to_eval["td3bc2"] = load_model("td3", td3_path, shared_env)
            print("TD3BC2 model loaded!")
        else:
            print(f"ERROR: TD3 checkpoint not found at {td3_path}")
            shared_env.close()
            return
    
    if args.model in ["td3bc2", "all"]:
        td3bc2_path = script_dir / args.td3bc2_checkpoint
        if td3bc2_path.exists():
            print(f"Loading TD3BC2 model from {td3bc2_path}...")
            models_to_eval["td3bc2"] = load_model("td3", td3bc2_path, shared_env)
            print("TD3BC2 model loaded!")
        else:
            print(f"ERROR: TD3BC2 checkpoint not found at {td3bc2_path}")
            shared_env.close()
            return
    
    if args.model == "pretrained":
        pretrained_path = script_dir / args.pretrained_checkpoint
        if pretrained_path.exists():
            print(f"Loading pretrained model from {pretrained_path}...")
            models_to_eval["pretrained"] = load_model(args.model_type, pretrained_path, shared_env)
            print("Pretrained model loaded!")
        else:
            print(f"ERROR: Pretrained checkpoint not found at {pretrained_path}")
            shared_env.close()
            return
    
    # Get shared_env_config for parallel evaluation
    shared_env_config = make_shared_env_config(
        use_image=True, 
        start_seed=min_seed,
        num_scenarios=num_scenarios,
        daytime=args.daytime
    )
    
    # Close the shared_env used for model loading
    shared_env.close()
    
    # Evaluate each model (VecEnv created per batch inside evaluate_seeds_parallel)
    all_results = {}
    
    for model_name, model in models_to_eval.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.upper()} on {len(seeds)} scenarios (parallel with {args.num_envs} envs per batch)")
        print(f"{'='*60}")
        
        model_results = evaluate_seeds_parallel(
            model, seeds, args.num_envs, shared_env_config, 
            expert_model=expert_model
        )
        all_results[model_name] = model_results
        
        # Print summary
        rewards = [r['reward'] for r in model_results]
        success_rates = [r['success_rate'] for r in model_results]
        success_no_bad = [r['success_no_bad_event_rate'] for r in model_results]
        route_completion = [r['route_completion'] for r in model_results]
        route_no_bad = [r['route_completion_no_bad_event'] for r in model_results]
        crash_rates = [r['crash_vehicle_rate'] for r in model_results]
        any_bad_rates = [r['any_bad_event_rate'] for r in model_results]
        
        # Traffic proximity metrics
        min_dists = [r['min_vehicle_distance'] for r in model_results if r['min_vehicle_distance'] > 0]
        close_encounters = [r['total_close_encounters'] for r in model_results]
        close_rates = [r['close_encounter_rate'] for r in model_results]
        safe_passes = [r['safe_pass_count'] for r in model_results]
        dangerous_closes = [r['dangerous_close_count'] for r in model_results]
        close_crash_rates = [r['close_encounter_crash_rate'] for r in model_results]
        # Behavioral metrics
        avg_speeds = [r.get('avg_speed', 0) for r in model_results]
        speed_stds = [r.get('speed_std', 0) for r in model_results]
        steering_stds = [r.get('steering_std', 0) for r in model_results]
        accel_stds = [r.get('accel_std', 0) for r in model_results]
        hard_brake_ratios = [r.get('hard_brake_ratio', 0) for r in model_results]
        hard_steer_ratios = [r.get('hard_steer_ratio', 0) for r in model_results]
        
        # Out of road rate
        out_of_road_rates = [r.get('out_of_road_rate', 0) for r in model_results]
        
        print(f"\n{model_name.upper()} Summary:")
        print(f"  Reward: mean={np.mean(rewards):.1f}, std={np.std(rewards):.1f}")
        print(f"  Route Completion: {np.mean(route_completion)*100:.1f}%")
        print(f"  Route Completion (no bad event): {np.mean(route_no_bad)*100:.1f}%")
        print(f"  Success Rate: {np.mean(success_rates)*100:.1f}%")
        print(f"  Success Rate (no bad event): {np.mean(success_no_bad)*100:.1f}%")
        print(f"  --- Safety Metrics ---")
        print(f"  Crash Vehicle Rate: {np.mean(crash_rates)*100:.1f}%")
        print(f"  Out of Road Rate: {np.mean(out_of_road_rates)*100:.1f}%")
        print(f"  Any Bad Event Rate: {np.mean(any_bad_rates)*100:.1f}%")
        print(f"  --- Behavioral Metrics ---")
        print(f"  Avg Speed: {np.mean(avg_speeds):.2f} m/s")
        print(f"  Speed Std: {np.mean(speed_stds):.2f}")
        print(f"  Steering Std: {np.mean(steering_stds):.4f}")
        print(f"  Accel Std: {np.mean(accel_stds):.4f}")
        print(f"  Hard Brake Ratio: {np.mean(hard_brake_ratios)*100:.1f}%")
        print(f"  Hard Steer Ratio: {np.mean(hard_steer_ratios)*100:.1f}%")
        print(f"  --- Traffic Proximity ---")
        print(f"  Min Vehicle Distance: {np.mean(min_dists):.2f}m" if min_dists else "  Min Vehicle Distance: N/A")
        print(f"  Avg Close Encounters: {np.mean(close_encounters):.1f}")
        print(f"  Close Encounter Rate: {np.mean(close_rates)*100:.1f}%")
        print(f"  Safe Pass Count: {np.mean(safe_passes):.1f}")
        # Compute safe pass rate
        total_close_enc = sum(close_encounters)
        total_safe_pass = sum(safe_passes)
        safe_pass_rate = total_safe_pass / total_close_enc if total_close_enc > 0 else 0.0
        print(f"  Safe Pass Rate: {safe_pass_rate*100:.1f}%")
        print(f"  Dangerous Close Count: {np.mean(dangerous_closes):.1f}")
        print(f"  Close Encounter Crash Rate: {np.mean(close_crash_rates)*100:.1f}%")
        
        # Expert Comparison Metrics (if available)
        expert_diffs = [r.get('expert_action_diff_l2', None) for r in model_results]
        expert_diffs = [d for d in expert_diffs if d is not None]
        if expert_diffs:
            print(f"  --- Expert Comparison ---")
            print(f"  Expert Action Diff (L2): {np.mean(expert_diffs):.4f}")
            steering_diffs = [r.get('expert_steering_diff', 0) for r in model_results if 'expert_steering_diff' in r]
            accel_diffs = [r.get('expert_accel_diff', 0) for r in model_results if 'expert_accel_diff' in r]
            agreements = [r.get('expert_agreement_ratio', 0) for r in model_results if 'expert_agreement_ratio' in r]
            if steering_diffs:
                print(f"  Expert Steering Diff: {np.mean(steering_diffs):.4f}")
            if accel_diffs:
                print(f"  Expert Accel Diff: {np.mean(accel_diffs):.4f}")
            if agreements:
                print(f"  Expert Agreement Ratio: {np.mean(agreements)*100:.1f}%")
        
        # Difficulty Segment Analysis
        print(f"  --- Difficulty Segments (Hardest to Easiest) ---")
        segment_names = ["hardest", "hard", "medium", "easy", "easiest"]
        num_per_segment = len(model_results) // 5 if len(model_results) >= 5 else len(model_results)
        for seg_idx, seg_name in enumerate(segment_names):
            start_idx = seg_idx * num_per_segment
            end_idx = min(start_idx + num_per_segment, len(model_results))
            if start_idx >= len(model_results):
                break
            seg_results = model_results[start_idx:end_idx]
            seg_success = np.mean([r['success_rate'] for r in seg_results]) * 100
            seg_crash = np.mean([r['crash_vehicle_rate'] for r in seg_results]) * 100
            seg_reward = np.mean([r['reward'] for r in seg_results])
            print(f"  [{seg_name:8s}] Success: {seg_success:5.1f}%, Crash: {seg_crash:5.1f}%, Reward: {seg_reward:.1f}")
    
    # Save results
    if args.num_jobs > 1:
        # Distributed mode: save partial results with job_id
        result_file = output_path / f"hard_scenario_results_job{args.job_id}.json"
        with open(result_file, 'w') as f:
            json.dump({
                'job_id': args.job_id,
                'num_jobs': args.num_jobs,
                'seeds_evaluated': seeds,
                'results': all_results
            }, f, indent=2)
        print(f"\n[Job {args.job_id}] Saved partial results to {result_file}")
        print(f"Run merge script after all jobs complete to get final results.")
    else:
        with open(output_path / "hard_scenario_results_parallel.json", 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Model comparison (skip for distributed jobs, will be done after merge)
    if args.num_jobs > 1:
        print(f"\n[Job {args.job_id}] Completed. Merge results after all {args.num_jobs} jobs finish.")
        total_time = time.time() - start_time
        print(f"[Job {args.job_id}] Total time: {total_time/60:.1f} minutes")
        return
    
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        model_names = list(all_results.keys())
        
        print(f"\n{'Metric':<30}", end="")
        for name in model_names:
            print(f"{name.upper():<15}", end="")
        print()
        print("-" * (30 + 15 * len(model_names)))
        
        metrics = [
            ('reward', 'Reward (mean)', lambda x: np.mean([r['reward'] for r in x]), '{:.1f}'),
            ('success', 'Success Rate', lambda x: np.mean([r['success_rate'] for r in x]), '{:.1%}'),
            ('success_nb', 'Success (no bad)', lambda x: np.mean([r['success_no_bad_event_rate'] for r in x]), '{:.1%}'),
            ('route', 'Route Completion', lambda x: np.mean([r['route_completion'] for r in x]), '{:.1%}'),
            ('route_nb', 'Route (no bad)', lambda x: np.mean([r['route_completion_no_bad_event'] for r in x]), '{:.1%}'),
            ('bad_event', 'Any Bad Event', lambda x: np.mean([r['any_bad_event_rate'] for r in x]), '{:.1%}'),
            ('crash', 'Crash Vehicle', lambda x: np.mean([r['crash_vehicle_rate'] for r in x]), '{:.1%}'),
            # Traffic proximity metrics
            ('min_dist', 'Min Vehicle Dist (m)', lambda x: np.mean([r['min_vehicle_distance'] for r in x if r['min_vehicle_distance'] > 0]), '{:.2f}'),
            ('close_enc', 'Avg Close Encounters', lambda x: np.mean([r['total_close_encounters'] for r in x]), '{:.1f}'),
            ('close_rate', 'Close Enc Rate', lambda x: np.mean([r['close_encounter_rate'] for r in x]), '{:.1%}'),
            ('close_crash', 'Close Enc Crash Rate', lambda x: np.mean([r['close_encounter_crash_rate'] for r in x]), '{:.1%}'),
        ]
        
        for key, label, func, fmt in metrics:
            print(f"{label:<30}", end="")
            for name in model_names:
                val = func(all_results[name])
                print(f"{fmt.format(val):<15}", end="")
            print()
    
    # VecEnv is closed after each batch inside evaluate_seeds_parallel
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Results saved to {output_path}/")


if __name__ == "__main__":
    main()
