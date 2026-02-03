"""
Evaluate BC checkpoint on hard scenarios.
Uses the TOP_200_SEEDS from eval_hard_scenarios_parallel.py (different from training seeds).

Usage:
    python eval_bc_checkpoint.py --checkpoint /data/caihy/bc_training/xxx/bc_step_010000.zip --num_envs 30
"""

import os
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
import psutil
import collections
from pathlib import Path

import gymnasium
import gymnasium.spaces
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
gymnasium.spaces.space = gymnasium.spaces

# Suppress metadrive logging
import logging
logging.getLogger('metadrive').setLevel(logging.WARNING)
logging.getLogger('panda3d').setLevel(logging.WARNING)

# Top 200 hardest scenarios for TRAINING (seeds in [0, 1000))
# MUST match generate_bc_data_sequential.py HARD_200_SEEDS exactly!
TOP_200_TRAIN_SEEDS = [
    751, 849, 61, 963, 160, 257, 678, 491, 892, 410,
    403, 35, 473, 725, 171, 536, 376, 192, 755, 169,
    947, 586, 59, 535, 418, 694, 424, 452, 64, 100,
    548, 973, 180, 369, 313, 488, 557, 821, 318, 91,
    481, 969, 230, 120, 646, 942, 73, 806, 549, 315,
    799, 816, 740, 781, 382, 738, 609, 765, 930, 932,
    534, 747, 183, 401, 300, 692, 41, 367, 804, 959,
    400, 984, 492, 46, 389, 354, 706, 448, 435, 92,
    277, 543, 986, 903, 413, 239, 90, 484, 837, 819,
    791, 454, 459, 758, 85, 168, 194, 567, 680, 205,
    126, 373, 297, 208, 44, 794, 525, 533, 57, 142,
    12, 762, 844, 598, 685, 368, 832, 946, 966, 178,
    250, 572, 9, 825, 584, 11, 728, 269, 232, 898,
    668, 374, 29, 760, 238, 5, 653, 524, 950, 611,
    888, 243, 185, 215, 319, 934, 72, 923, 345, 236,
    119, 248, 364, 553, 985, 199, 207, 392, 352, 355,
    214, 894, 70, 847, 156, 336, 221, 965, 184, 735,
    620, 305, 944, 718, 226, 773, 530, 987, 891, 261,
    859, 394, 188, 951, 622, 769, 340, 225, 344, 540,
    640, 975, 114, 727, 68, 310, 265, 273, 726, 42,
]

# Top 200 hardest scenarios for TESTING (seeds in [1000, 2000))
# (Different from training seeds!)
TOP_200_EVAL_SEEDS = [
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


def get_memory_mb():
    """Get current memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


class SeedPoolEnv:
    """Wrapper that manages a queue of seeds for continuous evaluation."""
    def __init__(self, env, seeds):
        self.env = env
        self._original_seeds = list(seeds)
        self.seed_queue = collections.deque(seeds)
        self.current_seed = None
        self.finished = False
        self._last_info = {}
    
    def reset_pool(self):
        self.seed_queue = collections.deque(self._original_seeds)
        self.current_seed = None
        self.finished = False
        return self.reset()
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    
    def reset(self, **kwargs):
        if not self.seed_queue:
            self.finished = True
            dummy_obs = self.observation_space.sample()
            self._last_info = {'finished': True, 'seed': None}
            return dummy_obs
        
        self.current_seed = self.seed_queue.popleft()
        self.finished = False
        result = self.env.reset(seed=self.current_seed)
        
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
        return obs
    
    def step(self, action):
        if self.finished:
            dummy_obs = self.observation_space.sample()
            self._last_info = {'finished': True, 'seed': None}
            return dummy_obs, 0, True, self._last_info
        
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        
        if not isinstance(info, dict):
            info = {}
        info['seed'] = self.current_seed
        info['finished'] = False
        self._last_info = info
        return obs, reward, done, info
    
    def close(self):
        return self.env.close()


def make_env_config(daytime="08:30", use_original_config=False, use_test_seeds=True):
    """Create environment config for image-based evaluation.
    
    CORRECT CONFIG (matching training data generation):
        - image_observation=True
        - crash_vehicle_done=False
        - traffic_density=NOT SET (env default 0.06)
        - random_traffic=NOT SET (env default)
        - daytime=as specified
    
    Args:
        daytime: Time of day setting
        use_original_config: DEPRECATED - now always uses correct config
        use_test_seeds: If True, use start_seed=1000 (test), else start_seed=0 (train)
    """
    from metadrive.component.sensors.rgb_camera import RGBCamera
    
    # Set start_seed based on which seed set we're using
    start_seed = 1000 if use_test_seeds else 0
    
    # CORRECT config: match the intended training/eval settings
    # DO NOT SET traffic_density or random_traffic - use env defaults (0.06)
    return dict(
        use_render=False,
        manual_control=False,
        num_scenarios=1000,
        start_seed=start_seed,  # Train seeds [0,1000) or Test seeds [1000,2000)
        horizon=1500,
        # Key safety settings
        crash_vehicle_done=False,
        crash_object_done=False,
        cost_to_reward=False,
        crash_vehicle_penalty=5.0,
        crash_object_penalty=5.0,
        out_of_road_penalty=5.0,
        # Image observation
        image_observation=True,
        vehicle_config=dict(image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, 84, 84)},
        stack_size=3,
        interface_panel=["rgb_camera", "dashboard"],
        # Daytime
        daytime=daytime,
        # NOTE: traffic_density and random_traffic NOT SET
        # This uses env default (traffic_density=0.06)
    )


def load_bc_model(checkpoint_path, env):
    """Load BC or IQL model from checkpoint."""
    import zipfile
    from pvp.sb3.sac.our_features_extractor import OurFeaturesExtractorCNN as OurFeaturesExtractor
    from pvp.sb3.common.save_util import load_from_zip_file
    
    # Check if this is an IQL checkpoint (has value_features_extractor)
    is_iql = False
    with zipfile.ZipFile(checkpoint_path, 'r') as zf:
        if 'value_features_extractor.pth' in zf.namelist() or 'value_mlp.pth' in zf.namelist():
            is_iql = True
    
    policy_kwargs = dict(
        features_extractor_class=OurFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=275),
        share_features_extractor=False,
        net_arch=[256,]
    )
    
    if is_iql:
        print(f"Detected IQL checkpoint, loading with IQL model...")
        from pvp.sb3.td3.iql import IQL
        from pvp.sb3.td3.policies import TD3Policy
        
        model = IQL(
            policy=TD3Policy,
            env=env,
            learning_rate=1e-4,
            buffer_size=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            verbose=0,
            device="auto",
            policy_kwargs=policy_kwargs,
            iql_tau=0.7,
            iql_beta=3.0,
        )
    else:
        print(f"Detected BC/TD3 checkpoint, loading with TD3 model...")
        from pvp.sb3.td3.td3 import TD3
        from pvp.sb3.td3.policies import TD3Policy
        
        model = TD3(
            policy=TD3Policy,
            env=env,
            learning_rate=1e-4,
            buffer_size=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            verbose=0,
            device="auto",
            policy_kwargs=policy_kwargs,
        )
    
    data, params, pytorch_variables = load_from_zip_file(
        checkpoint_path, device=model.device, print_system_info=False
    )
    model.set_parameters(params, exact_match=False, device=model.device)
    
    return model


def evaluate_sequential(model, seeds, env_config):
    """Sequential evaluation using a single environment."""
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    
    min_seed = min(seeds)
    max_seed = max(seeds)
    num_scenarios = max_seed - min_seed + 100
    
    config = env_config.copy()
    config["start_seed"] = min_seed
    config["num_scenarios"] = num_scenarios
    env = HumanInTheLoopEnv(config=config)
    
    all_results = []
    start_time = time.time()
    
    for i, seed in enumerate(seeds):
        obs, info = env.reset(seed=seed)
        
        episode_reward = 0
        episode_length = 0
        route_completion = 0
        had_crash_vehicle = False
        had_crash_object = False
        had_out_of_road = False
        had_bad_event = False
        arrive_dest = False
        crash_count = 0
        min_vehicle_distance = float('inf')
        total_close_encounters = 0
        safe_pass_count = 0
        dangerous_close_count = 0
        all_actions = []
        all_speeds = []
        
        # Reward component tracking
        total_step_rewards = 0.0
        total_crash_penalty = 0.0
        total_out_of_road_penalty = 0.0
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            route_completion = max(route_completion, info.get('route_completion', 0.0))
            all_actions.append(action.copy())
            
            # Track reward components
            total_step_rewards += info.get('step_reward', reward)
            
            step_crash = False
            if info.get('crash_vehicle', False):
                had_crash_vehicle = True
                crash_count += 1
                step_crash = True
                total_crash_penalty += 5.0  # crash_vehicle_penalty
            if info.get('crash_object', False):
                had_crash_object = True
                crash_count += 1
                step_crash = True
                total_crash_penalty += 5.0  # crash_object_penalty
            if info.get('out_of_road', False):
                had_out_of_road = True
                total_out_of_road_penalty += 5.0  # out_of_road_penalty
            if had_crash_vehicle or had_crash_object or had_out_of_road:
                had_bad_event = True
            if info.get('arrive_dest', False):
                arrive_dest = True
            
            if 'velocity' in info:
                speed = np.linalg.norm(info['velocity'])
                all_speeds.append(speed)
            
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
        
        actions_arr = np.array(all_actions) if all_actions else np.array([[0, 0]])
        speeds_arr = np.array(all_speeds) if all_speeds else np.array([0.0])
        rc_no_bad = route_completion if not had_bad_event else route_completion * 0.5
        close_enc_crash_rate = dangerous_close_count / total_close_encounters if total_close_encounters > 0 else 0.0
        
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
            'avg_speed': float(np.mean(speeds_arr)),
            'speed_std': float(np.std(speeds_arr)),
            'steering_std': float(np.std(actions_arr[:, 0])) if len(actions_arr) > 0 else 0.0,
            'accel_std': float(np.std(actions_arr[:, 1])) if len(actions_arr) > 0 else 0.0,
            'mean_steering_abs': float(np.mean(np.abs(actions_arr[:, 0]))) if len(actions_arr) > 0 else 0.0,
            'mean_accel_abs': float(np.mean(np.abs(actions_arr[:, 1]))) if len(actions_arr) > 0 else 0.0,
            'hard_brake_ratio': float(np.mean(actions_arr[:, 1] < -0.5)) if len(actions_arr) > 0 else 0.0,
            'hard_steer_ratio': float(np.mean(np.abs(actions_arr[:, 0]) > 0.5)) if len(actions_arr) > 0 else 0.0,
            'min_vehicle_distance': float(min_vehicle_distance) if min_vehicle_distance != float('inf') else -1,
            'total_close_encounters': float(total_close_encounters),
            'close_encounter_rate': float(total_close_encounters / episode_length) if episode_length > 0 else 0.0,
            'safe_pass_count': float(safe_pass_count),
            'dangerous_close_count': float(dangerous_close_count),
            'close_encounter_crash_rate': float(close_enc_crash_rate),
            # Reward component breakdown
            'total_step_rewards': float(total_step_rewards),
            'total_crash_penalty': float(total_crash_penalty),
            'total_out_of_road_penalty': float(total_out_of_road_penalty),
            'estimated_driving_reward': float(total_step_rewards + total_crash_penalty + total_out_of_road_penalty),
        }
        all_results.append(result)
        
        elapsed = time.time() - start_time
        if (i + 1) % 10 == 0 or i == 0:
            avg_success = np.mean([r['success_rate'] for r in all_results])
            avg_reward = np.mean([r['reward'] for r in all_results])
            eta = elapsed / (i + 1) * (len(seeds) - i - 1)
            print(f"  [{i+1}/{len(seeds)}] Reward={avg_reward:.1f}, Success={avg_success:.1%}, Mem={get_memory_mb():.0f}MB, ETA={eta:.0f}s", flush=True)
    
    env.close()
    print(f"  Done! {len(all_results)} seeds in {time.time() - start_time:.1f}s")
    return all_results


def evaluate_parallel(model, seeds, num_envs, env_config):
    """Parallel evaluation using SubprocVecEnv."""
    from pvp.sb3.common.vec_env import SubprocVecEnv
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    
    # Distribute seeds
    seeds_per_env = [[] for _ in range(num_envs)]
    for i, seed in enumerate(seeds):
        seeds_per_env[i % num_envs].append(seed)
    
    print(f"  Distributing {len(seeds)} seeds across {num_envs} envs")
    
    min_seed = min(seeds)
    max_seed = max(seeds)
    num_scenarios = max_seed - min_seed + 100
    
    def make_env_with_pool(env_idx):
        env_seeds = seeds_per_env[env_idx]
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        
        def _init():
            import os as subprocess_os
            subprocess_os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            subprocess_os.environ["SDL_VIDEODRIVER"] = "offscreen"
            subprocess_os.environ["PYOPENGL_PLATFORM"] = "egl"
            if "DISPLAY" in subprocess_os.environ:
                del subprocess_os.environ["DISPLAY"]
            
            config = env_config.copy()
            config["start_seed"] = min_seed
            config["num_scenarios"] = num_scenarios
            env = HumanInTheLoopEnv(config=config)
            return SeedPoolEnv(env, env_seeds)
        return _init
    
    print(f"  Creating {num_envs} subprocess environments...", flush=True)
    env_fns = [make_env_with_pool(i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    print(f"  VecEnv created with {num_envs} envs", flush=True)
    
    # Reset
    obs = vec_env.reset()
    
    # Per-env tracking
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
    # Reward decomposition tracking
    total_crash_penalties = np.zeros(num_envs)
    total_out_of_road_penalties = np.zeros(num_envs)
    out_of_road_counts = np.zeros(num_envs, dtype=int)
    
    all_actions = [[] for _ in range(num_envs)]
    all_speeds = [[] for _ in range(num_envs)]
    current_seeds = [seeds_per_env[i][0] if seeds_per_env[i] else None for i in range(num_envs)]
    env_finished = np.zeros(num_envs, dtype=bool)
    
    all_results = []
    start_time = time.time()
    last_print_time = 0
    last_print_count = 0
    
    while not np.all(env_finished):
        actions, _ = model.predict(obs, deterministic=True)
        
        for i in range(num_envs):
            if not env_finished[i]:
                all_actions[i].append(actions[i].copy())
        
        obs, rewards, dones, infos = vec_env.step(actions)
        
        for i in range(num_envs):
            if env_finished[i]:
                continue
            
            info = infos[i] if isinstance(infos[i], dict) else {}
            
            if info.get('finished', False):
                env_finished[i] = True
                continue
            
            episode_rewards[i] += rewards[i]
            episode_lengths[i] += 1
            route_completions[i] = max(route_completions[i], info.get('route_completion', 0.0))
            
            # Track crashes and penalties
            step_crash = False
            if info.get('crash_vehicle', False):
                had_crash_vehicle[i] = True
                crash_counts[i] += 1
                total_crash_penalties[i] += 5.0  # crash_vehicle_penalty
                step_crash = True
            if info.get('crash_object', False):
                had_crash_object[i] = True
                crash_counts[i] += 1
                total_crash_penalties[i] += 5.0  # crash_object_penalty
                step_crash = True
            if info.get('out_of_road', False):
                had_out_of_road[i] = True
                out_of_road_counts[i] += 1
                total_out_of_road_penalties[i] += 5.0  # out_of_road_penalty
            if had_crash_vehicle[i] or had_crash_object[i] or had_out_of_road[i]:
                had_bad_event[i] = True
            if info.get('arrive_dest', False):
                arrive_dests[i] = True
            
            # Speed tracking
            if 'velocity' in info:
                speed = np.linalg.norm(info['velocity'])
                all_speeds[i].append(speed)
            
            # Traffic proximity
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
            
            # Episode done
            if dones[i]:
                seed = current_seeds[i]
                actions_arr = np.array(all_actions[i]) if all_actions[i] else np.array([[0, 0]])
                rc_no_bad = route_completions[i] if not had_bad_event[i] else route_completions[i] * 0.5
                
                close_enc_crash_rate = 0.0
                if total_close_encounters[i] > 0:
                    close_enc_crash_rate = dangerous_close_counts[i] / total_close_encounters[i]
                
                speeds_arr = np.array(all_speeds[i]) if all_speeds[i] else np.array([0.0])
                
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
                    'avg_speed': float(np.mean(speeds_arr)),
                    'speed_std': float(np.std(speeds_arr)),
                    'steering_std': float(np.std(actions_arr[:, 0])) if len(actions_arr) > 0 else 0.0,
                    'accel_std': float(np.std(actions_arr[:, 1])) if len(actions_arr) > 0 else 0.0,
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
                    'close_encounter_crash_rate': float(close_enc_crash_rate),
                    # Reward decomposition (PRECISE tracking)
                    'total_step_rewards': float(episode_rewards[i]),
                    'total_crash_penalty': float(total_crash_penalties[i]),
                    'total_out_of_road_penalty': float(total_out_of_road_penalties[i]),
                    'out_of_road_count': int(out_of_road_counts[i]),
                    'estimated_driving_reward': float(episode_rewards[i] + total_crash_penalties[i] + total_out_of_road_penalties[i]),
                }
                
                all_results.append(result)
                
                # Reset tracking
                episode_rewards[i] = 0
                episode_lengths[i] = 0
                route_completions[i] = 0
                had_crash_vehicle[i] = False
                had_crash_object[i] = False
                had_out_of_road[i] = False
                had_bad_event[i] = False
                arrive_dests[i] = False
                crash_counts[i] = 0
                all_speeds[i] = []
                min_vehicle_distances[i] = float('inf')
                total_close_encounters[i] = 0
                safe_pass_counts[i] = 0
                dangerous_close_counts[i] = 0
                all_actions[i] = []
                current_seeds[i] = info.get('seed', None)
                # Reset reward decomposition
                total_crash_penalties[i] = 0
                total_out_of_road_penalties[i] = 0
                out_of_road_counts[i] = 0
        
        # Progress logging
        elapsed = time.time() - start_time
        if elapsed - last_print_time >= 10 or len(all_results) >= last_print_count + 20:
            last_print_time = elapsed
            last_print_count = (len(all_results) // 20) * 20
            
            active_envs = num_envs - np.sum(env_finished)
            
            # Compute CUMULATIVE running metrics (all results so far)
            if all_results:
                avg_reward = np.mean([r['reward'] for r in all_results])
                avg_success = np.mean([r['success_rate'] for r in all_results])
                avg_success_no_bad = np.mean([r['success_no_bad_event_rate'] for r in all_results])
                avg_route_comp = np.mean([r['route_completion'] for r in all_results])
                avg_route_no_bad = np.mean([r['route_completion_no_bad_event'] for r in all_results])
                avg_crash_rate = np.mean([r['crash_vehicle_rate'] for r in all_results])
                # Reward decomposition
                avg_crash_penalty = np.mean([r.get('total_crash_penalty', 0) for r in all_results])
                avg_oor_penalty = np.mean([r.get('total_out_of_road_penalty', 0) for r in all_results])
                avg_driving_reward = np.mean([r.get('estimated_driving_reward', 0) for r in all_results])
            else:
                avg_reward = avg_success = avg_success_no_bad = avg_route_comp = avg_route_no_bad = avg_crash_rate = 0
                avg_crash_penalty = avg_oor_penalty = avg_driving_reward = 0
            
            seeds_per_sec = len(all_results) / elapsed if elapsed > 0 else 0
            remaining = len(seeds) - len(all_results)
            eta = remaining / seeds_per_sec if seeds_per_sec > 0 else 0
            mem = get_memory_mb()
            
            print(f"  [{len(all_results)}/{len(seeds)}] {elapsed:.1f}s | "
                  f"R={avg_reward:.1f} Succ={avg_success:.0%} SuccNoBad={avg_success_no_bad:.0%} "
                  f"RC={avg_route_comp:.0%} RCNoBad={avg_route_no_bad:.0%} Crash={avg_crash_rate:.0%} | "
                  f"CrashP={avg_crash_penalty:.1f} OorP={avg_oor_penalty:.1f} DrivR={avg_driving_reward:.1f} | "
                  f"ETA={eta:.0f}s", flush=True)
    
    vec_env.close()
    print(f"  Done! {len(all_results)} seeds in {time.time() - start_time:.1f}s")
    return all_results


def compute_metrics(results):
    """Compute all evaluation metrics from results."""
    metrics = {}
    
    # Core metrics
    rewards = [r['reward'] for r in results]
    success_rates = [r['success_rate'] for r in results]
    success_no_bad = [r['success_no_bad_event_rate'] for r in results]
    route_completion = [r['route_completion'] for r in results]
    route_no_bad = [r['route_completion_no_bad_event'] for r in results]
    crash_vehicle_rates = [r['crash_vehicle_rate'] for r in results]
    crash_object_rates = [r['crash_object_rate'] for r in results]
    out_of_road_rates = [r['out_of_road_rate'] for r in results]
    any_bad_rates = [r['any_bad_event_rate'] for r in results]
    lengths = [r['length'] for r in results]
    
    metrics['eval/reward_mean'] = float(np.mean(rewards))
    metrics['eval/reward_std'] = float(np.std(rewards))
    metrics['eval/success_rate'] = float(np.mean(success_rates))
    metrics['eval/success_rate_no_bad'] = float(np.mean(success_no_bad))
    metrics['eval/route_completion'] = float(np.mean(route_completion))
    metrics['eval/route_completion_no_bad'] = float(np.mean(route_no_bad))
    metrics['eval/crash_vehicle_rate'] = float(np.mean(crash_vehicle_rates))
    metrics['eval/crash_object_rate'] = float(np.mean(crash_object_rates))
    metrics['eval/out_of_road_rate'] = float(np.mean(out_of_road_rates))
    metrics['eval/any_bad_event_rate'] = float(np.mean(any_bad_rates))
    metrics['eval/episode_length_mean'] = float(np.mean(lengths))
    
    # Behavioral metrics
    avg_speeds = [r.get('avg_speed', 0) for r in results]
    speed_stds = [r.get('speed_std', 0) for r in results]
    steering_stds = [r.get('steering_std', 0) for r in results]
    accel_stds = [r.get('accel_std', 0) for r in results]
    hard_brake_ratios = [r.get('hard_brake_ratio', 0) for r in results]
    hard_steer_ratios = [r.get('hard_steer_ratio', 0) for r in results]
    mean_steering_abs = [r.get('mean_steering_abs', 0) for r in results]
    mean_accel_abs = [r.get('mean_accel_abs', 0) for r in results]
    
    metrics['eval/avg_speed'] = float(np.mean(avg_speeds))
    metrics['eval/speed_std'] = float(np.mean(speed_stds))
    metrics['eval/steering_std'] = float(np.mean(steering_stds))
    metrics['eval/accel_std'] = float(np.mean(accel_stds))
    metrics['eval/hard_brake_ratio'] = float(np.mean(hard_brake_ratios))
    metrics['eval/hard_steer_ratio'] = float(np.mean(hard_steer_ratios))
    metrics['eval/mean_steering_abs'] = float(np.mean(mean_steering_abs))
    metrics['eval/mean_accel_abs'] = float(np.mean(mean_accel_abs))
    
    # Traffic proximity metrics
    min_dists = [r['min_vehicle_distance'] for r in results if r['min_vehicle_distance'] > 0]
    close_encounters = [r['total_close_encounters'] for r in results]
    close_rates = [r['close_encounter_rate'] for r in results]
    safe_passes = [r['safe_pass_count'] for r in results]
    dangerous_closes = [r['dangerous_close_count'] for r in results]
    close_crash_rates = [r['close_encounter_crash_rate'] for r in results]
    
    metrics['eval/min_vehicle_distance'] = float(np.mean(min_dists)) if min_dists else -1
    metrics['eval/total_close_encounters'] = float(np.mean(close_encounters))
    metrics['eval/close_encounter_rate'] = float(np.mean(close_rates))
    metrics['eval/safe_pass_count'] = float(np.mean(safe_passes))
    metrics['eval/dangerous_close_count'] = float(np.mean(dangerous_closes))
    metrics['eval/close_encounter_crash_rate'] = float(np.mean(close_crash_rates))
    
    # Safe pass rate
    total_close_enc = sum(close_encounters)
    total_safe_pass = sum(safe_passes)
    metrics['eval/safe_pass_rate'] = float(total_safe_pass / total_close_enc) if total_close_enc > 0 else 0.0
    
    # Reward component breakdown
    total_step_rewards = [r.get('total_step_rewards', 0) for r in results]
    total_crash_penalties = [r.get('total_crash_penalty', 0) for r in results]
    total_out_of_road_penalties = [r.get('total_out_of_road_penalty', 0) for r in results]
    estimated_driving_rewards = [r.get('estimated_driving_reward', 0) for r in results]
    
    metrics['reward/total_step_rewards_mean'] = float(np.mean(total_step_rewards))
    metrics['reward/crash_penalty_mean'] = float(np.mean(total_crash_penalties))
    metrics['reward/out_of_road_penalty_mean'] = float(np.mean(total_out_of_road_penalties))
    metrics['reward/estimated_driving_reward_mean'] = float(np.mean(estimated_driving_rewards))
    
    # Difficulty segment analysis
    num_per_segment = len(results) // 5 if len(results) >= 5 else len(results)
    segment_names = ["hardest", "hard", "medium", "easy", "easiest"]
    for seg_idx, seg_name in enumerate(segment_names):
        start_idx = seg_idx * num_per_segment
        end_idx = min(start_idx + num_per_segment, len(results))
        if start_idx >= len(results):
            break
        seg_results = results[start_idx:end_idx]
        metrics[f'eval/segment_{seg_name}_success'] = float(np.mean([r['success_rate'] for r in seg_results]))
        metrics[f'eval/segment_{seg_name}_crash'] = float(np.mean([r['crash_vehicle_rate'] for r in seg_results]))
        metrics[f'eval/segment_{seg_name}_reward'] = float(np.mean([r['reward'] for r in seg_results]))
    
    return metrics


def load_lidar_expert(device="auto"):
    """Load the lidar-based PPO expert."""
    from pvp.sb3.ppo import PPO
    from pvp.sb3.ppo.policies import ActorCriticPolicy
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    from pvp.sb3.common.save_util import load_from_zip_file
    
    # Create a temporary lidar env for model loading
    temp_env = HumanInTheLoopEnv(config={
        'manual_control': False, 
        'use_render': False,
        'image_observation': False,
    })
    
    expert = PPO(
        policy=ActorCriticPolicy,
        env=temp_env,
        n_steps=1024,
        verbose=0,
        device=device,
    )
    
    # Use absolute path
    ckpt = Path("/p0/user/caihy/pvp/pvp/experiments/metadrive/egpo/metadrive_pvp_20m_steps.zip")
    
    print(f"Loading lidar expert from: {ckpt}")
    if ckpt.exists():
        data, params, pytorch_variables = load_from_zip_file(ckpt, device=expert.device, print_system_info=False)
        expert.set_parameters(params, exact_match=True, device=expert.device)
    else:
        raise FileNotFoundError(f"Lidar expert not found at {ckpt}")
    
    temp_env.close()
    return expert


def make_lidar_env_config(daytime="08:30", use_test_seeds=False):
    """Create environment config for lidar-based evaluation (matching image eval but without image)."""
    start_seed = 1000 if use_test_seeds else 0
    
    return dict(
        manual_control=False,
        use_render=False,
        start_seed=start_seed,
        num_scenarios=1000,
        horizon=1500,
        crash_vehicle_done=False,
        crash_object_done=False,
        cost_to_reward=False,
        crash_vehicle_penalty=5.0,
        crash_object_penalty=5.0,
        out_of_road_penalty=5.0,
        # Lidar observation (NOT image)
        image_observation=False,
        # Daytime (may not affect lidar, but keep consistent)
        daytime=daytime,
        # NOTE: traffic_density and random_traffic NOT SET (use env default 0.06)
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC/IQL/Lidar on hard scenarios")
    parser.add_argument("--model", type=str, default="image", choices=["image", "lidar"],
                        help="Model type: 'image' for BC/IQL, 'lidar' for PPO expert")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to BC/IQL checkpoint (required for image model)")
    parser.add_argument("--num_envs", type=int, default=30,
                        help="Number of parallel environments")
    parser.add_argument("--num_seeds", type=int, default=200,
                        help="Number of seeds to evaluate")
    parser.add_argument("--use_test_seeds", action="store_true",
                        help="Use test seeds [1000,2000) instead of train seeds [0,1000)")
    parser.add_argument("--daytime", type=str, default="08:30",
                        help="Daytime setting")
    parser.add_argument("--use_original_config", action="store_true",
                        help="Use original env config from run 4v3utz7d (crash penalties, etc.)")
    parser.add_argument("--wandb", action="store_true",
                        help="Log to wandb")
    parser.add_argument("--wandb_project", type=str, default="bc-hard-seeds",
                        help="Wandb project")
    parser.add_argument("--wandb_team", type=str, default="victorique",
                        help="Wandb team")
    parser.add_argument("--exp_name", type=str, default="bc-eval",
                        help="Experiment name")
    parser.add_argument("--output_dir", type=str, default="/data/caihy/bc_eval",
                        help="Output directory")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential evaluation (slower but more stable)")
    args = parser.parse_args()
    
    print("=" * 80)
    if args.model == "lidar":
        print("Lidar PPO Expert Evaluation on Hard Scenarios")
    else:
        print("BC/IQL Checkpoint Evaluation on Hard Scenarios")
    print("=" * 80)
    print(f"Model type: {args.model}")
    if args.model == "image":
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Num envs: {args.num_envs}")
    print(f"Num seeds: {args.num_seeds}")
    print(f"Daytime: {args.daytime}")
    print(f"Initial memory: {get_memory_mb():.1f} MB")
    print("=" * 80)
    
    # Check checkpoint exists (only for image model)
    if args.model == "image":
        if not args.checkpoint:
            print(f"ERROR: --checkpoint is required for image model")
            sys.exit(1)
        if not os.path.exists(args.checkpoint):
            print(f"ERROR: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
    
    # Seeds to evaluate
    if args.use_test_seeds:
        seeds = TOP_200_EVAL_SEEDS[:args.num_seeds]
        seed_type = "TEST (TOP 200 in [1000, 2000))"
    else:
        seeds = TOP_200_TRAIN_SEEDS[:args.num_seeds]
        seed_type = "TRAIN (TOP 200 in [0, 1000))"
    print(f"Evaluating on {len(seeds)} seeds - {seed_type}")
    print(f"First 10 seeds: {seeds[:10]}")
    
    # Create env config based on model type
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    min_seed = min(seeds)
    max_seed = max(seeds)
    num_scenarios = max_seed - min_seed + 100
    
    if args.model == "lidar":
        # Lidar expert uses state-based observation
        env_config = make_lidar_env_config(daytime=args.daytime, use_test_seeds=args.use_test_seeds)
        print(f"Environment config (LIDAR): daytime={args.daytime}, use_test_seeds={args.use_test_seeds}")
        print(f"  image_observation=False (lidar mode)")
        
        # Load lidar expert
        print(f"\nLoading Lidar PPO Expert...")
        model = load_lidar_expert(device="auto")
        model_name = "lidar_expert"
    else:
        # Image-based BC/IQL
        env_config = make_env_config(daytime=args.daytime, use_original_config=args.use_original_config, use_test_seeds=args.use_test_seeds)
        print(f"Environment config (IMAGE): daytime={args.daytime}, use_original_config={args.use_original_config}, use_test_seeds={args.use_test_seeds}")
        
        # Create dummy env for model loading
        dummy_config = env_config.copy()
        dummy_config["start_seed"] = min_seed
        dummy_config["num_scenarios"] = num_scenarios
        dummy_env = HumanInTheLoopEnv(config=dummy_config)
        
        # Load BC/IQL model
        print(f"\nLoading BC/IQL model from {args.checkpoint}...")
        model = load_bc_model(args.checkpoint, dummy_env)
        dummy_env.close()
        model_name = Path(args.checkpoint).stem
    
    print(f"Model loaded! Device: {model.device}")
    
    # Create a temporary env to read ACTUAL config values after initialization
    print(f"\nReading actual env config from initialized environment...")
    temp_config = env_config.copy()
    temp_config["start_seed"] = min_seed
    temp_config["num_scenarios"] = num_scenarios
    temp_env = HumanInTheLoopEnv(config=temp_config)
    
    # Read ACTUAL values from the initialized environment
    actual_env_config = temp_env.config
    actual_traffic_density = actual_env_config.get('traffic_density', 'UNKNOWN')
    actual_random_traffic = actual_env_config.get('random_traffic', 'UNKNOWN')
    actual_out_of_road_penalty = actual_env_config.get('out_of_road_penalty', 'UNKNOWN')
    actual_crash_vehicle_penalty = actual_env_config.get('crash_vehicle_penalty', 'UNKNOWN')
    actual_crash_object_penalty = actual_env_config.get('crash_object_penalty', 'UNKNOWN')
    actual_driving_reward = actual_env_config.get('driving_reward', 'UNKNOWN')
    actual_speed_reward = actual_env_config.get('speed_reward', 'UNKNOWN')
    actual_use_lateral_reward = actual_env_config.get('use_lateral_reward', 'UNKNOWN')
    actual_horizon = actual_env_config.get('horizon', 'UNKNOWN')
    actual_daytime = actual_env_config.get('daytime', 'UNKNOWN')
    actual_image_observation = actual_env_config.get('image_observation', 'UNKNOWN')
    
    print(f"  ACTUAL traffic_density: {actual_traffic_density}")
    print(f"  ACTUAL random_traffic: {actual_random_traffic}")
    print(f"  ACTUAL out_of_road_penalty: {actual_out_of_road_penalty}")
    print(f"  ACTUAL crash_vehicle_penalty: {actual_crash_vehicle_penalty}")
    print(f"  ACTUAL crash_object_penalty: {actual_crash_object_penalty}")
    print(f"  ACTUAL driving_reward: {actual_driving_reward}")
    print(f"  ACTUAL speed_reward: {actual_speed_reward}")
    print(f"  ACTUAL horizon: {actual_horizon}")
    print(f"  ACTUAL daytime: {actual_daytime}")
    
    temp_env.close()
    
    # Build comprehensive env config for logging
    # Determine seed type and range for clear logging
    seed_type = "TEST" if args.use_test_seeds else "TRAIN"
    seed_range = "[1000, 2000)" if args.use_test_seeds else "[0, 1000)"
    
    # Create a hash of seeds for easy comparison between runs
    import hashlib
    seeds_str = ','.join(map(str, seeds))
    seeds_hash = hashlib.md5(seeds_str.encode()).hexdigest()[:8]
    
    # Seeds at key positions for visual verification
    seeds_at_50_60 = seeds[50:60] if len(seeds) > 60 else seeds[50:] if len(seeds) > 50 else []
    
    env_config_log = {
        # === SEED CONFIGURATION (CRITICAL FOR VERIFICATION) ===
        'config/seed_type': seed_type,
        'config/seed_range': seed_range,
        'config/use_test_seeds': args.use_test_seeds,
        'config/num_seeds_evaluated': len(seeds),
        'config/seeds_hash': seeds_hash,  # Quick comparison: same hash = same seeds
        'config/seeds_first_5': str(seeds[:5]),
        'config/seeds_at_50_60': str(seeds_at_50_60),  # Key position for bug detection
        'config/seeds_last_5': str(seeds[-5:]),
        'config/seeds_min': int(min(seeds)),
        'config/seeds_max': int(max(seeds)),
        'config/seeds_full_list': seeds_str,  # Full list for detailed comparison
        # === ACTUAL ENVIRONMENT CONFIGURATION (read from initialized env) ===
        'env/model_type': args.model,
        'env/use_original_config': args.use_original_config,
        'env/image_observation_ACTUAL': actual_image_observation,
        'env/traffic_density_ACTUAL': actual_traffic_density,
        'env/random_traffic_ACTUAL': actual_random_traffic,
        'env/daytime_ACTUAL': actual_daytime,
        'env/crash_vehicle_done': env_config.get('crash_vehicle_done'),
        'env/crash_object_done': env_config.get('crash_object_done'),
        'env/cost_to_reward': env_config.get('cost_to_reward'),
        'env/crash_vehicle_penalty_ACTUAL': actual_crash_vehicle_penalty,
        'env/crash_object_penalty_ACTUAL': actual_crash_object_penalty,
        'env/out_of_road_penalty_ACTUAL': actual_out_of_road_penalty,
        'env/driving_reward_ACTUAL': actual_driving_reward,
        'env/speed_reward_ACTUAL': actual_speed_reward,
        'env/use_lateral_reward_ACTUAL': actual_use_lateral_reward,
        'env/horizon_ACTUAL': actual_horizon,
        'env/start_seed': env_config.get('start_seed'),
        'env/num_scenarios': env_config.get('num_scenarios'),
        'env/stack_size': env_config.get('stack_size'),
    }
    
    print(f"\nEnvironment config (FULL):")
    for key, val in env_config_log.items():
        print(f"    {key}={val}")
    
    # Initialize wandb
    wandb_run = None
    if args.wandb:
        print("\nInitializing wandb...")
        try:
            import wandb as wandb_module
            from pvp.utils.utils import get_time_str
            import uuid
            trial_name = f"{args.exp_name}_{get_time_str()}_{uuid.uuid4().hex[:8]}"
            # Merge args and env_config for wandb config
            wandb_config = {**vars(args), **env_config_log}
            wandb_run = wandb_module.init(
                project=args.wandb_project,
                entity=args.wandb_team,
                name=trial_name,
                config=wandb_config,
            )
            print(f"Wandb initialized: {trial_name}")
            
            # Also log env config as metrics so they appear in charts (not just Overview)
            # This makes it easier to compare across runs without clicking into each one
            env_config_metrics = {}
            for key, val in env_config_log.items():
                # Convert to numeric where possible for chart display
                metric_key = key.replace('/', '_')  # wandb metrics can't have /
                if isinstance(val, (int, float, bool)):
                    env_config_metrics[f"env_cfg/{metric_key}"] = float(val) if isinstance(val, bool) else val
                elif isinstance(val, str):
                    # For string values, try to extract numeric part or hash
                    if val.replace('.', '').replace('-', '').isdigit():
                        env_config_metrics[f"env_cfg/{metric_key}"] = float(val)
                    elif key == 'config/seeds_hash':
                        # Convert hash to int for comparison
                        env_config_metrics[f"env_cfg/{metric_key}"] = int(val, 16) % 1000000
            
            # Log key numeric configs as step=0 metrics
            wandb_module.log({
                "env_cfg/traffic_density": actual_env_config.get('traffic_density', 0.06),
                "env_cfg/out_of_road_penalty": actual_env_config.get('out_of_road_penalty', 5.0),
                "env_cfg/crash_vehicle_penalty": actual_env_config.get('crash_vehicle_penalty', 5.0),
                "env_cfg/crash_object_penalty": actual_env_config.get('crash_object_penalty', 5.0),
                "env_cfg/driving_reward": actual_env_config.get('driving_reward', 1.0),
                "env_cfg/speed_reward": actual_env_config.get('speed_reward', 0.1),
                "env_cfg/horizon": actual_env_config.get('horizon', 1500),
                "env_cfg/num_seeds": len(seeds),
                "env_cfg/seeds_hash_numeric": int(seeds_hash, 16) % 1000000,  # For quick comparison
                "env_cfg/seeds_min": int(min(seeds)),
                "env_cfg/seeds_max": int(max(seeds)),
                "env_cfg/use_test_seeds": 1 if args.use_test_seeds else 0,
            }, step=0)
            print(f"    Env config logged as metrics (env_cfg/*)")
            
        except Exception as e:
            print(f"WARNING: Failed to init wandb: {e}")
    
    # Run evaluation
    start_time = time.time()
    
    if args.sequential or args.num_envs == 1:
        print(f"\nStarting sequential evaluation...")
        results = evaluate_sequential(model, seeds, env_config)
    else:
        print(f"\nStarting parallel evaluation with {args.num_envs} envs...")
        results = evaluate_parallel(model, seeds, args.num_envs, env_config)
    
    eval_time = time.time() - start_time
    print(f"\nEvaluation completed in {eval_time:.1f}s ({eval_time/60:.1f} min)")
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  Reward: mean={metrics['eval/reward_mean']:.1f}, std={metrics['eval/reward_std']:.1f}")
    print(f"  Route Completion: {metrics['eval/route_completion']*100:.1f}%")
    print(f"  Route Completion (no bad): {metrics['eval/route_completion_no_bad']*100:.1f}%")
    print(f"  Success Rate: {metrics['eval/success_rate']*100:.1f}%")
    print(f"  Success Rate (no bad): {metrics['eval/success_rate_no_bad']*100:.1f}%")
    print(f"  --- Safety Metrics ---")
    print(f"  Crash Vehicle Rate: {metrics['eval/crash_vehicle_rate']*100:.1f}%")
    print(f"  Crash Object Rate: {metrics['eval/crash_object_rate']*100:.1f}%")
    print(f"  Out of Road Rate: {metrics['eval/out_of_road_rate']*100:.1f}%")
    print(f"  Any Bad Event Rate: {metrics['eval/any_bad_event_rate']*100:.1f}%")
    print(f"  --- Behavioral Metrics ---")
    print(f"  Avg Speed: {metrics['eval/avg_speed']:.2f} m/s")
    print(f"  Speed Std: {metrics['eval/speed_std']:.2f}")
    print(f"  Steering Std: {metrics['eval/steering_std']:.4f}")
    print(f"  Accel Std: {metrics['eval/accel_std']:.4f}")
    print(f"  Hard Brake Ratio: {metrics['eval/hard_brake_ratio']*100:.1f}%")
    print(f"  Hard Steer Ratio: {metrics['eval/hard_steer_ratio']*100:.1f}%")
    print(f"  --- Traffic Proximity ---")
    print(f"  Min Vehicle Distance: {metrics['eval/min_vehicle_distance']:.2f}m")
    print(f"  Avg Close Encounters: {metrics['eval/total_close_encounters']:.1f}")
    print(f"  Close Encounter Rate: {metrics['eval/close_encounter_rate']*100:.1f}%")
    print(f"  Safe Pass Rate: {metrics['eval/safe_pass_rate']*100:.1f}%")
    print(f"  Close Encounter Crash Rate: {metrics['eval/close_encounter_crash_rate']*100:.1f}%")
    print(f"  --- Difficulty Segments ---")
    for seg in ["hardest", "hard", "medium", "easy", "easiest"]:
        if f'eval/segment_{seg}_success' in metrics:
            print(f"  [{seg:8s}] Success: {metrics[f'eval/segment_{seg}_success']*100:5.1f}%, "
                  f"Crash: {metrics[f'eval/segment_{seg}_crash']*100:5.1f}%, "
                  f"Reward: {metrics[f'eval/segment_{seg}_reward']:.1f}")
    
    # Log all metrics to wandb
    if wandb_run:
        print(f"\nLogging {len(metrics)} metrics to wandb...")
        wandb_run.log(metrics)
        
        # Also log summary table
        wandb_run.summary.update(metrics)
        wandb_run.finish()
        print("Wandb finished!")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_name = Path(args.checkpoint).stem
    result_file = output_dir / f"eval_{ckpt_name}.json"
    with open(result_file, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'seeds': seeds,
            'metrics': metrics,
            'per_seed_results': results,
            'eval_time_seconds': eval_time,
        }, f, indent=2)
    print(f"\nResults saved to: {result_file}")
    
    print(f"\nFinal memory: {get_memory_mb():.1f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
