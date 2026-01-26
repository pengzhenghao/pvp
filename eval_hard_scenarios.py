"""
Evaluate IQL and TD3 models on the top 200 hardest scenarios.
Compares performance metrics between models on each scenario seed.

Usage:
    python eval_hard_scenarios.py --model iql
    python eval_hard_scenarios.py --model td3
    python eval_hard_scenarios.py --model both
"""

import argparse
import os
import json
import numpy as np
import sys
import gymnasium
import gymnasium.spaces
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
gymnasium.spaces.space = gymnasium.spaces

from pathlib import Path
from tqdm import tqdm

# Suppress metadrive logging
import logging
logging.getLogger('metadrive').setLevel(logging.WARNING)
logging.getLogger('panda3d').setLevel(logging.WARNING)


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


def make_env_config(seed, use_image=True, use_render=False):
    """Create environment config for a specific seed."""
    from metadrive.component.sensors.rgb_camera import RGBCamera
    sensor_size = (84, 84)
    
    config = dict(
        use_render=use_render,
        manual_control=False,
        start_seed=seed,
        num_scenarios=1,  # Only this seed (for single-seed mode)
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
            daytime="06:10",
        ))
    else:
        config['image_observation'] = False
    
    return config


def make_shared_env_config(use_image=True, use_render=False, start_seed=1000, num_scenarios=1000):
    """Create config for a shared environment that supports multiple seeds via reset(seed=...)."""
    from metadrive.component.sensors.rgb_camera import RGBCamera
    sensor_size = (84, 84)
    
    config = dict(
        use_render=use_render,
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
            daytime="06:10",
        ))
    else:
        config['image_observation'] = False
    
    return config


def load_iql_model(checkpoint_path, env):
    """Load IQL model from checkpoint."""
    from pvp.sb3.td3.iql import IQL
    from pvp.sb3.td3.policies import TD3Policy
    from pvp.sb3.sac.our_features_extractor import OurFeaturesExtractorCNN as OurFeaturesExtractor
    from pvp.sb3.common.save_util import load_from_zip_file
    
    policy_kwargs = dict(
        features_extractor_class=OurFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=275),
        share_features_extractor=False,
        net_arch=[256,]
    )
    
    model = IQL(
        policy=TD3Policy,
        env=env,
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


def load_td3_model(checkpoint_path, env):
    """Load TD3 model from checkpoint."""
    from pvp.sb3.td3.td3 import TD3
    from pvp.sb3.td3.policies import TD3Policy
    from pvp.sb3.sac.our_features_extractor import OurFeaturesExtractorCNN as OurFeaturesExtractor
    from pvp.sb3.common.save_util import load_from_zip_file
    
    policy_kwargs = dict(
        features_extractor_class=OurFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=275),
        share_features_extractor=False,
        net_arch=[256,]
    )
    
    model = TD3(
        policy=TD3Policy,
        env=env,
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


def load_lidar_expert():
    """Load the lidar-based PPO expert for comparison."""
    from pvp.sb3.ppo import PPO
    from pvp.sb3.ppo.policies import ActorCriticPolicy
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    from pvp.sb3.common.save_util import load_from_zip_file
    
    # Create temp env for PPO
    temp_env = HumanInTheLoopEnv(config={'manual_control': False, 'use_render': False})
    
    expert = PPO(
        policy=ActorCriticPolicy,
        env=temp_env,
        n_steps=1024,
        verbose=0,
        device="auto",
    )
    
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    ckpt = script_dir / "pvp" / "experiments" / "metadrive" / "egpo" / "metadrive_pvp_20m_steps.zip"
    
    if ckpt.exists():
        data, params, pytorch_variables = load_from_zip_file(ckpt, device=expert.device, print_system_info=False)
        expert.set_parameters(params, exact_match=True, device=expert.device)
        print(f"Expert loaded from {ckpt}")
    else:
        print(f"WARNING: Expert checkpoint not found at {ckpt}")
    
    temp_env.close()
    return expert


def evaluate_on_seed(model, seed, num_episodes=1, use_render=False, expert_model=None, shared_env=None):
    """Evaluate model on a specific scenario seed with comprehensive metrics.
    
    Args:
        model: The model to evaluate
        seed: The scenario seed
        num_episodes: Number of episodes to run
        use_render: Whether to render (ignored if shared_env provided)
        expert_model: Optional expert model for comparison
        shared_env: Optional shared environment (faster - uses reset(seed=seed) instead of creating new env)
    """
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    
    # Use shared environment if provided (much faster - avoids creating new env for each seed)
    if shared_env is not None:
        env = shared_env
        owns_env = False
    else:
        env = HumanInTheLoopEnv(config=make_env_config(seed, use_image=True, use_render=use_render))
        owns_env = True
    
    # For expert comparison, we'll get expert action from the same observation
    # No need for separate lidar env - expert can predict from lidar obs in info
    lidar_env = None  # Disabled to avoid multi-env issues
    
    results = {
        'rewards': [],
        'lengths': [],
        'route_completions': [],
        'crash_vehicles': [],
        'crash_objects': [],
        'out_of_roads': [],
        'arrive_dests': [],
        # New metrics
        'route_completion_no_bad_event': [],  # Route completion weighted by no bad events
        'any_bad_event': [],
        'success_no_bad_event': [],
        'crash_count': [],
        # Action statistics
        'mean_steering_abs': [],
        'mean_accel_abs': [],
        'hard_brake_ratio': [],
        'hard_steer_ratio': [],
        # Expert comparison
        'expert_action_diff_l2': [],
        'expert_steering_diff': [],
        'expert_accel_diff': [],
        'expert_agreement_ratio': [],
        # Traffic proximity metrics (NEW)
        'min_vehicle_distance': [],          # Minimum distance to any vehicle during episode
        'total_close_encounters': [],        # Total times within 15m of another vehicle
        'close_encounter_rate': [],          # Close encounters per step
        'safe_pass_count': [],               # Times passed close (<15m) without crash
        'dangerous_close_count': [],         # Times close (<15m) AND crashed
        'close_encounter_crash_rate': [],    # Crash rate during close encounters
    }
    
    for ep in range(num_episodes):
        # Use seed parameter when using shared env (supports fast scenario switching)
        obs = env.reset(seed=seed) if shared_env is not None else env.reset()
        
        done = False
        episode_reward = 0
        episode_length = 0
        episode_actions = []
        episode_speeds = []
        episode_expert_diffs = []
        episode_steering_diffs = []
        episode_accel_diffs = []
        crash_count = 0
        had_bad_event = False
        # Cumulative flags for each type of bad event
        had_crash_vehicle = False
        had_crash_object = False
        had_out_of_road = False
        # Traffic proximity tracking (NEW)
        min_vehicle_distance = float('inf')
        total_close_encounters = 0
        safe_pass_count = 0        # Close (<15m) but no crash at that moment
        dangerous_close_count = 0  # Close (<15m) AND crashed at that moment
        steps_with_traffic = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            episode_actions.append(action)
            obs, reward, done, info = env.step(action)
            
            # Get expert action for comparison using lidar_obs from info
            if expert_model is not None and 'lidar_obs' in info:
                lidar_obs = info['lidar_obs']
                expert_action, _ = expert_model.predict(lidar_obs, deterministic=True)
                # Compute action difference
                action_diff = np.linalg.norm(action - expert_action)
                steering_diff = abs(action[0] - expert_action[0])
                accel_diff = abs(action[1] - expert_action[1])
                episode_expert_diffs.append(action_diff)
                episode_steering_diffs.append(steering_diff)
                episode_accel_diffs.append(accel_diff)
            
            episode_reward += reward
            episode_length += 1
            
            # Track crashes per step (cumulative across entire episode)
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
            
            # Track speed
            episode_speeds.append(info.get('velocity', 0))
            
            # Track traffic proximity (NEW)
            vehicle_dist = info.get('min_vehicle_distance', -1)
            if vehicle_dist > 0 and vehicle_dist < min_vehicle_distance:
                min_vehicle_distance = vehicle_dist
            
            # Count steps with nearby traffic (< 20m)
            if vehicle_dist > 0 and vehicle_dist < 20:
                steps_with_traffic += 1
            
            close_count = info.get('close_vehicle_count', 0)
            if close_count > 0:
                total_close_encounters += close_count
                if step_crash:
                    dangerous_close_count += 1  # Crashed while close to vehicle
                else:
                    safe_pass_count += 1  # Close but didn't crash
        
        # Episode-level metrics
        route_completion = info.get('route_completion', 0)
        arrive_dest = info.get('arrive_dest', False)
        
        results['rewards'].append(episode_reward)
        results['lengths'].append(episode_length)
        results['route_completions'].append(route_completion)
        # FIXED: Use cumulative flags tracked during episode, not final step values
        results['crash_vehicles'].append(had_crash_vehicle)
        results['crash_objects'].append(had_crash_object)
        results['out_of_roads'].append(had_out_of_road)
        results['arrive_dests'].append(arrive_dest)
        
        # New metrics
        results['any_bad_event'].append(had_bad_event)
        results['success_no_bad_event'].append(arrive_dest and not had_bad_event)
        results['crash_count'].append(crash_count)
        
        # Route completion weighted by no bad events (key metric!)
        # If had bad event, penalize route completion
        rc_no_bad = route_completion if not had_bad_event else route_completion * 0.5
        results['route_completion_no_bad_event'].append(rc_no_bad)
        
        # Action statistics
        episode_actions = np.array(episode_actions)
        results['mean_steering_abs'].append(np.mean(np.abs(episode_actions[:, 0])))
        results['mean_accel_abs'].append(np.mean(np.abs(episode_actions[:, 1])))
        results['hard_brake_ratio'].append(np.mean(episode_actions[:, 1] < -0.5))
        results['hard_steer_ratio'].append(np.mean(np.abs(episode_actions[:, 0]) > 0.5))
        
        # Additional behavioral metrics for consistency with expert evaluation
        if 'steering_std' not in results:
            results['steering_std'] = []
            results['accel_std'] = []
            results['avg_speed'] = []
            results['speed_std'] = []
            results['steps_with_traffic'] = []
            results['traffic_density_rate'] = []
        results['steering_std'].append(float(np.std(episode_actions[:, 0])) if len(episode_actions) > 0 else 0.0)
        results['accel_std'].append(float(np.std(episode_actions[:, 1])) if len(episode_actions) > 0 else 0.0)
        speeds_arr = np.array(episode_speeds)
        results['avg_speed'].append(float(np.mean(speeds_arr)) if len(speeds_arr) > 0 else 0.0)
        results['speed_std'].append(float(np.std(speeds_arr)) if len(speeds_arr) > 0 else 0.0)
        results['steps_with_traffic'].append(steps_with_traffic)
        traffic_density_rate = steps_with_traffic / max(episode_length, 1)
        results['traffic_density_rate'].append(traffic_density_rate)
        
        # Expert comparison metrics
        if len(episode_expert_diffs) > 0:
            results['expert_action_diff_l2'].append(np.mean(episode_expert_diffs))
            results['expert_steering_diff'].append(np.mean(episode_steering_diffs))
            results['expert_accel_diff'].append(np.mean(episode_accel_diffs))
            results['expert_agreement_ratio'].append(np.mean(np.array(episode_expert_diffs) < 0.3))
        
        # Traffic proximity metrics (NEW)
        results['min_vehicle_distance'].append(float(min_vehicle_distance) if min_vehicle_distance != float('inf') else -1)
        results['total_close_encounters'].append(total_close_encounters)
        close_encounter_rate = total_close_encounters / max(episode_length, 1)
        results['close_encounter_rate'].append(close_encounter_rate)
        results['safe_pass_count'].append(safe_pass_count)
        results['dangerous_close_count'].append(dangerous_close_count)
        # Crash rate during close encounters: how often we crash when close to other vehicles
        total_close_steps = safe_pass_count + dangerous_close_count
        close_encounter_crash_rate = dangerous_close_count / max(total_close_steps, 1) if total_close_steps > 0 else 0
        results['close_encounter_crash_rate'].append(close_encounter_crash_rate)
    
    # Only close env if we created it (not shared)
    if owns_env:
        env.close()
    
    # Compute summary statistics
    summary = {
        'seed': seed,
        'reward': float(np.mean(results['rewards'])),
        'reward_std': float(np.std(results['rewards'])),
        'length': float(np.mean(results['lengths'])),
        'route_completion': float(np.mean(results['route_completions'])),
        'route_completion_no_bad_event': float(np.mean(results['route_completion_no_bad_event'])),
        'crash_vehicle_rate': float(np.mean(results['crash_vehicles'])),
        'crash_object_rate': float(np.mean(results['crash_objects'])),
        'out_of_road_rate': float(np.mean(results['out_of_roads'])),
        'any_bad_event_rate': float(np.mean(results['any_bad_event'])),
        'success_rate': float(np.mean(results['arrive_dests'])),
        'success_no_bad_event_rate': float(np.mean(results['success_no_bad_event'])),
        'crash_count_mean': float(np.mean(results['crash_count'])),
        'mean_steering_abs': float(np.mean(results['mean_steering_abs'])),
        'mean_accel_abs': float(np.mean(results['mean_accel_abs'])),
        'hard_brake_ratio': float(np.mean(results['hard_brake_ratio'])),
        'hard_steer_ratio': float(np.mean(results['hard_steer_ratio'])),
    }
    
    # Add expert comparison metrics if available
    if len(results['expert_action_diff_l2']) > 0:
        summary['expert_action_diff_l2'] = float(np.mean(results['expert_action_diff_l2']))
        summary['expert_steering_diff'] = float(np.mean(results['expert_steering_diff']))
        summary['expert_accel_diff'] = float(np.mean(results['expert_accel_diff']))
        summary['expert_agreement_ratio'] = float(np.mean(results['expert_agreement_ratio']))
    
    # Add traffic proximity metrics (NEW)
    valid_distances = [d for d in results['min_vehicle_distance'] if d > 0]
    summary['min_vehicle_distance'] = float(np.min(valid_distances)) if valid_distances else -1
    summary['avg_min_vehicle_distance'] = float(np.mean(valid_distances)) if valid_distances else -1
    summary['total_close_encounters'] = float(np.mean(results['total_close_encounters']))
    summary['close_encounter_rate'] = float(np.mean(results['close_encounter_rate']))
    summary['safe_pass_count'] = float(np.mean(results['safe_pass_count']))
    summary['dangerous_close_count'] = float(np.mean(results['dangerous_close_count']))
    summary['close_encounter_crash_rate'] = float(np.mean(results['close_encounter_crash_rate']))
    
    # Additional behavioral metrics
    if 'steering_std' in results and len(results['steering_std']) > 0:
        summary['steering_std'] = float(np.mean(results['steering_std']))
        summary['accel_std'] = float(np.mean(results['accel_std']))
        summary['avg_speed'] = float(np.mean(results['avg_speed']))
        summary['speed_std'] = float(np.mean(results['speed_std']))
        summary['steps_with_traffic'] = float(np.mean(results['steps_with_traffic']))
        summary['traffic_density_rate'] = float(np.mean(results['traffic_density_rate']))
    
    return summary


def evaluate_expert_on_seed(expert_model, seed, num_episodes=1, shared_env=None):
    """Evaluate the lidar expert directly on a seed using lidar observation.
    
    Args:
        expert_model: The lidar PPO expert
        seed: The scenario seed
        num_episodes: Number of episodes
        shared_env: Shared environment (must be lidar-based)
    """
    import numpy as np
    env = shared_env
    
    results = {
        'rewards': [],
        'lengths': [],
        'route_completions': [],
        'crash_vehicles': [],
        'crash_objects': [],
        'out_of_roads': [],
        'arrive_dests': [],
        'any_bad_event': [],
        'success_no_bad_event': [],
        'crash_count': [],
        # Traffic proximity metrics
        'min_vehicle_distance': [],
        'total_close_encounters': [],
        'close_encounter_rate': [],
        'safe_pass_count': [],
        'dangerous_close_count': [],
        'close_encounter_crash_rate': [],
        # Behavioral metrics
        'mean_steering_abs': [],
        'mean_accel_abs': [],
        'steering_std': [],
        'accel_std': [],
        'hard_brake_ratio': [],
        'hard_steer_ratio': [],
        'avg_speed': [],
        'speed_std': [],
        # Traffic density metrics
        'steps_with_traffic': [],
        'traffic_density_rate': [],
    }
    
    for ep in range(num_episodes):
        obs = env.reset(seed=seed)
        done = False
        episode_reward = 0
        episode_length = 0
        crash_count = 0
        had_bad_event = False
        had_crash_vehicle = False
        had_crash_object = False
        had_out_of_road = False
        min_vehicle_distance = float('inf')
        total_close_encounters = 0
        safe_pass_count = 0
        dangerous_close_count = 0
        
        # Action tracking
        actions_list = []
        speeds_list = []
        steps_with_traffic = 0
        
        while not done:
            action, _ = expert_model.predict(obs, deterministic=False)  # Stochastic for expert
            actions_list.append(action.copy())
            
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Track speed
            speeds_list.append(info.get('velocity', 0))
            
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
            
            vehicle_dist = info.get('min_vehicle_distance', -1)
            if vehicle_dist > 0 and vehicle_dist < min_vehicle_distance:
                min_vehicle_distance = vehicle_dist
            
            # Count steps with nearby traffic (< 20m)
            if vehicle_dist > 0 and vehicle_dist < 20:
                steps_with_traffic += 1
            
            close_count = info.get('close_vehicle_count', 0)
            if close_count > 0:
                total_close_encounters += close_count
                if step_crash:
                    dangerous_close_count += 1
                else:
                    safe_pass_count += 1
        
        route_completion = info.get('route_completion', 0)
        arrive_dest = info.get('arrive_dest', False)
        
        # Compute action statistics
        actions_arr = np.array(actions_list)
        speeds_arr = np.array(speeds_list)
        
        results['rewards'].append(episode_reward)
        results['lengths'].append(episode_length)
        results['route_completions'].append(route_completion)
        results['crash_vehicles'].append(had_crash_vehicle)
        results['crash_objects'].append(had_crash_object)
        results['out_of_roads'].append(had_out_of_road)
        results['arrive_dests'].append(arrive_dest)
        results['any_bad_event'].append(had_bad_event)
        results['success_no_bad_event'].append(arrive_dest and not had_bad_event)
        results['crash_count'].append(crash_count)
        results['min_vehicle_distance'].append(float(min_vehicle_distance) if min_vehicle_distance != float('inf') else -1)
        results['total_close_encounters'].append(total_close_encounters)
        close_encounter_rate = total_close_encounters / max(episode_length, 1)
        results['close_encounter_rate'].append(close_encounter_rate)
        results['safe_pass_count'].append(safe_pass_count)
        
        # Behavioral metrics
        results['mean_steering_abs'].append(float(np.mean(np.abs(actions_arr[:, 0]))) if len(actions_arr) > 0 else 0.0)
        results['mean_accel_abs'].append(float(np.mean(np.abs(actions_arr[:, 1]))) if len(actions_arr) > 0 else 0.0)
        results['steering_std'].append(float(np.std(actions_arr[:, 0])) if len(actions_arr) > 0 else 0.0)
        results['accel_std'].append(float(np.std(actions_arr[:, 1])) if len(actions_arr) > 0 else 0.0)
        results['hard_brake_ratio'].append(float(np.mean(actions_arr[:, 1] < -0.5)) if len(actions_arr) > 0 else 0.0)
        results['hard_steer_ratio'].append(float(np.mean(np.abs(actions_arr[:, 0]) > 0.5)) if len(actions_arr) > 0 else 0.0)
        results['avg_speed'].append(float(np.mean(speeds_arr)) if len(speeds_arr) > 0 else 0.0)
        results['speed_std'].append(float(np.std(speeds_arr)) if len(speeds_arr) > 0 else 0.0)
        
        # Traffic density metrics
        results['steps_with_traffic'].append(steps_with_traffic)
        traffic_density_rate = steps_with_traffic / max(episode_length, 1)
        results['traffic_density_rate'].append(traffic_density_rate)
        results['dangerous_close_count'].append(dangerous_close_count)
        total_close_steps = safe_pass_count + dangerous_close_count
        close_encounter_crash_rate = dangerous_close_count / max(total_close_steps, 1) if total_close_steps > 0 else 0
        results['close_encounter_crash_rate'].append(close_encounter_crash_rate)
    
    # Compute summary statistics
    summary = {
        'seed': seed,
        'reward': float(np.mean(results['rewards'])),
        'reward_std': float(np.std(results['rewards'])),
        'length': float(np.mean(results['lengths'])),
        'route_completion': float(np.mean(results['route_completions'])),
        'crash_vehicle_rate': float(np.mean(results['crash_vehicles'])),
        'crash_object_rate': float(np.mean(results['crash_objects'])),
        'out_of_road_rate': float(np.mean(results['out_of_roads'])),
        'any_bad_event_rate': float(np.mean(results['any_bad_event'])),
        'success_rate': float(np.mean(results['arrive_dests'])),
        'success_no_bad_event_rate': float(np.mean(results['success_no_bad_event'])),
        'crash_count_mean': float(np.mean(results['crash_count'])),
    }
    
    # Traffic proximity metrics
    valid_distances = [d for d in results['min_vehicle_distance'] if d > 0]
    summary['min_vehicle_distance'] = float(np.min(valid_distances)) if valid_distances else -1
    summary['avg_min_vehicle_distance'] = float(np.mean(valid_distances)) if valid_distances else -1
    summary['total_close_encounters'] = float(np.mean(results['total_close_encounters']))
    summary['close_encounter_rate'] = float(np.mean(results['close_encounter_rate']))
    summary['safe_pass_count'] = float(np.mean(results['safe_pass_count']))
    summary['dangerous_close_count'] = float(np.mean(results['dangerous_close_count']))
    summary['close_encounter_crash_rate'] = float(np.mean(results['close_encounter_crash_rate']))
    
    # Behavioral metrics
    summary['mean_steering_abs'] = float(np.mean(results['mean_steering_abs']))
    summary['mean_accel_abs'] = float(np.mean(results['mean_accel_abs']))
    summary['steering_std'] = float(np.mean(results['steering_std']))
    summary['accel_std'] = float(np.mean(results['accel_std']))
    summary['hard_brake_ratio'] = float(np.mean(results['hard_brake_ratio']))
    summary['hard_steer_ratio'] = float(np.mean(results['hard_steer_ratio']))
    summary['avg_speed'] = float(np.mean(results['avg_speed']))
    summary['speed_std'] = float(np.mean(results['speed_std']))
    
    # Traffic density metrics
    summary['steps_with_traffic'] = float(np.mean(results['steps_with_traffic']))
    summary['traffic_density_rate'] = float(np.mean(results['traffic_density_rate']))
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on hard scenarios")
    parser.add_argument("--model", type=str, required=True, choices=["iql", "td3", "td3bc2", "both", "all", "custom", "expert", "compare_expert"],
                        help="Which model to evaluate: iql, td3, td3bc2, both (iql+td3), all (iql+td3+td3bc2), custom, expert (lidar PPO), compare_expert (custom vs expert)")
    parser.add_argument("--custom_checkpoint", type=str, default="",
                        help="Path to custom model checkpoint (used with --model custom)")
    parser.add_argument("--custom_name", type=str, default="custom",
                        help="Name for custom model in output")
    parser.add_argument("--num_seeds", type=int, default=200,
                        help="Number of seeds to evaluate (from top 200)")
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="Number of episodes per seed")
    parser.add_argument("--output", type=str, default="./results/hard_scenario_eval",
                        help="Output directory")
    parser.add_argument("--render", action="store_true",
                        help="Enable rendering (slow)")
    parser.add_argument("--no_expert", action="store_true",
                        help="Skip expert comparison (faster)")
    parser.add_argument("--iql_checkpoint", type=str, default="IQLBEST1.zip",
                        help="IQL checkpoint path")
    parser.add_argument("--td3_checkpoint", type=str, default="TD3BCBEST1.zip",
                        help="TD3 (BC) checkpoint path")
    parser.add_argument("--td3bc2_checkpoint", type=str, default="TD3BCBEST2.zip",
                        help="TD3BC2 checkpoint path")
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    seeds = TOP_200_SEEDS[:args.num_seeds]
    print(f"Evaluating on {len(seeds)} hardest scenarios")
    
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
    
    # Create a shared environment for model initialization AND evaluation
    # This is much faster than creating a new env for each seed - uses reset(seed=seed)
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    min_seed = min(seeds)
    max_seed = max(seeds)
    num_scenarios = max_seed - min_seed + 100  # Extra buffer
    shared_env = HumanInTheLoopEnv(config=make_shared_env_config(
        use_image=True, 
        use_render=args.render,
        start_seed=min_seed,
        num_scenarios=num_scenarios
    ))
    print(f"Created shared env: start_seed={min_seed}, num_scenarios={num_scenarios}")
    
    # Load models
    models_to_eval = {}
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    if args.model in ["iql", "both", "all"]:
        iql_path = script_dir / args.iql_checkpoint
        if iql_path.exists():
            print(f"Loading IQL model from {iql_path}...")
            models_to_eval["iql"] = load_iql_model(iql_path, shared_env)
            print("IQL model loaded!")
        else:
            print(f"ERROR: IQL checkpoint not found at {iql_path}")
            return
    
    if args.model in ["td3", "both", "all"]:
        td3_path = script_dir / args.td3_checkpoint
        if td3_path.exists():
            print(f"Loading TD3BC1 model from {td3_path}...")
            models_to_eval["td3bc1"] = load_td3_model(td3_path, shared_env)
            print("TD3BC1 model loaded!")
        else:
            print(f"ERROR: TD3 checkpoint not found at {td3_path}")
            return
    
    if args.model in ["td3bc2", "all"]:
        td3bc2_path = script_dir / args.td3bc2_checkpoint
        if td3bc2_path.exists():
            print(f"Loading TD3BC2 model from {td3bc2_path}...")
            models_to_eval["td3bc2"] = load_td3_model(td3bc2_path, shared_env)
            print("TD3BC2 model loaded!")
        else:
            print(f"ERROR: TD3BC2 checkpoint not found at {td3bc2_path}")
            return
    
    if args.model == "custom":
        if not args.custom_checkpoint:
            print("ERROR: --custom_checkpoint required when using --model custom")
            return
        custom_path = Path(args.custom_checkpoint)
        if not custom_path.exists():
            custom_path = script_dir / args.custom_checkpoint
        if custom_path.exists():
            print(f"Loading custom model from {custom_path}...")
            models_to_eval[args.custom_name] = load_td3_model(custom_path, shared_env)
            print(f"Custom model '{args.custom_name}' loaded!")
        else:
            print(f"ERROR: Custom checkpoint not found at {custom_path}")
            return
    
    # Special mode: compare custom model vs expert
    if args.model == "compare_expert":
        if not args.custom_checkpoint:
            print("ERROR: --custom_checkpoint required when using --model compare_expert")
            return
        if expert_model is None:
            print("ERROR: Expert model required for compare_expert mode")
            return
        
        custom_path = Path(args.custom_checkpoint)
        if not custom_path.exists():
            custom_path = script_dir / args.custom_checkpoint
        if not custom_path.exists():
            print(f"ERROR: Custom checkpoint not found at {custom_path}")
            return
        
        print(f"Loading custom model from {custom_path}...")
        custom_model = load_td3_model(custom_path, shared_env)
        print(f"Custom model '{args.custom_name}' loaded!")
        
        # Run comparison - SEQUENTIALLY to avoid multi-env issues
        # First evaluate custom model on all seeds
        print(f"\n{'='*80}")
        print(f"Phase 1: Evaluating {args.custom_name.upper()} on {len(seeds)} scenarios")
        print(f"{'='*80}")
        
        custom_results = []
        shared_env.reset()  # Initial reset
        for i, seed in enumerate(tqdm(seeds, desc=f"{args.custom_name.upper()}")):
            result = evaluate_on_seed(
                custom_model, seed,
                num_episodes=args.num_episodes,
                use_render=False,
                expert_model=None,  # Skip expert comparison in first pass
                shared_env=shared_env
            )
            custom_results.append(result)
        
        # Close RGB env before creating lidar env
        shared_env.close()
        
        # Create lidar-based env for expert
        print(f"\n{'='*80}")
        print(f"Phase 2: Evaluating LIDAR_EXPERT on {len(seeds)} scenarios")
        print(f"{'='*80}")
        
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        lidar_config = make_shared_env_config(
            use_image=False,  # Lidar only
            use_render=False,
            start_seed=min_seed,
            num_scenarios=num_scenarios
        )
        lidar_config['num_agents'] = 1
        lidar_env = HumanInTheLoopEnv(config=lidar_config)
        lidar_env.reset()
        
        
        expert_results = []
        for i, seed in enumerate(tqdm(seeds, desc="EXPERT")):
            result = evaluate_expert_on_seed(
                expert_model, seed,
                num_episodes=args.num_episodes,
                shared_env=lidar_env
            )
            expert_results.append(result)
            
            # Print comparison every 10 seeds
            if (i + 1) % 10 == 0:
                print(f"\n{'='*60}")
                print(f"Progress: {i+1}/{len(seeds)} seeds completed")
                print(f"{'='*60}")
                
                # Compute current metrics (compare to same number of custom results)
                n = i + 1
                c_crash_rate = np.mean([r['crash_vehicle_rate'] for r in custom_results[:n]])
                e_crash_rate = np.mean([r['crash_vehicle_rate'] for r in expert_results])
                c_close_crash = np.mean([r['close_encounter_crash_rate'] for r in custom_results[:n]])
                e_close_crash = np.mean([r['close_encounter_crash_rate'] for r in expert_results])
                c_safe_pass = np.mean([r['safe_pass_count'] for r in custom_results[:n]])
                e_safe_pass = np.mean([r['safe_pass_count'] for r in expert_results])
                c_dangerous = np.mean([r['dangerous_close_count'] for r in custom_results[:n]])
                e_dangerous = np.mean([r['dangerous_close_count'] for r in expert_results])
                c_min_dist_list = [r['min_vehicle_distance'] for r in custom_results[:n] if r['min_vehicle_distance'] > 0]
                e_min_dist_list = [r['min_vehicle_distance'] for r in expert_results if r['min_vehicle_distance'] > 0]
                c_min_dist = np.mean(c_min_dist_list) if c_min_dist_list else 0
                e_min_dist = np.mean(e_min_dist_list) if e_min_dist_list else 0
                c_success = np.mean([r['success_no_bad_event_rate'] for r in custom_results[:n]])
                e_success = np.mean([r['success_no_bad_event_rate'] for r in expert_results])
                c_any_bad = np.mean([r['any_bad_event_rate'] for r in custom_results[:n]])
                e_any_bad = np.mean([r['any_bad_event_rate'] for r in expert_results])
                
                print(f"{'Metric':<35} {args.custom_name.upper():<12} EXPERT       优势谁")
                print("-" * 70)
                
                # Compare each metric
                def compare_lower_better(name, cv, ev):
                    winner = "EXPERT ✓" if ev < cv else (f"{args.custom_name.upper()} ✓" if cv < ev else "平手")
                    print(f"{name:<35} {cv*100:>10.1f}% {ev*100:>10.1f}%   {winner}")
                
                def compare_higher_better(name, cv, ev):
                    winner = f"{args.custom_name.upper()} ✓" if cv > ev else ("EXPERT ✓" if ev > cv else "平手")
                    print(f"{name:<35} {cv:>10.1f} {ev:>10.1f}   {winner}")
                
                compare_lower_better("Crash Vehicle Rate", c_crash_rate, e_crash_rate)
                compare_lower_better("Any Bad Event Rate", c_any_bad, e_any_bad)
                compare_lower_better("Close Encounter Crash Rate", c_close_crash, e_close_crash)
                compare_higher_better("Dangerous Close Count (avg)", c_dangerous, e_dangerous)
                compare_higher_better("Safe Pass Count (avg)", c_safe_pass, e_safe_pass)
                compare_higher_better("Min Vehicle Distance (m)", c_min_dist, e_min_dist)
                compare_higher_better("Success (no bad event) Rate %", c_success * 100, e_success * 100)
                
                # Count per-seed wins on first n seeds
                custom_wins = 0
                expert_wins = 0
                for j in range(n):
                    if custom_results[j]['close_encounter_crash_rate'] < expert_results[j]['close_encounter_crash_rate']:
                        custom_wins += 1
                    elif expert_results[j]['close_encounter_crash_rate'] < custom_results[j]['close_encounter_crash_rate']:
                        expert_wins += 1
                print(f"\nPer-seed wins (by close_encounter_crash_rate): {args.custom_name.upper()}={custom_wins}, EXPERT={expert_wins}, Tie={n-custom_wins-expert_wins}")
        
        # Final summary with comprehensive metrics
        print(f"\n{'='*80}")
        print("DETAILED BEHAVIORAL COMPARISON")
        print(f"{'='*80}")
        
        def get_mean(results, key, default=0):
            vals = [r.get(key, default) for r in results]
            return np.mean(vals) if vals else default
        
        # Helper to get min of positive values
        def get_min_positive(results, key):
            vals = [r[key] for r in results if r.get(key, -1) > 0]
            return np.min(vals) if vals else -1
        
        # Build comparison table
        metrics = [
            # (name, key, higher_is_better, format_str, multiply_100)
            ("======= 安全性指标 =======", None, None, None, None),
            ("Crash Vehicle Rate", 'crash_vehicle_rate', False, '{:.1f}%', True),
            ("Crash Object Rate", 'crash_object_rate', False, '{:.1f}%', True),
            ("Out of Road Rate", 'out_of_road_rate', False, '{:.1f}%', True),
            ("Any Bad Event Rate", 'any_bad_event_rate', False, '{:.1f}%', True),
            ("Success (no bad event)", 'success_no_bad_event_rate', True, '{:.1f}%', True),
            ("Total Crash Count (avg)", 'crash_count_mean', False, '{:.1f}', False),
            ("======= 交通交互指标 =======", None, None, None, None),
            ("Close Encounter Crash Rate", 'close_encounter_crash_rate', False, '{:.1f}%', True),
            ("Safe Pass Count (avg)", 'safe_pass_count', True, '{:.1f}', False),
            ("Dangerous Close Count (avg)", 'dangerous_close_count', False, '{:.1f}', False),
            ("Total Close Encounters (avg)", 'total_close_encounters', None, '{:.1f}', False),
            ("Traffic Density Rate", 'traffic_density_rate', None, '{:.1f}%', True),
            ("======= 驾驶行为指标 =======", None, None, None, None),
            ("Mean |Steering|", 'mean_steering_abs', None, '{:.3f}', False),
            ("Mean |Accel|", 'mean_accel_abs', None, '{:.3f}', False),
            ("Steering Std", 'steering_std', None, '{:.3f}', False),
            ("Accel Std", 'accel_std', None, '{:.3f}', False),
            ("Hard Brake Ratio (accel<-0.5)", 'hard_brake_ratio', None, '{:.1f}%', True),
            ("Hard Steer Ratio (|steer|>0.5)", 'hard_steer_ratio', None, '{:.1f}%', True),
            ("Average Speed", 'avg_speed', None, '{:.1f}', False),
            ("Speed Std", 'speed_std', None, '{:.1f}', False),
            ("======= 距离指标 =======", None, None, None, None),
            ("Min Vehicle Distance (avg)", 'min_vehicle_distance', True, '{:.1f}m', False),
            ("Episode Length (avg)", 'length', None, '{:.0f}', False),
            ("Route Completion (avg)", 'route_completion', True, '{:.1f}%', True),
            ("Episode Reward (avg)", 'reward', True, '{:.1f}', False),
        ]
        
        print(f"\n{'Metric':<35} {args.custom_name.upper():<15} {'EXPERT':<15} 优势方")
        print("=" * 80)
        
        for metric_info in metrics:
            name, key, higher_is_better, fmt, mult_100 = metric_info
            
            if key is None:
                # Section header
                print(f"\n{name}")
                continue
            
            c_val = get_mean(custom_results, key)
            e_val = get_mean(expert_results, key)
            
            if mult_100:
                c_disp = fmt.format(c_val * 100)
                e_disp = fmt.format(e_val * 100)
            else:
                c_disp = fmt.format(c_val)
                e_disp = fmt.format(e_val)
            
            if higher_is_better is None:
                winner = ""
            elif higher_is_better:
                if c_val > e_val * 1.05:
                    winner = f"{args.custom_name.upper()} ✓"
                elif e_val > c_val * 1.05:
                    winner = "EXPERT ✓"
                else:
                    winner = "≈"
            else:
                if c_val < e_val * 0.95:
                    winner = f"{args.custom_name.upper()} ✓"
                elif e_val < c_val * 0.95:
                    winner = "EXPERT ✓"
                else:
                    winner = "≈"
            
            print(f"{name:<35} {c_disp:<15} {e_disp:<15} {winner}")
        
        # Special metrics with min values
        print(f"\n{'='*80}")
        print("特殊统计")
        print(f"{'='*80}")
        min_dists = [r['min_vehicle_distance'] for r in custom_results if r.get('min_vehicle_distance', -1) > 0]
        min_dists_e = [r['min_vehicle_distance'] for r in expert_results if r.get('min_vehicle_distance', -1) > 0]
        if min_dists:
            print(f"{args.custom_name.upper()} 最近车距: avg={np.mean(min_dists):.1f}m, min={np.min(min_dists):.1f}m, max={np.max(min_dists):.1f}m")
        if min_dists_e:
            print(f"EXPERT 最近车距: avg={np.mean(min_dists_e):.1f}m, min={np.min(min_dists_e):.1f}m, max={np.max(min_dists_e):.1f}m")
        
        # Per-seed comparison
        print(f"\n{'='*80}")
        print("Per-Seed Wins (哪个模型在每个seed上表现更好)")
        print(f"{'='*80}")
        
        metrics_to_compare = [
            ('crash_vehicle_rate', False, 'Crash Rate'),
            ('close_encounter_crash_rate', False, 'Close Enc Crash'),
            ('route_completion', True, 'Route Completion'),
            ('safe_pass_count', True, 'Safe Pass'),
        ]
        
        for key, higher_is_better, display_name in metrics_to_compare:
            custom_wins = 0
            expert_wins = 0
            ties = 0
            for i in range(len(seeds)):
                c_val = custom_results[i].get(key, 0)
                e_val = expert_results[i].get(key, 0)
                if higher_is_better:
                    if c_val > e_val:
                        custom_wins += 1
                    elif e_val > c_val:
                        expert_wins += 1
                    else:
                        ties += 1
                else:
                    if c_val < e_val:
                        custom_wins += 1
                    elif e_val < c_val:
                        expert_wins += 1
                    else:
                        ties += 1
            print(f"{display_name:<20}: {args.custom_name.upper()}={custom_wins}, EXPERT={expert_wins}, Tie={ties}")
        
        # Save results
        all_results = {
            args.custom_name: custom_results,
            'lidar_expert': expert_results
        }
        with open(output_path / "compare_expert_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        lidar_env.close()
        print(f"\nResults saved to {output_path}/")
        return
    
    # Don't close shared_env yet - it will be used for all evaluations
    
    # Evaluate each model on each seed
    all_results = {}
    
    for model_name, model in models_to_eval.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.upper()} on {len(seeds)} scenarios")
        print(f"{'='*60}")
        
        model_results = []
        
        for i, seed in enumerate(tqdm(seeds, desc=f"{model_name.upper()}")):
            result = evaluate_on_seed(
                model, seed, 
                num_episodes=args.num_episodes,
                use_render=args.render,
                expert_model=expert_model,
                shared_env=shared_env  # Use shared env for faster evaluation
            )
            model_results.append(result)
            
            # Print progress every 20 seeds
            if (i + 1) % 20 == 0:
                avg_reward = np.mean([r['reward'] for r in model_results])
                avg_success = np.mean([r['success_rate'] for r in model_results])
                print(f"\n  Progress {i+1}/{len(seeds)}: "
                      f"Avg Reward={avg_reward:.1f}, Success Rate={avg_success*100:.1f}%")
        
        all_results[model_name] = model_results
        
        # Print summary for this model
        rewards = [r['reward'] for r in model_results]
        success_rates = [r['success_rate'] for r in model_results]
        success_no_bad = [r['success_no_bad_event_rate'] for r in model_results]
        route_completion = [r['route_completion'] for r in model_results]
        route_no_bad = [r['route_completion_no_bad_event'] for r in model_results]
        crash_rates = [r['crash_vehicle_rate'] for r in model_results]
        any_bad_rates = [r['any_bad_event_rate'] for r in model_results]
        
        print(f"\n{model_name.upper()} Summary:")
        print(f"  Reward: mean={np.mean(rewards):.1f}, std={np.std(rewards):.1f}")
        print(f"  Route Completion: {np.mean(route_completion)*100:.1f}%")
        print(f"  Route Completion (no bad event): {np.mean(route_no_bad)*100:.1f}%")
        print(f"  Success Rate: {np.mean(success_rates)*100:.1f}%")
        print(f"  Success Rate (no bad event): {np.mean(success_no_bad)*100:.1f}%")
        print(f"  Crash Vehicle Rate: {np.mean(crash_rates)*100:.1f}%")
        print(f"  Any Bad Event Rate: {np.mean(any_bad_rates)*100:.1f}%")
        
        # Expert comparison if available
        if 'expert_action_diff_l2' in model_results[0]:
            expert_diffs = [r['expert_action_diff_l2'] for r in model_results]
            expert_agreement = [r['expert_agreement_ratio'] for r in model_results]
            print(f"  Expert Action Diff (L2): {np.mean(expert_diffs):.3f}")
            print(f"  Expert Agreement Ratio: {np.mean(expert_agreement)*100:.1f}%")
        
        # Traffic proximity metrics (NEW)
        if 'min_vehicle_distance' in model_results[0]:
            min_dists = [r['min_vehicle_distance'] for r in model_results if r['min_vehicle_distance'] > 0]
            close_encounters = [r['total_close_encounters'] for r in model_results]
            safe_passes = [r['safe_pass_count'] for r in model_results]
            dangerous = [r['dangerous_close_count'] for r in model_results]
            close_crash_rate = [r['close_encounter_crash_rate'] for r in model_results]
            
            print(f"  -- Traffic Proximity Metrics --")
            print(f"  Min Vehicle Distance: {np.mean(min_dists):.1f}m (min={np.min(min_dists):.1f}m)" if min_dists else "  Min Vehicle Distance: N/A")
            print(f"  Close Encounters (avg): {np.mean(close_encounters):.1f}")
            print(f"  Safe Passes (close but no crash): {np.mean(safe_passes):.1f}")
            print(f"  Dangerous Close (close + crash): {np.mean(dangerous):.1f}")
            print(f"  Close Encounter Crash Rate: {np.mean(close_crash_rate)*100:.1f}%")
    
    # Save results
    with open(output_path / "hard_scenario_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Compare models if multiple evaluated
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        model_names = list(all_results.keys())
        
        # Collect key metrics for each model
        metrics_by_model = {}
        for name, results in all_results.items():
            metrics_by_model[name] = {
                'reward': [r['reward'] for r in results],
                'success_rate': [r['success_rate'] for r in results],
                'success_no_bad': [r['success_no_bad_event_rate'] for r in results],
                'route_completion': [r['route_completion'] for r in results],
                'route_no_bad': [r['route_completion_no_bad_event'] for r in results],
                'any_bad_event': [r['any_bad_event_rate'] for r in results],
                'crash_vehicle': [r['crash_vehicle_rate'] for r in results],
            }
            if 'expert_agreement_ratio' in results[0]:
                metrics_by_model[name]['expert_agreement'] = [r['expert_agreement_ratio'] for r in results]
            # Traffic proximity metrics
            if 'close_encounter_crash_rate' in results[0]:
                metrics_by_model[name]['close_encounter_crash_rate'] = [r['close_encounter_crash_rate'] for r in results]
                metrics_by_model[name]['total_close_encounters'] = [r['total_close_encounters'] for r in results]
                metrics_by_model[name]['safe_pass_count'] = [r['safe_pass_count'] for r in results]
        
        # Print comparison table
        print(f"\n{'Metric':<30}", end="")
        for name in model_names:
            print(f"{name.upper():<15}", end="")
        print()
        print("-" * (30 + 15 * len(model_names)))
        
        metric_display = [
            ('reward', 'Reward (mean)', '{:.1f}'),
            ('success_rate', 'Success Rate', '{:.1%}'),
            ('success_no_bad', 'Success (no bad event)', '{:.1%}'),
            ('route_completion', 'Route Completion', '{:.1%}'),
            ('route_no_bad', 'Route (no bad event)', '{:.1%}'),
            ('any_bad_event', 'Any Bad Event Rate', '{:.1%}'),
            ('crash_vehicle', 'Crash Vehicle Rate', '{:.1%}'),
            ('expert_agreement', 'Expert Agreement', '{:.1%}'),
            # Traffic proximity metrics
            ('total_close_encounters', 'Close Encounters (avg)', '{:.1f}'),
            ('safe_pass_count', 'Safe Passes (avg)', '{:.1f}'),
            ('close_encounter_crash_rate', 'Close Encounter Crash Rate', '{:.1%}'),
        ]
        
        for key, label, fmt in metric_display:
            if key not in metrics_by_model[model_names[0]]:
                continue
            print(f"{label:<30}", end="")
            for name in model_names:
                val = np.mean(metrics_by_model[name][key])
                print(f"{fmt.format(val):<15}", end="")
            print()
        
        # Per-seed pairwise wins
        print(f"\n{'Pairwise Wins (by reward):'}")
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                rewards1 = metrics_by_model[name1]['reward']
                rewards2 = metrics_by_model[name2]['reward']
                wins1 = sum(1 for j in range(len(seeds)) if rewards1[j] > rewards2[j])
                wins2 = sum(1 for j in range(len(seeds)) if rewards2[j] > rewards1[j])
                ties = len(seeds) - wins1 - wins2
                print(f"  {name1.upper()} vs {name2.upper()}: {wins1}-{wins2}-{ties} (wins-losses-ties)")
        
        # Save comparison summary
        comparison = {'models': model_names}
        for name in model_names:
            comparison[f'{name}_reward'] = float(np.mean(metrics_by_model[name]['reward']))
            comparison[f'{name}_success_rate'] = float(np.mean(metrics_by_model[name]['success_rate']))
            comparison[f'{name}_success_no_bad'] = float(np.mean(metrics_by_model[name]['success_no_bad']))
            comparison[f'{name}_route_no_bad'] = float(np.mean(metrics_by_model[name]['route_no_bad']))
            comparison[f'{name}_any_bad_event'] = float(np.mean(metrics_by_model[name]['any_bad_event']))
            if 'expert_agreement' in metrics_by_model[name]:
                comparison[f'{name}_expert_agreement'] = float(np.mean(metrics_by_model[name]['expert_agreement']))
        
        with open(output_path / "comparison_summary.json", 'w') as f:
            json.dump(comparison, f, indent=2)
    
    # Clean up shared environment
    shared_env.close()
    
    print(f"\nResults saved to {output_path}/")


if __name__ == "__main__":
    main()
