#!/usr/bin/env python
"""
Compare multiple models (RL25500, PRETRAINED, EXPERT) on hard scenarios.
Outputs all metrics and provides rankings.

Usage:
    python compare_models_full.py --num_seeds 200
    python compare_models_full.py --num_seeds 50 --use_render  # for debugging
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Setup gymnasium compatibility
import gymnasium
sys.modules['gym'] = gymnasium

# Add project root to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from stable_baselines3 import PPO
from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
from metadrive.component.sensors.rgb_camera import RGBCamera


def load_hard_scenarios(json_path, num_seeds=200):
    """Load hard scenario seeds from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if 'scenarios' in data:
        # Format: {"scenarios": [{"scenario_seed": 1234, ...}, ...]}
        seeds = [s['scenario_seed'] for s in data['scenarios'][:num_seeds]]
    elif 'sorting_methods' in data:
        # Format: {"sorting_methods": {"by_reward_ascending": [1234, 5678, ...], ...}}
        # Use by_reward_ascending (hardest scenarios first)
        seeds = data['sorting_methods']['by_reward_ascending'][:num_seeds]
    else:
        raise ValueError(f"Unknown JSON format. Keys: {list(data.keys())}")
    
    print(f"Loaded {len(seeds)} hard scenario seeds")
    return seeds


def make_rgb_env_config(start_seed=1000):
    """Environment config for RGB-based models (RL25500, PRETRAINED)."""
    sensor_size = (84, 84)
    return dict(
        use_render=False,
        manual_control=False,
        traffic_density=0.12,
        random_traffic=False,
        accident_prob=0.0,
        horizon=1500,
        driving_reward=1.0,
        speed_reward=0.05,
        use_lateral_reward=False,
        out_of_road_penalty=40.0,
        crash_vehicle_penalty=40.0,
        decision_repeat=5,
        out_of_route_done=True,
        on_continuous_line_done=True,
        crash_vehicle_done=True,
        crash_object_done=False,
        image_observation=True,
        image_on_cuda=False,
        vehicle_config=dict(image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, *sensor_size)},
        stack_size=3,
        interface_panel=[],
        daytime="06:10",
        start_seed=start_seed,
        num_scenarios=1000,
        num_agents=1,
    )


def make_lidar_env_config(start_seed=1000):
    """Environment config for LIDAR-based models (EXPERT)."""
    return dict(
        use_render=False,
        manual_control=False,
        traffic_density=0.12,
        random_traffic=False,
        accident_prob=0.0,
        horizon=1500,
        driving_reward=1.0,
        speed_reward=0.05,
        use_lateral_reward=False,
        out_of_road_penalty=40.0,
        crash_vehicle_penalty=40.0,
        decision_repeat=5,
        out_of_route_done=True,
        on_continuous_line_done=True,
        crash_vehicle_done=True,
        crash_object_done=False,
        image_observation=False,
        interface_panel=[],
        daytime="06:10",
        start_seed=start_seed,
        num_scenarios=1000,
        num_agents=1,
    )


def load_td3_model(checkpoint_path, env):
    """Load TD3-based model (RL25500 or PRETRAINED)."""
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


def load_expert_policy(checkpoint_path):
    """Load LIDAR expert policy (returns policy only, not full model)."""
    from pvp.sb3.ppo import PPO as CustomPPO
    from pvp.sb3.ppo.policies import ActorCriticPolicy
    from pvp.sb3.common.save_util import load_from_zip_file
    
    # Create temp env for PPO initialization
    temp_env = HumanInTheLoopEnv(config={'manual_control': False, 'use_render': False})
    
    expert = CustomPPO(
        policy=ActorCriticPolicy,
        env=temp_env,
        n_steps=1024,
        verbose=0,
        device="auto",
    )
    
    if Path(checkpoint_path).exists():
        data, params, pytorch_variables = load_from_zip_file(
            checkpoint_path, device=expert.device, print_system_info=False
        )
        expert.set_parameters(params, exact_match=True, device=expert.device)
        print(f"Expert loaded from {checkpoint_path}")
    else:
        print(f"WARNING: Expert checkpoint not found at {checkpoint_path}")
    
    temp_env.close()
    return expert.policy


def evaluate_rgb_model(model, env, seed, expert_policy=None):
    """Evaluate RGB-based model on a single seed."""
    reset_result = env.reset(seed=seed)
    # Handle both old (obs) and new (obs, info) reset API
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
        info = {}
    
    # Episode tracking
    done = False
    total_reward = 0.0
    step_count = 0
    
    # Bad event tracking
    had_crash_vehicle = False
    had_crash_object = False
    had_out_of_road = False
    
    # Traffic proximity tracking
    min_vehicle_distance = float('inf')
    total_close_encounters = 0
    safe_pass_count = 0
    dangerous_close_count = 0
    steps_with_traffic = 0
    
    # Behavioral tracking
    steering_actions = []
    accel_actions = []
    speeds = []
    
    # Expert comparison
    expert_action_diffs = []
    expert_agreements = []
    
    while not done:
        # Get model action
        action, _ = model.predict(obs, deterministic=True)
        
        # Expert comparison (if expert available)
        if expert_policy is not None:
            try:
                lidar_obs = info.get('lidar_obs', None)
                if lidar_obs is not None:
                    import torch
                    with torch.no_grad():
                        lidar_tensor = torch.FloatTensor(lidar_obs).unsqueeze(0).to("cuda")
                        expert_action = expert_policy(lidar_tensor, deterministic=True)[0].cpu().numpy().flatten()
                    action_diff = np.linalg.norm(action - expert_action)
                    expert_action_diffs.append(action_diff)
                    agreement = 1.0 if action_diff < 0.3 else 0.0
                    expert_agreements.append(agreement)
            except:
                pass
        
        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Track actions and speed
        steering_actions.append(action[0])
        accel_actions.append(action[1])
        if 'velocity' in info:
            speeds.append(np.linalg.norm(info['velocity']))
        
        # Track bad events
        if info.get('crash_vehicle', False):
            had_crash_vehicle = True
        if info.get('crash_object', False):
            had_crash_object = True
        if info.get('out_of_road', False):
            had_out_of_road = True
        
        # Track traffic proximity
        if 'min_vehicle_distance' in info:
            dist = info['min_vehicle_distance']
            if dist < float('inf'):
                min_vehicle_distance = min(min_vehicle_distance, dist)
                steps_with_traffic += 1
        
        if 'close_vehicle_count' in info:
            ce = info['close_vehicle_count']
            total_close_encounters += ce
            if ce > 0:
                if had_crash_vehicle:
                    dangerous_close_count += ce
                else:
                    safe_pass_count += ce
    
    # Compute metrics
    had_any_bad = had_crash_vehicle or had_crash_object or had_out_of_road
    arrive_dest = info.get('arrive_dest', False)
    route_completion = info.get('route_completion', 0.0)
    
    # Behavioral metrics
    steering_std = np.std(steering_actions) if steering_actions else 0.0
    accel_std = np.std(accel_actions) if accel_actions else 0.0
    mean_steering_abs = np.mean(np.abs(steering_actions)) if steering_actions else 0.0
    mean_accel_abs = np.mean(np.abs(accel_actions)) if accel_actions else 0.0
    hard_brake_ratio = np.mean([1 if a < -0.5 else 0 for a in accel_actions]) if accel_actions else 0.0
    hard_steer_ratio = np.mean([1 if abs(s) > 0.5 else 0 for s in steering_actions]) if steering_actions else 0.0
    avg_speed = np.mean(speeds) if speeds else 0.0
    speed_std = np.std(speeds) if speeds else 0.0
    
    # Traffic density
    traffic_density_rate = steps_with_traffic / step_count if step_count > 0 else 0.0
    
    # Close encounter crash rate
    total_close = safe_pass_count + dangerous_close_count
    close_enc_crash_rate = dangerous_close_count / total_close if total_close > 0 else 0.0
    
    # Safe pass rate
    safe_pass_rate = safe_pass_count / total_close_encounters if total_close_encounters > 0 else 0.0
    
    return {
        'seed': seed,
        'reward': total_reward,
        'route_completion': route_completion,
        'arrive_dest': bool(arrive_dest),
        'success_no_bad': bool(arrive_dest and not had_any_bad),
        'route_completion_no_bad': route_completion if not had_any_bad else 0.0,
        'crash_vehicle': had_crash_vehicle,
        'crash_object': had_crash_object,
        'out_of_road': had_out_of_road,
        'any_bad_event': had_any_bad,
        'episode_length': step_count,
        # Traffic proximity
        'min_vehicle_distance': min_vehicle_distance if min_vehicle_distance < float('inf') else 100.0,
        'total_close_encounters': total_close_encounters,
        'safe_pass_count': safe_pass_count,
        'dangerous_close_count': dangerous_close_count,
        'close_enc_crash_rate': close_enc_crash_rate,
        'safe_pass_rate': safe_pass_rate,
        'steps_with_traffic': steps_with_traffic,
        'traffic_density_rate': traffic_density_rate,
        # Behavioral
        'steering_std': steering_std,
        'accel_std': accel_std,
        'mean_steering_abs': mean_steering_abs,
        'mean_accel_abs': mean_accel_abs,
        'hard_brake_ratio': hard_brake_ratio,
        'hard_steer_ratio': hard_steer_ratio,
        'avg_speed': avg_speed,
        'speed_std': speed_std,
        # Expert comparison
        'expert_action_diff': np.mean(expert_action_diffs) if expert_action_diffs else 0.0,
        'expert_agreement': np.mean(expert_agreements) if expert_agreements else 0.0,
    }


def evaluate_expert(policy, env, seed):
    """Evaluate LIDAR expert on a single seed."""
    import torch
    
    reset_result = env.reset(seed=seed)
    # Handle both old (obs) and new (obs, info) reset API
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
        info = {}
    
    # Episode tracking
    done = False
    total_reward = 0.0
    step_count = 0
    
    # Bad event tracking
    had_crash_vehicle = False
    had_crash_object = False
    had_out_of_road = False
    
    # Traffic proximity tracking
    min_vehicle_distance = float('inf')
    total_close_encounters = 0
    safe_pass_count = 0
    dangerous_close_count = 0
    steps_with_traffic = 0
    
    # Behavioral tracking
    steering_actions = []
    accel_actions = []
    speeds = []
    
    while not done:
        # Get expert action
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to("cuda")
            action = policy(obs_tensor, deterministic=True)[0].cpu().numpy().flatten()
        
        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Track actions and speed
        steering_actions.append(action[0])
        accel_actions.append(action[1])
        if 'velocity' in info:
            speeds.append(np.linalg.norm(info['velocity']))
        
        # Track bad events
        if info.get('crash_vehicle', False):
            had_crash_vehicle = True
        if info.get('crash_object', False):
            had_crash_object = True
        if info.get('out_of_road', False):
            had_out_of_road = True
        
        # Track traffic proximity
        if 'min_vehicle_distance' in info:
            dist = info['min_vehicle_distance']
            if dist < float('inf'):
                min_vehicle_distance = min(min_vehicle_distance, dist)
                steps_with_traffic += 1
        
        if 'close_vehicle_count' in info:
            ce = info['close_vehicle_count']
            total_close_encounters += ce
            if ce > 0:
                if had_crash_vehicle:
                    dangerous_close_count += ce
                else:
                    safe_pass_count += ce
    
    # Compute metrics
    had_any_bad = had_crash_vehicle or had_crash_object or had_out_of_road
    arrive_dest = info.get('arrive_dest', False)
    route_completion = info.get('route_completion', 0.0)
    
    # Behavioral metrics
    steering_std = np.std(steering_actions) if steering_actions else 0.0
    accel_std = np.std(accel_actions) if accel_actions else 0.0
    mean_steering_abs = np.mean(np.abs(steering_actions)) if steering_actions else 0.0
    mean_accel_abs = np.mean(np.abs(accel_actions)) if accel_actions else 0.0
    hard_brake_ratio = np.mean([1 if a < -0.5 else 0 for a in accel_actions]) if accel_actions else 0.0
    hard_steer_ratio = np.mean([1 if abs(s) > 0.5 else 0 for s in steering_actions]) if steering_actions else 0.0
    avg_speed = np.mean(speeds) if speeds else 0.0
    speed_std = np.std(speeds) if speeds else 0.0
    
    # Traffic density
    traffic_density_rate = steps_with_traffic / step_count if step_count > 0 else 0.0
    
    # Close encounter crash rate
    total_close = safe_pass_count + dangerous_close_count
    close_enc_crash_rate = dangerous_close_count / total_close if total_close > 0 else 0.0
    
    # Safe pass rate
    safe_pass_rate = safe_pass_count / total_close_encounters if total_close_encounters > 0 else 0.0
    
    return {
        'seed': seed,
        'reward': total_reward,
        'route_completion': route_completion,
        'arrive_dest': bool(arrive_dest),
        'success_no_bad': bool(arrive_dest and not had_any_bad),
        'route_completion_no_bad': route_completion if not had_any_bad else 0.0,
        'crash_vehicle': had_crash_vehicle,
        'crash_object': had_crash_object,
        'out_of_road': had_out_of_road,
        'any_bad_event': had_any_bad,
        'episode_length': step_count,
        # Traffic proximity
        'min_vehicle_distance': min_vehicle_distance if min_vehicle_distance < float('inf') else 100.0,
        'total_close_encounters': total_close_encounters,
        'safe_pass_count': safe_pass_count,
        'dangerous_close_count': dangerous_close_count,
        'close_enc_crash_rate': close_enc_crash_rate,
        'safe_pass_rate': safe_pass_rate,
        'steps_with_traffic': steps_with_traffic,
        'traffic_density_rate': traffic_density_rate,
        # Behavioral
        'steering_std': steering_std,
        'accel_std': accel_std,
        'mean_steering_abs': mean_steering_abs,
        'mean_accel_abs': mean_accel_abs,
        'hard_brake_ratio': hard_brake_ratio,
        'hard_steer_ratio': hard_steer_ratio,
        'avg_speed': avg_speed,
        'speed_std': speed_std,
        # Expert comparison (N/A for expert itself)
        'expert_action_diff': 0.0,
        'expert_agreement': 1.0,
    }


def aggregate_results(results_list):
    """Aggregate results from multiple seeds."""
    if not results_list:
        return {}
    
    agg = {}
    keys = results_list[0].keys()
    
    for key in keys:
        if key == 'seed':
            continue
        values = [r[key] for r in results_list]
        if isinstance(values[0], bool):
            agg[key] = np.mean([float(v) for v in values])
        elif isinstance(values[0], (int, float, np.floating)):
            agg[f'{key}_mean'] = np.mean(values)
            agg[f'{key}_std'] = np.std(values)
            agg[f'{key}_min'] = np.min(values)
            agg[f'{key}_max'] = np.max(values)
    
    return agg


def print_comparison_table(all_results, model_names):
    """Print comprehensive comparison table."""
    # Ensure model_names is a list for indexing
    model_names = list(model_names)
    
    print("\n" + "=" * 100)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 100)
    
    # Define metrics to compare
    metrics = [
        # (display_name, key, higher_is_better, format_as_percent)
        ("安全性指标", None, None, None),
        ("Crash Vehicle Rate", "crash_vehicle", False, True),
        ("Crash Object Rate", "crash_object", False, True),
        ("Out of Road Rate", "out_of_road", False, True),
        ("Any Bad Event Rate", "any_bad_event", False, True),
        ("Success (no bad event)", "success_no_bad", True, True),
        
        ("交通交互指标", None, None, None),
        ("Min Vehicle Distance (m)", "min_vehicle_distance_mean", True, False),
        ("Close Encounters (avg)", "total_close_encounters_mean", None, False),
        ("Safe Pass Count", "safe_pass_count_mean", True, False),
        ("Dangerous Close Count", "dangerous_close_count_mean", False, False),
        ("Close Enc Crash Rate", "close_enc_crash_rate_mean", False, True),
        ("Safe Pass Rate", "safe_pass_rate_mean", True, True),
        ("Traffic Density Rate", "traffic_density_rate_mean", None, True),
        
        ("导航表现", None, None, None),
        ("Route Completion", "route_completion_mean", True, True),
        ("Route Comp (no bad)", "route_completion_no_bad_mean", True, True),
        ("Arrive Dest Rate", "arrive_dest", True, True),
        ("Episode Reward", "reward_mean", True, False),
        ("Episode Length", "episode_length_mean", None, False),
        
        ("行为指标", None, None, None),
        ("Avg Speed", "avg_speed_mean", True, False),
        ("Speed Std", "speed_std_mean", None, False),
        ("Steering Std", "steering_std_mean", None, False),
        ("Accel Std", "accel_std_mean", None, False),
        ("Hard Brake Ratio", "hard_brake_ratio_mean", False, True),
        ("Hard Steer Ratio", "hard_steer_ratio_mean", None, True),
        
        ("专家对比", None, None, None),
        ("Expert Action Diff (L2)", "expert_action_diff_mean", False, False),
        ("Expert Agreement Ratio", "expert_agreement_mean", True, True),
    ]
    
    # Print header
    header = f"{'Metric':<30}"
    for name in model_names:
        header += f"{name:>18}"
    header += "   Ranking"
    print(header)
    print("-" * 100)
    
    # Track wins
    wins = {name: 0 for name in model_names}
    
    for metric_info in metrics:
        display_name, key, higher_is_better, as_percent = metric_info
        
        if key is None:
            print(f"\n>>> {display_name} <<<")
            continue
        
        row = f"{display_name:<30}"
        values = []
        
        for name in model_names:
            r = all_results.get(name, {})
            v = r.get(key, 0.0)
            # Handle None or NaN values
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = 0.0
            values.append(float(v))
            
            if as_percent:
                row += f"{v*100:>17.1f}%"
            else:
                row += f"{v:>18.2f}"
        
        # Determine ranking
        if higher_is_better is not None and len(set(values)) > 1:
            if higher_is_better:
                sorted_indices = np.argsort(values)[::-1]  # descending
            else:
                sorted_indices = np.argsort(values)  # ascending
            
            ranking = []
            for i, idx in enumerate(sorted_indices):
                ranking.append(f"{i+1}.{model_names[idx]}")
                if i == 0:
                    wins[model_names[idx]] += 1
            
            row += f"   {' > '.join(ranking[:2])}"
        else:
            row += ""
        
        print(row)
    
    # Print win summary
    print("\n" + "=" * 100)
    print("WIN SUMMARY (lower/higher is better depending on metric)")
    print("=" * 100)
    for name in model_names:
        print(f"  {name}: {wins[name]} metric wins")
    
    # Overall ranking
    print("\n" + "=" * 100)
    print("OVERALL RANKING")
    print("=" * 100)
    
    # Compute composite scores
    scores = {}
    for name in model_names:
        r = all_results.get(name, {})
        if not r:
            print(f"  WARNING: No results for {name}")
            scores[name] = 0.0
            continue
            
        # Safety score (higher is better)
        crash_vehicle = r.get('crash_vehicle', 0) or 0
        any_bad_event = r.get('any_bad_event', 0) or 0
        success_no_bad = r.get('success_no_bad', 0) or 0
        safety = (1 - crash_vehicle) * 0.4 + \
                 (1 - any_bad_event) * 0.3 + \
                 success_no_bad * 0.3
        
        # Traffic interaction score
        close_enc_crash = r.get('close_enc_crash_rate_mean', 0) or 0
        safe_pass_rate = r.get('safe_pass_rate_mean', 0) or 0
        traffic = (1 - close_enc_crash) * 0.5 + safe_pass_rate * 0.5
        
        # Navigation score
        route_comp = r.get('route_completion_mean', 0) or 0
        arrive_dest = r.get('arrive_dest', 0) or 0
        nav = route_comp * 0.5 + arrive_dest * 0.5
        
        # Composite
        scores[name] = safety * 0.4 + traffic * 0.3 + nav * 0.3
    
    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\nComposite Score (Safety 40% + Traffic Interaction 30% + Navigation 30%):")
    for i, (name, score) in enumerate(sorted_models):
        emoji = ['🥇', '🥈', '🥉'][i] if i < 3 else f'{i+1}.'
        print(f"  {emoji} {name}: {score:.4f}")
    
    return scores


def main():
    parser = argparse.ArgumentParser(description="Compare models on hard scenarios")
    parser.add_argument("--num_seeds", type=int, default=200, help="Number of seeds to evaluate")
    parser.add_argument("--use_render", action="store_true", help="Enable rendering")
    parser.add_argument("--output_dir", type=str, default="results/model_comparison", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load hard scenarios
    json_path = script_dir / "results" / "hard_scenario_eval" / "hard_scenarios_expert.json"
    if not json_path.exists():
        json_path = script_dir / "hard_scenarios_expert.json"
    seeds = load_hard_scenarios(json_path, args.num_seeds)
    
    # Model configurations
    # Evaluate EXPERT first for debugging
    from collections import OrderedDict
    models_config = OrderedDict([
        ('EXPERT', {
            'type': 'lidar',
            'checkpoint': str(script_dir / 'pvp' / 'experiments' / 'metadrive' / 'egpo' / 'metadrive_pvp_20m_steps.zip'),
        }),
        ('RL25500', {
            'type': 'rgb',
            'checkpoint': str(script_dir / 'rl_model_25500_steps.zip'),
        }),
        ('PRETRAINED', {
            'type': 'rgb', 
            'checkpoint': str(script_dir / 'pretrained.zip'),
        }),
    ])
    
    all_results = {}
    all_per_seed = {}
    
    # Evaluate each model
    for model_name, config in models_config.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        results_list = []
        
        if config['type'] == 'rgb':
            # Create RGB environment
            min_seed = min(seeds)
            env_config = make_rgb_env_config(start_seed=min_seed)
            if args.use_render:
                env_config['use_render'] = True
            
            env = HumanInTheLoopEnv(env_config)
            model = load_td3_model(config['checkpoint'], env)
            
            for seed in tqdm(seeds, desc=model_name):
                result = evaluate_rgb_model(model, env, seed)
                results_list.append(result)
            
            env.close()
            
        else:  # lidar
            # Create LIDAR environment
            min_seed = min(seeds)
            env_config = make_lidar_env_config(start_seed=min_seed)
            if args.use_render:
                env_config['use_render'] = True
            
            env = HumanInTheLoopEnv(env_config)
            policy = load_expert_policy(config['checkpoint'])
            
            for seed in tqdm(seeds, desc=model_name):
                result = evaluate_expert(policy, env, seed)
                results_list.append(result)
            
            env.close()
        
        # Aggregate results
        agg = aggregate_results(results_list)
        all_results[model_name] = agg
        all_per_seed[model_name] = results_list
        
        # Print ALL metrics (same as comparison table) for debugging
        print(f"\n{'='*60}")
        print(f"{model_name} - ALL METRICS (for debugging)")
        print(f"{'='*60}")
        
        # Helper to safely get value
        def safe_get(key, default=0):
            v = agg.get(key, default)
            return v if v is not None and not (isinstance(v, float) and np.isnan(v)) else default
        
        print(f"\n>>> 安全性指标 <<<")
        print(f"  Crash Vehicle Rate:      {safe_get('crash_vehicle')*100:>8.1f}%")
        print(f"  Crash Object Rate:       {safe_get('crash_object')*100:>8.1f}%")
        print(f"  Out of Road Rate:        {safe_get('out_of_road')*100:>8.1f}%")
        print(f"  Any Bad Event Rate:      {safe_get('any_bad_event')*100:>8.1f}%")
        print(f"  Success (no bad event):  {safe_get('success_no_bad')*100:>8.1f}%")
        
        print(f"\n>>> 交通交互指标 <<<")
        print(f"  Min Vehicle Distance:    {safe_get('min_vehicle_distance_mean'):>8.2f} m")
        print(f"  Close Encounters (avg):  {safe_get('total_close_encounters_mean'):>8.2f}")
        print(f"  Safe Pass Count:         {safe_get('safe_pass_count_mean'):>8.2f}")
        print(f"  Dangerous Close Count:   {safe_get('dangerous_close_count_mean'):>8.2f}")
        print(f"  Close Enc Crash Rate:    {safe_get('close_enc_crash_rate_mean')*100:>8.1f}%")
        print(f"  Safe Pass Rate:          {safe_get('safe_pass_rate_mean')*100:>8.1f}%")
        print(f"  Traffic Density Rate:    {safe_get('traffic_density_rate_mean')*100:>8.1f}%")
        
        print(f"\n>>> 导航表现 <<<")
        print(f"  Route Completion:        {safe_get('route_completion_mean')*100:>8.1f}%")
        print(f"  Route Comp (no bad):     {safe_get('route_completion_no_bad_mean')*100:>8.1f}%")
        print(f"  Arrive Dest Rate:        {safe_get('arrive_dest')*100:>8.1f}%")
        print(f"  Episode Reward:          {safe_get('reward_mean'):>8.1f}")
        print(f"  Episode Length:          {safe_get('episode_length_mean'):>8.1f}")
        
        print(f"\n>>> 行为指标 <<<")
        print(f"  Avg Speed:               {safe_get('avg_speed_mean'):>8.2f}")
        print(f"  Speed Std:               {safe_get('speed_std_mean'):>8.2f}")
        print(f"  Steering Std:            {safe_get('steering_std_mean'):>8.4f}")
        print(f"  Accel Std:               {safe_get('accel_std_mean'):>8.4f}")
        print(f"  Hard Brake Ratio:        {safe_get('hard_brake_ratio_mean')*100:>8.1f}%")
        print(f"  Hard Steer Ratio:        {safe_get('hard_steer_ratio_mean')*100:>8.1f}%")
        
        print(f"\n>>> 专家对比 <<<")
        print(f"  Expert Action Diff (L2): {safe_get('expert_action_diff_mean'):>8.4f}")
        print(f"  Expert Agreement Ratio:  {safe_get('expert_agreement_mean')*100:>8.1f}%")
        
        # Also print raw keys for debugging
        print(f"\n>>> Debug: Available keys in agg <<<")
        print(f"  {list(agg.keys())}")
    
    # Print comparison table
    model_names = list(models_config.keys())
    scores = print_comparison_table(all_results, model_names)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save aggregated results
    save_path = os.path.join(args.output_dir, f"comparison_{args.num_seeds}seeds_{timestamp}.json")
    with open(save_path, 'w') as f:
        json.dump({
            'num_seeds': args.num_seeds,
            'seeds': seeds,
            'aggregated_results': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                                       for kk, vv in v.items()} 
                                  for k, v in all_results.items()},
            'composite_scores': {k: float(v) for k, v in scores.items()},
        }, f, indent=2)
    print(f"\nResults saved to {save_path}")
    
    # Save per-seed results
    per_seed_path = os.path.join(args.output_dir, f"per_seed_{args.num_seeds}seeds_{timestamp}.json")
    with open(per_seed_path, 'w') as f:
        json.dump({
            k: [{kk: float(vv) if isinstance(vv, (np.floating, np.integer, np.bool_)) else vv 
                 for kk, vv in r.items()} 
                for r in v]
            for k, v in all_per_seed.items()
        }, f, indent=2)
    print(f"Per-seed results saved to {per_seed_path}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
