"""
Evaluation with Lidar Expert Comparison (Sequential Mode).

This script uses sequential evaluation with reset(seed=seed) to enable
lidar expert comparison. The lidar expert requires lidar observations
which are only available in sequential mode via info['lidar_obs'].

Usage:
    python eval_hard_with_expert.py --model iql --num_seeds 20
"""

import argparse
import os
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

# Suppress metadrive logging
import logging
logging.getLogger('metadrive').setLevel(logging.WARNING)
logging.getLogger('panda3d').setLevel(logging.WARNING)


# Top 200 hardest scenarios
TOP_200_SEEDS = [
    1832, 1683, 1786, 1081, 1175, 1111, 1946, 1839, 1821, 1802,
    1213, 1466, 1604, 1911, 1612, 1650, 1831, 1497, 1364, 1886,
    1543, 1970, 1787, 1389, 1118, 1516, 1847, 1425, 1151, 1208,
    1735, 1637, 1228, 1229, 1789, 1405, 1179, 1943, 1508, 1509,
    1790, 1137, 1538, 1837, 1764, 1935, 1335, 1424, 1026, 1272,
]


def load_lidar_expert():
    """Load the lidar-based PPO expert for comparison."""
    from pvp.sb3.ppo import PPO
    from pvp.sb3.ppo.policies import ActorCriticPolicy
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    
    # Create temp env with lidar observation for expert
    temp_env = HumanInTheLoopEnv(config={
        'manual_control': False, 
        'use_render': False,
        'image_observation': False,  # Lidar mode
    })
    
    expert = PPO(
        policy=ActorCriticPolicy,
        env=temp_env,
        n_steps=1024,
        verbose=0,
        device="auto",
    )
    
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    ckpt = script_dir / "pvp" / "experiments" / "metadrive" / "egpo" / "metadrive_pvp_20m_steps"
    
    expert.set_parameters(str(ckpt))
    temp_env.close()
    
    print("Lidar expert loaded!")
    return expert


def evaluate_with_expert(model, seeds, expert_model=None):
    """
    Evaluate model on seeds with expert comparison.
    Uses sequential mode to get lidar observations for expert.
    """
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    
    # Create shared env with both RGB and lidar info
    min_seed = min(seeds)
    max_seed = max(seeds)
    
    from metadrive.component.sensors.rgb_camera import RGBCamera
    env = HumanInTheLoopEnv(config={
        'use_render': False,
        'manual_control': False,
        'image_observation': True,  # RGB for model
        'start_seed': min_seed,
        'num_scenarios': max_seed - min_seed + 100,
        'sensors': {
            'rgb_camera': (RGBCamera, 84, 84),
        },
        'vehicle_config': {
            'image_source': 'rgb_camera',
        },
    })
    
    all_results = []
    start_time = time.time()
    
    for idx, seed in enumerate(seeds):
        result = env.reset(seed=seed)
        # Handle both (obs, info) tuple and just obs return
        if isinstance(result, tuple):
            obs, info = result
            if not isinstance(info, dict):
                info = {}
        else:
            obs = result
            info = {}
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Tracking
        route_completion = 0.0
        had_crash_vehicle = False
        had_crash_object = False
        had_out_of_road = False
        arrive_dest = False
        crash_count = 0
        min_vehicle_distance = float('inf')
        total_close_encounters = 0
        safe_pass_count = 0
        dangerous_close_count = 0
        
        actions_list = []
        expert_actions_list = []
        
        while not done:
            # Get model action
            action, _ = model.predict(obs, deterministic=True)
            actions_list.append(action.copy())
            
            # Get expert action from lidar observation (if available)
            if expert_model is not None and 'lidar_obs' in info and info['lidar_obs'] is not None:
                lidar_obs = info['lidar_obs']
                expert_action, _ = expert_model.predict(lidar_obs, deterministic=True)
                expert_actions_list.append(expert_action.copy())
            
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            episode_reward += reward
            episode_length += 1
            
            route_completion = max(route_completion, info.get('route_completion', 0.0))
            
            # Track crashes
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
            if info.get('arrive_dest', False):
                arrive_dest = True
            
            # Traffic proximity
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
        
        # Calculate metrics
        had_bad_event = had_crash_vehicle or had_crash_object or had_out_of_road
        rc_no_bad = route_completion if not had_bad_event else route_completion * 0.5
        close_encounter_crash_rate = dangerous_close_count / total_close_encounters if total_close_encounters > 0 else 0.0
        
        actions_arr = np.array(actions_list) if actions_list else np.array([[0, 0]])
        
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
            'mean_steering_abs': float(np.mean(np.abs(actions_arr[:, 0]))),
            'mean_accel_abs': float(np.mean(np.abs(actions_arr[:, 1]))),
            'min_vehicle_distance': float(min_vehicle_distance) if min_vehicle_distance != float('inf') else -1,
            'total_close_encounters': float(total_close_encounters),
            'close_encounter_rate': float(total_close_encounters / episode_length) if episode_length > 0 else 0.0,
            'safe_pass_count': float(safe_pass_count),
            'dangerous_close_count': float(dangerous_close_count),
            'close_encounter_crash_rate': float(close_encounter_crash_rate),
        }
        
        # Expert comparison
        if len(expert_actions_list) > 0:
            expert_arr = np.array(expert_actions_list)
            min_len = min(len(actions_arr), len(expert_arr))
            if min_len > 0:
                action_diffs = np.linalg.norm(actions_arr[:min_len] - expert_arr[:min_len], axis=1)
                steering_diffs = np.abs(actions_arr[:min_len, 0] - expert_arr[:min_len, 0])
                accel_diffs = np.abs(actions_arr[:min_len, 1] - expert_arr[:min_len, 1])
                result['expert_action_diff_l2'] = float(np.mean(action_diffs))
                result['expert_steering_diff'] = float(np.mean(steering_diffs))
                result['expert_accel_diff'] = float(np.mean(accel_diffs))
                result['expert_agreement_ratio'] = float(np.mean(action_diffs < 0.3))
        
        all_results.append(result)
        
        elapsed = time.time() - start_time
        expert_diff = result.get('expert_action_diff_l2', None)
        expert_diff_str = f"{expert_diff:.3f}" if expert_diff is not None else "N/A"
        print(f"  [{idx+1}/{len(seeds)}] seed={seed}, reward={episode_reward:.1f}, "
              f"success={arrive_dest}, expert_diff={expert_diff_str}")
    
    env.close()
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate with lidar expert comparison")
    parser.add_argument("--model", type=str, choices=['iql', 'td3bc2', 'both'], default='iql')
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--output", type=Path, default="./results/expert_comparison")
    parser.add_argument("--iql_checkpoint", type=str, default="IQLBEST1.zip")
    parser.add_argument("--td3_checkpoint", type=str, default="TD3BCBEST2.zip")
    args = parser.parse_args()
    
    seeds = TOP_200_SEEDS[:args.num_seeds]
    print(f"Evaluating on {len(seeds)} seeds with expert comparison")
    
    # Load expert
    print("Loading lidar expert...")
    expert_model = load_lidar_expert()
    
    # Load models
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    
    min_seed = min(seeds)
    max_seed = max(seeds)
    
    from metadrive.component.sensors.rgb_camera import RGBCamera
    temp_env = HumanInTheLoopEnv(config={
        'use_render': False,
        'manual_control': False,
        'image_observation': True,
        'start_seed': min_seed,
        'num_scenarios': max_seed - min_seed + 100,
        'sensors': {
            'rgb_camera': (RGBCamera, 84, 84),
        },
        'vehicle_config': {
            'image_source': 'rgb_camera',
        },
    })
    
    models_to_eval = {}
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    if args.model in ['iql', 'both']:
        from pvp.sb3.td3.iql import IQL
        iql_path = script_dir / args.iql_checkpoint
        if iql_path.exists():
            print(f"Loading IQL from {iql_path}...")
            iql_model = IQL.load(str(iql_path), env=temp_env)
            models_to_eval['iql'] = iql_model
            print("IQL loaded!")
    
    if args.model in ['td3bc2', 'both']:
        from pvp.sb3.td3.td3_with_bc import TD3BC
        td3_path = script_dir / args.td3_checkpoint
        if td3_path.exists():
            print(f"Loading TD3BC2 from {td3_path}...")
            td3_model = TD3BC.load(str(td3_path), env=temp_env)
            models_to_eval['td3bc2'] = td3_model
            print("TD3BC2 loaded!")
    
    temp_env.close()
    
    # Evaluate
    args.output.mkdir(parents=True, exist_ok=True)
    all_results = {}
    start_time = time.time()
    
    for model_name, model in models_to_eval.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.upper()} with expert comparison")
        print(f"{'='*60}")
        
        model_results = evaluate_with_expert(model, seeds, expert_model)
        all_results[model_name] = model_results
        
        # Print summary
        rewards = [r['reward'] for r in model_results]
        success_rates = [r['success_rate'] for r in model_results]
        crash_rates = [r['crash_vehicle_rate'] for r in model_results]
        
        print(f"\n{model_name.upper()} Summary:")
        print(f"  Reward: mean={np.mean(rewards):.1f}, std={np.std(rewards):.1f}")
        print(f"  Success Rate: {np.mean(success_rates)*100:.1f}%")
        print(f"  Crash Rate: {np.mean(crash_rates)*100:.1f}%")
        
        # Expert comparison
        if 'expert_action_diff_l2' in model_results[0]:
            expert_diffs = [r['expert_action_diff_l2'] for r in model_results]
            expert_agreement = [r['expert_agreement_ratio'] for r in model_results]
            print(f"  --- Expert Comparison ---")
            print(f"  Expert Action Diff (L2): {np.mean(expert_diffs):.3f}")
            print(f"  Expert Agreement Ratio: {np.mean(expert_agreement)*100:.1f}%")
    
    # Save results
    with open(args.output / "expert_comparison_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()
