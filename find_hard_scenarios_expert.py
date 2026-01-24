"""
Find hard scenarios using lidar PPO expert with noise-based difficulty measurement.

Difficulty is measured by:
1. Average reward (lower = harder)
2. Route completion (lower = harder)  
3. Sensitivity to noise (reward drop when adding noise = harder)

Usage (local):
    python find_hard_scenarios_expert.py --num_scenarios 1000

Usage (SLURM parallel):
    python find_hard_scenarios_expert.py --parallel --num_envs 10
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

import gymnasium
import gymnasium.spaces
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
gymnasium.spaces.space = gymnasium.spaces


def load_expert():
    """Load the PPO expert model."""
    from pvp.sb3.common.save_util import load_from_zip_file
    from pvp.sb3.ppo import PPO
    from pvp.sb3.ppo.policies import ActorCriticPolicy
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    import pathlib
    
    # Create a temporary env for PPO initialization
    temp_env = HumanInTheLoopEnv(config={'manual_control': False, "use_render": False})
    
    ppo_config = dict(
        policy=ActorCriticPolicy,
        n_steps=1024,
        n_epochs=20,
        learning_rate=5e-5,
        batch_size=256,
        clip_range=0.1,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=10.0,
        create_eval_env=False,
        verbose=0,
        device="auto",
        env=temp_env
    )
    model = PPO(**ppo_config)
    
    # Load expert checkpoint
    ckpt = pathlib.Path(__file__).parent / "pvp" / "experiments" / "metadrive" / "egpo" / "metadrive_pvp_20m_steps"
    if not ckpt.exists():
        # Try alternative path
        ckpt = pathlib.Path("./pvp/experiments/metadrive/egpo/metadrive_pvp_20m_steps")
    
    print(f"Loading expert from {ckpt}...")
    data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
    model.set_parameters(params, exact_match=True, device=model.device)
    temp_env.close()
    
    return model


def make_env_config(scenario_seed, crash_vehicle_penalty=5.0, crash_object_penalty=5.0, 
                     out_of_road_penalty=5.0, use_render=False):
    """Create lidar-based environment config (no image observation)."""
    return dict(
        use_render=use_render,
        manual_control=False,
        start_seed=scenario_seed,
        num_scenarios=1,
        horizon=1500,
        image_observation=False,  # Lidar only - much faster!
        crash_vehicle_done=False,
        crash_object_done=False,
        cost_to_reward=False,
        crash_vehicle_penalty=crash_vehicle_penalty,
        crash_object_penalty=crash_object_penalty,
        out_of_road_penalty=out_of_road_penalty,
    )


def evaluate_scenario_single(model, scenario_seed, noise_levels, args):
    """
    Evaluate a single scenario with different noise levels.
    Returns statistics for each noise level, including behavioral stress indicators.
    """
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    
    results = {}
    
    for eps in noise_levels:
        env = HumanInTheLoopEnv(config=make_env_config(
            scenario_seed, 
            args.crash_vehicle_penalty,
            args.crash_object_penalty,
            args.out_of_road_penalty,
            use_render=getattr(args, 'render', False)
        ))
        
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        route_completion = 0
        crash_vehicle = False
        crash_object = False
        out_of_road = False
        arrive_dest = False
        
        # Behavioral tracking (only for eps=0 baseline to analyze expert behavior)
        steering_actions = []
        accel_actions = []
        speeds = []
        min_vehicle_distance = float('inf')
        close_vehicle_count = 0  # Count how often we're near other vehicles
        vehicle_distances = []   # Track all vehicle distances for analysis
        
        while not done:
            # Get expert action
            action, _ = model.predict(obs, deterministic=True)
            
            # Track expert's original actions (before noise)
            if eps == 0:
                steering_actions.append(float(action[0]))
                accel_actions.append(float(action[1]))
            
            # NOTE: Noise temporarily disabled - relying on behavioral metrics instead
            # if eps > 0:
            #     action = action.copy()
            #     accel_noise = np.random.normal(eps * 0.3, eps * 0.5)
            #     action[1] = np.clip(action[1] + accel_noise, -1, 1)
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if isinstance(info, dict):
                route_completion = info.get('route_completion', route_completion)
                if info.get('crash_vehicle', False):
                    crash_vehicle = True
                if info.get('crash_object', False):
                    crash_object = True
                if info.get('out_of_road', False):
                    out_of_road = True
                if info.get('arrive_dest', False):
                    arrive_dest = True
                
                # Track speed and distances (eps=0 only)
                if eps == 0:
                    # Get speed from info or observation
                    speed = info.get('velocity', info.get('speed', 0))
                    if isinstance(speed, (list, np.ndarray)):
                        speed = np.linalg.norm(speed)
                    speeds.append(speed)
                    
                    # Try to get vehicle distance from environment directly
                    try:
                        if hasattr(env, 'agent') and env.agent is not None:
                            # Get surrounding vehicles from traffic manager
                            if hasattr(env, 'engine') and hasattr(env.engine, 'traffic_manager'):
                                traffic_mgr = env.engine.traffic_manager
                                if hasattr(traffic_mgr, 'vehicles'):
                                    ego_pos = env.agent.position
                                    for v in traffic_mgr.vehicles:
                                        if v != env.agent:
                                            dist = np.linalg.norm(np.array(v.position) - np.array(ego_pos))
                                            vehicle_distances.append(dist)
                                            if dist < min_vehicle_distance:
                                                min_vehicle_distance = dist
                                            if dist < 15:  # Within 15 meters = close encounter
                                                close_vehicle_count += 1
                    except Exception:
                        pass  # Silently ignore if we can't access vehicle info
        
        env.close()
        
        result = {
            'reward': episode_reward,
            'length': episode_length,
            'route_completion': route_completion,
            'crash_vehicle': crash_vehicle,
            'crash_object': crash_object,
            'out_of_road': out_of_road,
            'arrive_dest': arrive_dest,
        }
        
        # Add behavioral metrics for baseline (eps=0)
        if eps == 0 and len(steering_actions) > 1:
            steering_arr = np.array(steering_actions)
            accel_arr = np.array(accel_actions)
            
            # Calculate traffic interaction metrics
            close_encounter_rate = close_vehicle_count / max(episode_length, 1)
            avg_vehicle_dist = np.mean(vehicle_distances) if vehicle_distances else -1
            
            result['behavior'] = {
                # Steering behavior
                'steering_std': float(np.std(steering_arr)),
                'steering_abs_mean': float(np.mean(np.abs(steering_arr))),
                'steering_changes': float(np.mean(np.abs(np.diff(steering_arr)))),  # How jerky
                
                # Braking behavior
                'brake_ratio': float(np.mean(accel_arr < -0.1)),  # Fraction of time braking
                'hard_brake_ratio': float(np.mean(accel_arr < -0.5)),  # Hard braking
                'accel_std': float(np.std(accel_arr)),
                'accel_changes': float(np.mean(np.abs(np.diff(accel_arr)))),
                
                # Speed behavior
                'speed_mean': float(np.mean(speeds)) if speeds else 0,
                'speed_std': float(np.std(speeds)) if len(speeds) > 1 else 0,
                'speed_min': float(np.min(speeds)) if speeds else 0,
                
                # Traffic interaction metrics (NEW)
                'min_vehicle_distance': float(min_vehicle_distance) if min_vehicle_distance != float('inf') else -1,
                'close_encounter_rate': float(close_encounter_rate),  # How often near other vehicles
                'avg_vehicle_distance': float(avg_vehicle_dist),  # Average distance to other vehicles
                'total_close_encounters': int(close_vehicle_count),  # Total close passes
            }
        
        results[eps] = result
    
    return results


def evaluate_scenarios_parallel(model, scenario_seeds, noise_levels, args):
    """
    Evaluate multiple scenarios in parallel using SubprocVecEnv.
    Collects behavioral metrics (steering, braking, etc.) same as sequential version.
    """
    from pvp.sb3.common.vec_env import SubprocVecEnv
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    
    all_results = []
    num_envs = args.num_envs
    
    # Only process eps=0 for behavioral analysis (noise is disabled)
    eps = 0
    
    # Process scenarios in batches
    for batch_start in range(0, len(scenario_seeds), num_envs):
        batch_seeds = scenario_seeds[batch_start:batch_start + num_envs]
        batch_size = len(batch_seeds)
        
        # Create parallel envs (always disable rendering in parallel mode)
        def make_env(seed):
            def _init():
                return HumanInTheLoopEnv(config=make_env_config(
                    seed,
                    args.crash_vehicle_penalty,
                    args.crash_object_penalty, 
                    args.out_of_road_penalty,
                    use_render=False  # Must be False for parallel/headless
                ))
            return _init
        
        env_fns = [make_env(seed) for seed in batch_seeds]
        vec_env = SubprocVecEnv(env_fns)
        
        obs = vec_env.reset()
        dones = np.zeros(batch_size, dtype=bool)
        
        # Initialize tracking
        episode_rewards = np.zeros(batch_size)
        episode_lengths = np.zeros(batch_size)
        route_completions = np.zeros(batch_size)
        crash_vehicles = np.zeros(batch_size, dtype=bool)
        crash_objects = np.zeros(batch_size, dtype=bool)
        out_of_roads = np.zeros(batch_size, dtype=bool)
        arrive_dests = np.zeros(batch_size, dtype=bool)
        
        # Behavioral tracking for each env
        steering_history = [[] for _ in range(batch_size)]
        accel_history = [[] for _ in range(batch_size)]
        speed_history = [[] for _ in range(batch_size)]
        min_vehicle_distances = [float('inf')] * batch_size
        
        while not np.all(dones):
            actions, _ = model.predict(obs, deterministic=True)
            
            # Track actions for behavioral analysis
            for i in range(batch_size):
                if not dones[i]:
                    steering_history[i].append(float(actions[i, 0]))
                    accel_history[i].append(float(actions[i, 1]))
            
            obs, rewards, new_dones, infos = vec_env.step(actions)
            
            for i in range(batch_size):
                if not dones[i]:
                    episode_rewards[i] += rewards[i]
                    episode_lengths[i] += 1
                    
                    if isinstance(infos[i], dict):
                        route_completions[i] = infos[i].get('route_completion', route_completions[i])
                        if infos[i].get('crash_vehicle', False):
                            crash_vehicles[i] = True
                        if infos[i].get('crash_object', False):
                            crash_objects[i] = True
                        if infos[i].get('out_of_road', False):
                            out_of_roads[i] = True
                        if infos[i].get('arrive_dest', False):
                            arrive_dests[i] = True
                        
                        # Track speed
                        speed = infos[i].get('velocity', infos[i].get('speed', 0))
                        if isinstance(speed, (list, np.ndarray)):
                            speed = np.linalg.norm(speed)
                        speed_history[i].append(speed)
                        
                        # Track min distance to vehicles
                        vehicle_dist = infos[i].get('distance_to_vehicles', infos[i].get('min_distance', float('inf')))
                        if vehicle_dist is not None and vehicle_dist < min_vehicle_distances[i]:
                            min_vehicle_distances[i] = vehicle_dist
            
            dones = dones | new_dones
        
        vec_env.close()
        
        # Store results with behavioral metrics
        for i, seed in enumerate(batch_seeds):
            result = {
                'reward': float(episode_rewards[i]),
                'length': int(episode_lengths[i]),
                'route_completion': float(route_completions[i]),
                'crash_vehicle': bool(crash_vehicles[i]),
                'crash_object': bool(crash_objects[i]),
                'out_of_road': bool(out_of_roads[i]),
                'arrive_dest': bool(arrive_dests[i]),
            }
            
            # Compute behavioral metrics
            if len(steering_history[i]) > 1:
                steering_arr = np.array(steering_history[i])
                accel_arr = np.array(accel_history[i])
                speeds = speed_history[i]
                
                result['behavior'] = {
                    'steering_std': float(np.std(steering_arr)),
                    'steering_abs_mean': float(np.mean(np.abs(steering_arr))),
                    'steering_changes': float(np.mean(np.abs(np.diff(steering_arr)))),
                    'brake_ratio': float(np.mean(accel_arr < -0.1)),
                    'hard_brake_ratio': float(np.mean(accel_arr < -0.5)),
                    'accel_std': float(np.std(accel_arr)),
                    'accel_changes': float(np.mean(np.abs(np.diff(accel_arr)))),
                    'speed_mean': float(np.mean(speeds)) if speeds else 0,
                    'speed_std': float(np.std(speeds)) if len(speeds) > 1 else 0,
                    'speed_min': float(np.min(speeds)) if speeds else 0,
                    'min_vehicle_distance': float(min_vehicle_distances[i]) if min_vehicle_distances[i] != float('inf') else -1,
                }
            
            all_results.append({
                'scenario_seed': seed,
                'eps_results': {eps: result}
            })
    
    return all_results


def compute_difficulty_metrics(results_list, noise_levels):
    """
    Compute multiple difficulty metrics for each scenario.
    
    Metrics include:
    - Reward-based: variance, CV, AUC, decay rate
    - Robustness: success rate, failure threshold
    - Behavioral (from expert at eps=0): steering stress, braking, speed patterns
    """
    processed = []
    
    for r in results_list:
        seed = r['scenario_seed']
        eps_results = r['eps_results']
        
        # Collect rewards for all noise levels
        rewards = []
        routes = []
        successes = []
        for eps in sorted(noise_levels):
            if eps in eps_results:
                rewards.append(eps_results[eps].get('reward', 0))
                routes.append(eps_results[eps].get('route_completion', 0))
                successes.append(eps_results[eps].get('arrive_dest', False))
        
        rewards = np.array(rewards)
        routes = np.array(routes)
        
        # Get baseline (eps=0) performance
        baseline = eps_results.get(0, eps_results.get(0.0, {}))
        baseline_reward = baseline.get('reward', 0)
        baseline_route = baseline.get('route_completion', 0)
        
        # === Reward-based Metrics ===
        reward_variance = np.var(rewards) if len(rewards) > 1 else 0
        reward_mean = np.mean(rewards) if len(rewards) > 0 else 0
        reward_std = np.std(rewards) if len(rewards) > 1 else 0
        reward_cv = reward_std / reward_mean if reward_mean > 0 else 1.0
        
        # AUC
        sorted_eps = sorted([e for e in noise_levels if e in eps_results])
        if len(sorted_eps) >= 2:
            eps_rewards = [eps_results[e].get('reward', 0) for e in sorted_eps]
            reward_auc = np.trapz(eps_rewards, sorted_eps)
        else:
            reward_auc = baseline_reward
        
        # Decay rate
        if len(sorted_eps) >= 2:
            eps_array = np.array(sorted_eps)
            reward_array = np.array([eps_results[e].get('reward', 0) for e in sorted_eps])
            if np.var(eps_array) > 0:
                decay_rate = -np.cov(eps_array, reward_array)[0, 1] / np.var(eps_array)
            else:
                decay_rate = 0
        else:
            decay_rate = 0
        
        # === Robustness Metrics ===
        robustness = np.mean(successes) if successes else 0
        
        failure_threshold = max(noise_levels) + 0.1
        for eps in sorted(noise_levels):
            if eps in eps_results:
                info = eps_results[eps]
                if info.get('crash_vehicle') or info.get('crash_object') or info.get('out_of_road'):
                    if not info.get('arrive_dest', True):
                        failure_threshold = eps
                        break
        
        route_drop = baseline_route - np.mean(routes) if len(routes) > 0 else 0
        
        # === Behavioral Metrics (from eps=0 baseline) ===
        behavior = baseline.get('behavior', {})
        
        # Steering stress indicators
        steering_std = behavior.get('steering_std', 0)
        steering_abs_mean = behavior.get('steering_abs_mean', 0)
        steering_changes = behavior.get('steering_changes', 0)  # Jerkiness
        
        # Braking stress indicators
        brake_ratio = behavior.get('brake_ratio', 0)
        hard_brake_ratio = behavior.get('hard_brake_ratio', 0)
        accel_changes = behavior.get('accel_changes', 0)
        
        # Speed patterns
        speed_std = behavior.get('speed_std', 0)
        speed_min = behavior.get('speed_min', 0)
        
        # Traffic interaction metrics (NEW)
        min_vehicle_distance = behavior.get('min_vehicle_distance', -1)
        close_encounter_rate = behavior.get('close_encounter_rate', 0)
        avg_vehicle_distance = behavior.get('avg_vehicle_distance', -1)
        total_close_encounters = behavior.get('total_close_encounters', 0)
        
        # === Traffic Interaction Score (NEW) ===
        # Higher = more traffic interaction = harder scenario
        traffic_score = 0
        if close_encounter_rate > 0:
            traffic_score += close_encounter_rate * 50  # Frequent close encounters
        if total_close_encounters > 0:
            traffic_score += min(total_close_encounters / 100, 1.0)  # Cap at 100 encounters
        if min_vehicle_distance > 0 and min_vehicle_distance < 10:
            traffic_score += (10 - min_vehicle_distance) / 10  # Very close = bonus
        
        # === Behavioral Stress Score ===
        # Higher = more stressful driving behavior
        # REDUCED weight on steering to avoid false positives from curvy roads
        behavioral_stress = (
            steering_std * 1 +           # Reduced: might just be curvy road
            steering_changes * 2 +       # Jerky steering = reactive
            brake_ratio * 3 +            # Frequent braking = dangerous situations
            hard_brake_ratio * 5 +       # Hard braking = emergency
            accel_changes * 2            # Speed changes = unstable
        )
        
        # Normalize min_vehicle_distance (closer = more dangerous)
        if min_vehicle_distance > 0:
            proximity_danger = max(0, 1 - min_vehicle_distance / 20)  # Normalize to ~20m safe distance
        else:
            proximity_danger = 0
        
        # === Combined Difficulty Score ===
        # Combines reward-based, robustness, traffic interaction, and behavioral metrics
        max_eps = max(noise_levels) if max(noise_levels) > 0 else 1.0  # Avoid division by zero
        combined_difficulty = (
            0.05 * reward_cv +
            0.05 * (decay_rate / 100 if decay_rate > 0 else 0) +
            0.05 * (1 - robustness) +
            0.05 * (1 - failure_threshold / max_eps) +
            0.20 * behavioral_stress +       # Driving behavior
            0.30 * traffic_score +           # Traffic interaction (NEW - highest weight)
            0.30 * proximity_danger          # How close to other vehicles
        )
        
        # Check for any bad events
        any_bad_event = any(
            eps_results.get(eps, {}).get('crash_vehicle', False) or
            eps_results.get(eps, {}).get('crash_object', False) or
            eps_results.get(eps, {}).get('out_of_road', False)
            for eps in noise_levels
        )
        
        processed.append({
            'scenario_seed': seed,
            # Baseline metrics
            'baseline_reward': baseline_reward,
            'baseline_route_completion': baseline_route,
            # Reward variability metrics
            'reward_variance': reward_variance,
            'reward_cv': reward_cv,
            'reward_std': reward_std,
            'reward_mean': reward_mean,
            'reward_auc': reward_auc,
            'decay_rate': decay_rate,
            # Robustness metrics
            'robustness': robustness,
            'failure_threshold': failure_threshold,
            'route_drop': route_drop,
            # Behavioral metrics
            'steering_std': steering_std,
            'steering_changes': steering_changes,
            'brake_ratio': brake_ratio,
            'hard_brake_ratio': hard_brake_ratio,
            'accel_changes': accel_changes,
            'speed_std': speed_std,
            # Traffic interaction metrics (NEW)
            'min_vehicle_distance': min_vehicle_distance,
            'close_encounter_rate': close_encounter_rate,
            'avg_vehicle_distance': avg_vehicle_distance,
            'total_close_encounters': total_close_encounters,
            'traffic_score': traffic_score,
            # Scores
            'behavioral_stress': behavioral_stress,
            'proximity_danger': proximity_danger,
            # Combined
            'combined_difficulty': combined_difficulty,
            'any_bad_event': any_bad_event,
            # Raw data
            'eps_results': eps_results,
        })
    
    return processed


def save_results(processed_results, output_dir, noise_levels):
    """Save results with multiple sorting methods."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sort by different criteria
    # 1. By baseline reward (ascending = lower reward = harder)
    by_reward = sorted(processed_results, key=lambda x: x['baseline_reward'])
    
    # 2. By reward variance (descending = more unstable)
    by_variance = sorted(processed_results, key=lambda x: x['reward_variance'], reverse=True)
    
    # 3. By CV (descending = more variable relative to mean)
    by_cv = sorted(processed_results, key=lambda x: x['reward_cv'], reverse=True)
    
    # 4. By decay rate (descending = faster reward drop with noise)
    by_decay = sorted(processed_results, key=lambda x: x['decay_rate'], reverse=True)
    
    # 5. By robustness (ascending = less robust)
    by_robustness = sorted(processed_results, key=lambda x: x['robustness'])
    
    # 6. By failure threshold (ascending = fails at lower noise)
    by_threshold = sorted(processed_results, key=lambda x: x['failure_threshold'])
    
    # 7. By behavioral stress (descending = more stressful driving)
    by_stress = sorted(processed_results, key=lambda x: x['behavioral_stress'], reverse=True)
    
    # 8. By hard braking ratio (descending = more emergency braking)
    by_braking = sorted(processed_results, key=lambda x: x['hard_brake_ratio'], reverse=True)
    
    # 9. By proximity danger (descending = closer to other vehicles)
    by_proximity = sorted(processed_results, key=lambda x: x['proximity_danger'], reverse=True)
    
    # 10. By traffic score (descending = more vehicle interactions)
    by_traffic = sorted(processed_results, key=lambda x: x.get('traffic_score', 0), reverse=True)
    
    # 11. By close encounters (descending = more times near other vehicles)
    by_encounters = sorted(processed_results, key=lambda x: x.get('total_close_encounters', 0), reverse=True)
    
    # 12. By combined difficulty (descending)
    by_combined = sorted(processed_results, key=lambda x: x['combined_difficulty'], reverse=True)
    
    # Save full results
    full_results = {
        'total_scenarios': len(processed_results),
        'noise_levels': noise_levels,
        'sorting_methods': {
            'by_reward_ascending': [r['scenario_seed'] for r in by_reward],
            'by_variance_descending': [r['scenario_seed'] for r in by_variance],
            'by_cv_descending': [r['scenario_seed'] for r in by_cv],
            'by_decay_rate_descending': [r['scenario_seed'] for r in by_decay],
            'by_robustness_ascending': [r['scenario_seed'] for r in by_robustness],
            'by_failure_threshold_ascending': [r['scenario_seed'] for r in by_threshold],
            'by_behavioral_stress_descending': [r['scenario_seed'] for r in by_stress],
            'by_hard_braking_descending': [r['scenario_seed'] for r in by_braking],
            'by_proximity_danger_descending': [r['scenario_seed'] for r in by_proximity],
            'by_traffic_score_descending': [r['scenario_seed'] for r in by_traffic],
            'by_close_encounters_descending': [r['scenario_seed'] for r in by_encounters],
            'by_combined_difficulty_descending': [r['scenario_seed'] for r in by_combined],
        },
        'all_scenarios': processed_results,
    }
    
    with open(output_path / 'hard_scenarios_expert.json', 'w') as f:
        json.dump(full_results, f, indent=2)
    
    # Save detailed summary with traffic info
    def save_summary(sorted_list, filename, sort_desc):
        with open(output_path / filename, 'w') as f:
            f.write(f"Hard Scenarios Summary (sorted by {sort_desc})\n")
            f.write("=" * 170 + "\n")
            f.write(f"{'Rank':<5}{'Seed':<7}{'Reward':<9}{'Stress':<8}{'Brake%':<8}"
                    f"{'SteerChg':<9}{'MinDist':<8}{'Traffic':<9}{'Encntrs':<9}{'Combined':<10}{'BadEvt':<8}\n")
            f.write("-" * 170 + "\n")
            for i, r in enumerate(sorted_list[:100]):
                min_dist = r.get('min_vehicle_distance', -1)
                min_dist_str = f"{min_dist:.1f}" if min_dist >= 0 else "N/A"
                f.write(f"{i+1:<5}{r['scenario_seed']:<7}"
                        f"{r['baseline_reward']:<9.0f}"
                        f"{r['behavioral_stress']:<8.3f}"
                        f"{r['hard_brake_ratio']*100:<8.1f}"
                        f"{r['steering_changes']:<9.3f}"
                        f"{min_dist_str:<8}"
                        f"{r.get('traffic_score', 0):<9.3f}"
                        f"{r.get('total_close_encounters', 0):<9}"
                        f"{r['combined_difficulty']:<10.3f}"
                        f"{'Y' if r['any_bad_event'] else 'N':<8}\n")
    
    save_summary(by_reward, 'sorted_by_reward.txt', 'baseline reward (low to high)')
    save_summary(by_stress, 'sorted_by_stress.txt', 'behavioral stress (high to low)')
    save_summary(by_braking, 'sorted_by_braking.txt', 'hard braking ratio (high to low)')
    save_summary(by_proximity, 'sorted_by_proximity.txt', 'proximity danger (high to low)')
    save_summary(by_traffic, 'sorted_by_traffic.txt', 'traffic score (high to low = more traffic)')
    save_summary(by_encounters, 'sorted_by_encounters.txt', 'close encounters (high to low)')
    save_summary(by_robustness, 'sorted_by_robustness.txt', 'robustness (low to high)')
    save_summary(by_combined, 'sorted_by_combined.txt', 'combined difficulty (high to low)')
    
    print(f"\nResults saved to {output_path}/")
    print(f"  - hard_scenarios_expert.json (full results)")
    print(f"  - sorted_by_reward.txt (low reward = hard)")
    print(f"  - sorted_by_stress.txt (high behavioral stress)")
    print(f"  - sorted_by_braking.txt (high hard braking = emergencies)")
    print(f"  - sorted_by_proximity.txt (close to other vehicles)")
    print(f"  - sorted_by_traffic.txt (high traffic interaction)")
    print(f"  - sorted_by_encounters.txt (many close vehicle encounters)")
    print(f"  - sorted_by_robustness.txt (low robustness = often fails)")
    print(f"  - sorted_by_combined.txt (combined difficulty)")
    
    # Print top 10 from different criteria
    print("\n" + "=" * 80)
    print("Top 10 Most Traffic Interaction (by Traffic Score):")
    print("-" * 80)
    for i, r in enumerate(by_traffic[:10]):
        print(f"  {i+1}. Seed {r['scenario_seed']}: "
              f"TrafficScore={r.get('traffic_score', 0):.2f}, "
              f"CloseEncounters={r.get('total_close_encounters', 0)}, "
              f"MinDist={r.get('min_vehicle_distance', -1):.1f}m")
    
    print("\n" + "=" * 80)
    print("Top 10 Hardest Scenarios (by Combined Difficulty):")
    print("-" * 80)
    for i, r in enumerate(by_combined[:10]):
        print(f"  {i+1}. Seed {r['scenario_seed']}: "
              f"Reward={r['baseline_reward']:.0f}, "
              f"Traffic={r.get('traffic_score', 0):.2f}, "
              f"Stress={r['behavioral_stress']:.2f}, "
              f"Combined={r['combined_difficulty']:.2f}")
    
    print("\nTop 10 Most Stressful (by Expert Behavior):")
    print("-" * 80)
    for i, r in enumerate(by_stress[:10]):
        print(f"  {i+1}. Seed {r['scenario_seed']}: "
              f"Stress={r['behavioral_stress']:.2f}, "
              f"SteerChg={r['steering_changes']:.3f}, "
              f"HardBrake={r['hard_brake_ratio']*100:.0f}%")
    
    return by_combined


def main():
    parser = argparse.ArgumentParser(description="Find hard scenarios using PPO expert with noise")
    parser.add_argument("--start_seed", type=int, default=1000, help="Start scenario seed")
    parser.add_argument("--num_scenarios", type=int, default=1000, help="Number of scenarios to test")
    parser.add_argument("--parallel", action="store_true", help="Use parallel evaluation")
    parser.add_argument("--num_envs", type=int, default=10, help="Number of parallel environments")
    parser.add_argument("--output", type=str, default="./results/expert_difficulty", help="Output directory")
    parser.add_argument("--save_interval", type=int, default=50, help="Save results every N scenarios")
    parser.add_argument("--noise_levels", type=str, default="0",
                        help="Comma-separated noise levels (currently disabled, only eps=0 is used for behavioral analysis)")
    # Penalty parameters
    parser.add_argument("--crash_vehicle_penalty", type=float, default=5.0)
    parser.add_argument("--crash_object_penalty", type=float, default=5.0)
    parser.add_argument("--out_of_road_penalty", type=float, default=5.0)
    parser.add_argument("--render", action="store_true", help="Enable rendering (only works in sequential mode)")
    args = parser.parse_args()
    
    # Force render=False in parallel mode (no display on headless servers)
    if args.parallel and args.render:
        print("WARNING: --render is not supported in parallel mode. Disabling rendering.")
        args.render = False
    
    start_time = time.time()
    
    # Parse noise levels
    noise_levels = [float(x) for x in args.noise_levels.split(',')]
    print(f"Noise levels to test: {noise_levels}")
    
    # Load expert
    print("Loading PPO expert model...")
    model = load_expert()
    print("Expert loaded!")
    
    # Generate scenario seeds
    scenario_seeds = list(range(args.start_seed, args.start_seed + args.num_scenarios))
    
    print(f"\nEvaluating {len(scenario_seeds)} scenarios")
    print(f"Mode: {'Parallel' if args.parallel else 'Sequential'}")
    if args.parallel:
        print(f"Parallel environments: {args.num_envs}")
    print("=" * 60)
    
    all_results = []
    
    if args.parallel:
        # Process in chunks for intermediate saving
        for chunk_start in range(0, len(scenario_seeds), args.save_interval):
            chunk_end = min(chunk_start + args.save_interval, len(scenario_seeds))
            chunk_seeds = scenario_seeds[chunk_start:chunk_end]
            
            print(f"\nProcessing scenarios {chunk_seeds[0]}-{chunk_seeds[-1]} "
                  f"({chunk_start+1}-{chunk_end}/{len(scenario_seeds)})...")
            
            chunk_results = evaluate_scenarios_parallel(model, chunk_seeds, noise_levels, args)
            all_results.extend(chunk_results)
            
            # Compute metrics and save intermediate results
            processed = compute_difficulty_metrics(all_results, noise_levels)
            save_results(processed, args.output, noise_levels)
            
            elapsed = time.time() - start_time
            print(f"  Progress: {len(all_results)}/{len(scenario_seeds)}, Time: {elapsed/60:.1f}min")
    else:
        # Sequential processing with real-time debug output
        for i, seed in enumerate(scenario_seeds):
            eps_results = evaluate_scenario_single(model, seed, noise_levels, args)
            all_results.append({
                'scenario_seed': seed,
                'eps_results': eps_results,
            })
            
            # Compute metrics for current results
            processed = compute_difficulty_metrics(all_results, noise_levels)
            current = processed[-1]  # Get metrics for just-completed scenario
            
            # Print current scenario's metrics
            behavior = eps_results.get(0, {}).get('behavior', {})
            print(f"\n[{i+1}/{len(scenario_seeds)}] Seed {seed}:")
            print(f"  Reward={current['baseline_reward']:.0f}, Route={current['baseline_route_completion']*100:.0f}%")
            print(f"  SteerSTD={behavior.get('steering_std', 0):.3f}, SteerChg={behavior.get('steering_changes', 0):.3f}")
            print(f"  Brake={behavior.get('brake_ratio', 0)*100:.1f}%, HardBrake={behavior.get('hard_brake_ratio', 0)*100:.1f}%")
            # Traffic interaction info (NEW)
            print(f"  CloseEncounters={current.get('total_close_encounters', 0)}, MinVehicleDist={current.get('min_vehicle_distance', -1):.1f}m")
            print(f"  TrafficScore={current.get('traffic_score', 0):.3f}, Stress={current['behavioral_stress']:.3f}, Combined={current['combined_difficulty']:.3f}")
            
            # Print current top 5 ranking by traffic score (most traffic interaction)
            by_traffic = sorted(processed, key=lambda x: x.get('traffic_score', 0), reverse=True)
            print(f"  >> Top5 by Traffic: {[r['scenario_seed'] for r in by_traffic[:5]]}")
            
            # Also show top 5 by combined difficulty
            by_combined = sorted(processed, key=lambda x: x['combined_difficulty'], reverse=True)
            print(f"  >> Top5 by Combined: {[r['scenario_seed'] for r in by_combined[:5]]}")
            
            # Save intermediate results periodically
            if (i + 1) % args.save_interval == 0:
                print(f"\n--- Saving intermediate results ({i+1} scenarios) ---")
                save_results(processed, args.output, noise_levels)
    
    # Final save
    print(f"\n\n{'='*60}")
    print("FINAL RESULTS")
    print("=" * 60)
    
    processed = compute_difficulty_metrics(all_results, noise_levels)
    by_difficulty = save_results(processed, args.output, noise_levels)
    
    # Summary statistics
    elapsed_time = time.time() - start_time
    print(f"\nSummary:")
    print(f"  Total scenarios tested: {len(all_results)}")
    print(f"  Total time: {elapsed_time/60:.1f} minutes")
    print(f"  Time per scenario: {elapsed_time/len(all_results):.2f} seconds")
    
    # Statistics
    rewards = [r['baseline_reward'] for r in processed]
    routes = [r['baseline_route_completion'] for r in processed]
    sensitivities = [r['noise_sensitivity'] for r in processed]
    
    print(f"\n  Reward stats: min={min(rewards):.1f}, max={max(rewards):.1f}, mean={np.mean(rewards):.1f}")
    print(f"  Route completion stats: min={min(routes)*100:.0f}%, max={max(routes)*100:.0f}%, mean={np.mean(routes)*100:.0f}%")
    print(f"  Sensitivity stats: min={min(sensitivities):.3f}, max={max(sensitivities):.3f}, mean={np.mean(sensitivities):.3f}")


if __name__ == "__main__":
    main()
