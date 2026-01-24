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
        num_scenarios=1,  # Only this seed
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
    ckpt = script_dir / "pvp" / "experiments" / "metadrive" / "egpo" / "metadrive_pvp_20m_steps"
    
    if ckpt.exists():
        data, params, pytorch_variables = load_from_zip_file(ckpt, device=expert.device, print_system_info=False)
        expert.set_parameters(params, exact_match=True, device=expert.device)
    
    temp_env.close()
    return expert


def evaluate_on_seed(model, seed, num_episodes=1, use_render=False, expert_model=None):
    """Evaluate model on a specific scenario seed with comprehensive metrics."""
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    
    env = HumanInTheLoopEnv(config=make_env_config(seed, use_image=True, use_render=use_render))
    
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
    }
    
    for ep in range(num_episodes):
        obs = env.reset()
        
        done = False
        episode_reward = 0
        episode_length = 0
        episode_actions = []
        episode_expert_diffs = []
        episode_steering_diffs = []
        episode_accel_diffs = []
        crash_count = 0
        had_bad_event = False
        
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
            
            # Track crashes per step
            if info.get('crash_vehicle', False) or info.get('crash_object', False):
                crash_count += 1
            if info.get('crash_vehicle', False) or info.get('crash_object', False) or info.get('out_of_road', False):
                had_bad_event = True
        
        # Episode-level metrics
        route_completion = info.get('route_completion', 0)
        arrive_dest = info.get('arrive_dest', False)
        crash_vehicle = info.get('crash_vehicle', False)
        crash_object = info.get('crash_object', False)
        out_of_road = info.get('out_of_road', False)
        
        results['rewards'].append(episode_reward)
        results['lengths'].append(episode_length)
        results['route_completions'].append(route_completion)
        results['crash_vehicles'].append(crash_vehicle)
        results['crash_objects'].append(crash_object)
        results['out_of_roads'].append(out_of_road)
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
        
        # Expert comparison metrics
        if len(episode_expert_diffs) > 0:
            results['expert_action_diff_l2'].append(np.mean(episode_expert_diffs))
            results['expert_steering_diff'].append(np.mean(episode_steering_diffs))
            results['expert_accel_diff'].append(np.mean(episode_accel_diffs))
            results['expert_agreement_ratio'].append(np.mean(np.array(episode_expert_diffs) < 0.3))
    
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
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on hard scenarios")
    parser.add_argument("--model", type=str, required=True, choices=["iql", "td3", "td3bc2", "both", "all"],
                        help="Which model to evaluate: iql, td3, td3bc2, both (iql+td3), all (iql+td3+td3bc2)")
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
    
    # Create a temporary environment for model initialization
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    temp_env = HumanInTheLoopEnv(config=make_env_config(seeds[0], use_image=True))
    
    # Load models
    models_to_eval = {}
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    if args.model in ["iql", "both", "all"]:
        iql_path = script_dir / args.iql_checkpoint
        if iql_path.exists():
            print(f"Loading IQL model from {iql_path}...")
            models_to_eval["iql"] = load_iql_model(iql_path, temp_env)
            print("IQL model loaded!")
        else:
            print(f"ERROR: IQL checkpoint not found at {iql_path}")
            return
    
    if args.model in ["td3", "both", "all"]:
        td3_path = script_dir / args.td3_checkpoint
        if td3_path.exists():
            print(f"Loading TD3BC1 model from {td3_path}...")
            models_to_eval["td3bc1"] = load_td3_model(td3_path, temp_env)
            print("TD3BC1 model loaded!")
        else:
            print(f"ERROR: TD3 checkpoint not found at {td3_path}")
            return
    
    if args.model in ["td3bc2", "all"]:
        td3bc2_path = script_dir / args.td3bc2_checkpoint
        if td3bc2_path.exists():
            print(f"Loading TD3BC2 model from {td3bc2_path}...")
            models_to_eval["td3bc2"] = load_td3_model(td3bc2_path, temp_env)
            print("TD3BC2 model loaded!")
        else:
            print(f"ERROR: TD3BC2 checkpoint not found at {td3bc2_path}")
            return
    
    temp_env.close()
    
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
                expert_model=expert_model
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
    
    print(f"\nResults saved to {output_path}/")


if __name__ == "__main__":
    main()
