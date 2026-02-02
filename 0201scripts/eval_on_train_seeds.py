"""
Evaluate models on the 200 TRAINING seeds (not evaluation seeds).
This helps us understand the training data distribution performance.

Supports:
1. Lidar PPO Expert (state-based observation)
2. IQL/BC models (image-based observation)
"""

import os
os.environ["SDL_VIDEODRIVER"] = "offscreen"
os.environ["PYOPENGL_PLATFORM"] = "egl"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

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
import psutil

# Suppress metadrive logging
import logging
logging.getLogger('metadrive').setLevel(logging.WARNING)
logging.getLogger('panda3d').setLevel(logging.WARNING)


# 200 TRAINING seeds (from generate_bc_data_sequential.py)
HARD_200_TRAIN_SEEDS = [
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


def make_env_config(use_image=True, daytime="08:30", use_test_seeds=False, use_simple_seeds=False, simple_start_seed=5000, use_original_config=False):
    """Create environment config."""
    # Training seeds are in range 0-999, Test seeds are in range 1000-1999
    # Simple seeds are sequential starting from simple_start_seed
    if use_simple_seeds:
        start_seed = simple_start_seed
        num_scenarios = 1000
    elif use_test_seeds:
        start_seed = 1000
        num_scenarios = 1000
    else:
        start_seed = 0
        num_scenarios = 1000
    
    if use_original_config:
        # MINIMAL config matching run 8o6jrf5k
        # Only set absolutely necessary params, let env use its defaults
        # Env defaults: traffic_density=0.06, out_of_route_done=True
        config = dict(
            use_render=False,
            manual_control=False,
            num_scenarios=num_scenarios,
            start_seed=start_seed,
            horizon=1500,
            # These are from eval_hard_scenarios_parallel.py
            crash_vehicle_done=False,
            crash_object_done=False,
            cost_to_reward=False,
            crash_vehicle_penalty=5.0,
            crash_object_penalty=5.0,
            out_of_road_penalty=5.0,
            # Use daytime if specified, otherwise use env default
            daytime=daytime,
            # NO traffic_density, NO random_traffic - use env defaults (0.06)!
        )
    else:
        # Current training data config (higher traffic)
        config = dict(
            use_render=False,
            manual_control=False,
            num_scenarios=num_scenarios,
            start_seed=start_seed,
            horizon=1500,
            crash_vehicle_done=False,
            crash_object_done=False,
            cost_to_reward=False,
            crash_vehicle_penalty=5.0,
            crash_object_penalty=5.0,
            out_of_road_penalty=5.0,
            daytime=daytime,
            traffic_density=0.1,
            random_traffic=True,
        )
    
    if use_image:
        from metadrive.component.sensors.rgb_camera import RGBCamera
        config.update(dict(
            image_observation=True,
            vehicle_config=dict(image_source="rgb_camera"),
            sensors={"rgb_camera": (RGBCamera, 84, 84)},
            stack_size=3,
        ))
    else:
        config['image_observation'] = False
    
    return config


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
    
    # Use absolute path to avoid issues when script is moved
    ckpt = Path("/p0/user/caihy/pvp/pvp/experiments/metadrive/egpo/metadrive_pvp_20m_steps.zip")
    
    print(f"Loading lidar expert from: {ckpt}")
    if ckpt.exists():
        data, params, pytorch_variables = load_from_zip_file(ckpt, device=expert.device, print_system_info=False)
        expert.set_parameters(params, exact_match=True, device=expert.device)
        print(f"Lidar expert loaded! Device: {expert.device}")
    else:
        raise FileNotFoundError(f"Lidar expert not found at {ckpt}")
    
    temp_env.close()
    return expert


def load_iql_model(checkpoint_path, device="auto"):
    """Load an IQL model from checkpoint."""
    from pvp.sb3.td3.policies import TD3Policy
    from pvp.sb3.sac.our_features_extractor import OurFeaturesExtractorCNN as OurFeaturesExtractor
    from pvp.sb3.td3.iql import IQL
    from pvp.sb3.common.save_util import load_from_zip_file
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    
    # Create a temporary image env for model loading
    env_config = make_env_config(use_image=True, daytime="08:30")
    temp_env = HumanInTheLoopEnv(config=env_config)
    
    policy_kwargs = dict(
        features_extractor_class=OurFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=275),
        share_features_extractor=False,
        net_arch=[256,]
    )
    
    model = IQL(
        policy=TD3Policy,
        env=temp_env,
        learning_rate=1e-4,
        policy_kwargs=policy_kwargs,
        buffer_size=1000,
        verbose=0,
        device=device,
    )
    
    print(f"Loading IQL model from: {checkpoint_path}")
    data, params, pytorch_variables = load_from_zip_file(
        checkpoint_path, device=model.device, print_system_info=False
    )
    model.set_parameters(params, exact_match=False, device=model.device)
    print(f"IQL model loaded! Device: {model.device}")
    
    temp_env.close()
    return model


def evaluate_sequential(model, seeds, env_config, model_name="model"):
    """Evaluate the model sequentially on all seeds."""
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    
    print(f"Creating environment for {model_name}...")
    env = HumanInTheLoopEnv(config=env_config)
    
    all_results = []
    start_time = time.time()
    
    for seed_idx, seed in enumerate(seeds):
        # Reset with specific seed
        obs = env.reset(seed=seed)
        if isinstance(obs, tuple):
            obs = obs[0]
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        # Tracking
        had_crash_vehicle = False
        had_crash_object = False
        had_out_of_road = False
        had_bad_event = False
        arrive_dest = False
        route_completion = 0.0
        
        # Episodic cost and reward breakdown
        episode_cost = 0.0
        crash_vehicle_count = 0
        crash_object_count = 0
        out_of_road_count = 0
        
        # Reward component tracking
        total_velocity = 0.0
        total_step_rewards = 0.0
        total_crash_penalty = 0.0
        total_out_of_road_penalty = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if isinstance(obs, tuple):
                obs = obs[0]
            
            episode_reward += reward
            episode_length += 1
            
            # Track cost
            step_cost = info.get('cost', 0.0)
            episode_cost += step_cost
            
            # Track reward components (estimate from env config)
            # velocity is in m/s, step_reward ≈ driving_reward * velocity * dt + speed_reward - penalties
            velocity = info.get('velocity', 0.0)
            if isinstance(velocity, (list, np.ndarray)):
                velocity = np.linalg.norm(velocity)
            total_velocity += velocity
            total_step_rewards += info.get('step_reward', reward)
            
            # Track bad events (cumulative count, not just binary)
            if info.get('crash_vehicle', False):
                had_crash_vehicle = True
                crash_vehicle_count += 1
                total_crash_penalty += 5.0  # crash_vehicle_penalty
            if info.get('crash_object', False):
                had_crash_object = True
                crash_object_count += 1
                total_crash_penalty += 5.0  # crash_object_penalty
            if info.get('out_of_road', False):
                had_out_of_road = True
                out_of_road_count += 1
                total_out_of_road_penalty += 5.0  # out_of_road_penalty
            if had_crash_vehicle or had_crash_object or had_out_of_road:
                had_bad_event = True
            if info.get('arrive_dest', False):
                arrive_dest = True
            route_completion = max(route_completion, info.get('route_completion', 0.0))
        
        # Route completion no bad event: full RC if no bad, else scaled down
        route_completion_no_bad = route_completion if not had_bad_event else 0.0
        
        # Estimate reward components (based on env config)
        avg_velocity = total_velocity / episode_length if episode_length > 0 else 0.0
        # driving_reward ≈ total_distance * driving_reward_per_meter (roughly)
        # We estimate based on total_step_rewards which includes driving + speed - penalties
        estimated_driving_reward = total_step_rewards + total_crash_penalty + total_out_of_road_penalty
        
        result = {
            'seed': seed,
            'reward': float(episode_reward),
            'length': int(episode_length),
            'route_completion': float(route_completion),
            'route_completion_no_bad': float(route_completion_no_bad),
            'crash_vehicle_rate': float(had_crash_vehicle),
            'crash_object_rate': float(had_crash_object),
            'out_of_road_rate': float(had_out_of_road),
            'any_bad_event_rate': float(had_bad_event),
            'success_rate': float(arrive_dest),
            'success_no_bad_event_rate': float(arrive_dest and not had_bad_event),
            # Cost metrics
            'episode_cost': float(episode_cost),
            'crash_vehicle_count': int(crash_vehicle_count),
            'crash_object_count': int(crash_object_count),
            'out_of_road_count': int(out_of_road_count),
            # Reward component breakdown
            'avg_velocity': float(avg_velocity),
            'total_step_rewards': float(total_step_rewards),
            'total_crash_penalty': float(total_crash_penalty),
            'total_out_of_road_penalty': float(total_out_of_road_penalty),
            'estimated_driving_reward': float(estimated_driving_reward),
        }
        all_results.append(result)
        
        # Progress logging (every 10 seeds or first seed)
        elapsed = time.time() - start_time
        if (seed_idx + 1) % 10 == 0 or seed_idx == 0:
            avg_reward = np.mean([r['reward'] for r in all_results])
            avg_success = np.mean([r['success_rate'] for r in all_results])
            avg_success_no_bad = np.mean([r['success_no_bad_event_rate'] for r in all_results])
            avg_crash = np.mean([r['any_bad_event_rate'] for r in all_results])
            avg_rc = np.mean([r['route_completion'] for r in all_results])
            mem = psutil.Process().memory_info().rss / 1024 / 1024
            
            seeds_per_sec = (seed_idx + 1) / elapsed if elapsed > 0 else 0
            remaining = len(seeds) - (seed_idx + 1)
            eta = remaining / seeds_per_sec if seeds_per_sec > 0 else 0
            
            avg_rc_no_bad = np.mean([r['route_completion_no_bad'] for r in all_results])
            avg_cost = np.mean([r['episode_cost'] for r in all_results])
            # Reward decomposition
            avg_crash_penalty = np.mean([r['total_crash_penalty'] for r in all_results])
            avg_oor_penalty = np.mean([r['total_out_of_road_penalty'] for r in all_results])
            avg_driving_reward = np.mean([r['estimated_driving_reward'] for r in all_results])
            print(f"  [{seed_idx+1}/{len(seeds)}] {elapsed:.1f}s | Mem={mem:.0f}MB | "
                  f"R={avg_reward:.1f} Succ={avg_success:.0%} SuccNoBad={avg_success_no_bad:.0%} "
                  f"RCNoBad={avg_rc_no_bad:.0%} | "
                  f"CrashP={avg_crash_penalty:.1f} OorP={avg_oor_penalty:.1f} DrivR={avg_driving_reward:.1f} | "
                  f"ETA={eta:.0f}s", flush=True)
    
    env.close()
    return all_results


def print_summary(results, model_name):
    """Print evaluation summary."""
    rewards = [r['reward'] for r in results]
    success_rates = [r['success_rate'] for r in results]
    success_no_bad = [r['success_no_bad_event_rate'] for r in results]
    route_completion = [r['route_completion'] for r in results]
    crash_vehicle = [r['crash_vehicle_rate'] for r in results]
    any_bad = [r['any_bad_event_rate'] for r in results]
    out_of_road = [r['out_of_road_rate'] for r in results]
    
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} RESULTS on 200 TRAINING SEEDS")
    print(f"{'='*60}")
    print(f"Reward: mean={np.mean(rewards):.1f}, std={np.std(rewards):.1f}")
    print(f"Route Completion: {np.mean(route_completion)*100:.1f}%")
    print(f"Success Rate: {np.mean(success_rates)*100:.1f}%")
    print(f"Success Rate (no bad event): {np.mean(success_no_bad)*100:.1f}%")
    print(f"Crash Vehicle Rate: {np.mean(crash_vehicle)*100:.1f}%")
    print(f"Out of Road Rate: {np.mean(out_of_road)*100:.1f}%")
    print(f"Any Bad Event Rate: {np.mean(any_bad)*100:.1f}%")


# 200 EVALUATION/TEST seeds (from eval_bc_checkpoint.py - different from training!)
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on 200 TRAINING or TEST seeds")
    parser.add_argument("--model", type=str, required=True, choices=["lidar", "iql"],
                        help="Model type to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (required for IQL)")
    parser.add_argument("--num_seeds", type=int, default=200,
                        help="Number of seeds to evaluate")
    parser.add_argument("--use_test_seeds", action="store_true",
                        help="Use TEST/EVAL seeds instead of TRAINING seeds")
    parser.add_argument("--use_simple_seeds", action="store_true",
                        help="Use simple sequential seeds (like start_seed 5000)")
    parser.add_argument("--start_seed", type=int, default=5000,
                        help="Start seed for simple sequential seeds (default: 5000)")
    parser.add_argument("--use_original_config", action="store_true",
                        help="Use original config from run 8o6jrf5k (traffic_density=0.06, daytime=06:10)")
    parser.add_argument("--daytime", type=str, default="08:30",
                        help="Daytime setting")
    parser.add_argument("--output_dir", type=str, default="/p0/user/caihy/eval_train_seeds",
                        help="Output directory")
    parser.add_argument("--wandb", action="store_true",
                        help="Log results to wandb")
    parser.add_argument("--wandb_project", type=str, default="iql-eval-train-seeds",
                        help="Wandb project name")
    parser.add_argument("--exp_name", type=str, default="eval",
                        help="Experiment name for wandb")
    args = parser.parse_args()
    
    if args.use_simple_seeds:
        seed_type = f"SIMPLE (start={args.start_seed})"
        seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    elif args.use_test_seeds:
        seed_type = "TEST (HARD)"
        seeds = TOP_200_EVAL_SEEDS[:args.num_seeds]
    else:
        seed_type = "TRAINING (HARD)"
        seeds = HARD_200_TRAIN_SEEDS[:args.num_seeds]
    
    print("=" * 70)
    print(f"EVALUATION ON {args.num_seeds} {seed_type} SEEDS")
    print("=" * 70)
    print(f"Model: {args.model}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Daytime: {args.daytime}")
    print(f"Seed type: {seed_type}")
    print(f"Seeds: {seeds[:5]}...{seeds[-5:]}" if len(seeds) > 10 else f"Seeds: {seeds}")
    print("=" * 70)
    
    if args.model == "lidar":
        # Lidar expert uses state-based observation
        print("\n[1] Loading lidar PPO expert...")
        model = load_lidar_expert(device="auto")
        env_config = make_env_config(
            use_image=False, daytime=args.daytime, 
            use_test_seeds=args.use_test_seeds,
            use_simple_seeds=args.use_simple_seeds,
            simple_start_seed=args.start_seed,
            use_original_config=args.use_original_config
        )
        model_name = "lidar_ppo_expert"
    else:
        # IQL uses image-based observation
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for IQL model")
        print("\n[1] Loading IQL model...")
        model = load_iql_model(args.checkpoint, device="auto")
        env_config = make_env_config(
            use_image=True, daytime=args.daytime,
            use_test_seeds=args.use_test_seeds,
            use_simple_seeds=args.use_simple_seeds,
            simple_start_seed=args.start_seed,
            use_original_config=args.use_original_config
        )
        model_name = f"iql_{Path(args.checkpoint).stem}"
    
    # Build comprehensive env config dict for logging
    # Determine seed range for clear logging
    if args.use_simple_seeds:
        seed_range = f"[{args.start_seed}, {args.start_seed + args.num_seeds})"
    elif args.use_test_seeds:
        seed_range = "[1000, 2000)"
    else:
        seed_range = "[0, 1000)"
    
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
        # === ENVIRONMENT CONFIGURATION ===
        'config/use_original_config': args.use_original_config,
        'config/image_observation': env_config.get('image_observation'),
        'config/traffic_density': env_config.get('traffic_density', 'NOT_SET_default_0.06'),
        'config/random_traffic': env_config.get('random_traffic', 'NOT_SET_default'),
        'config/daytime': env_config.get('daytime', 'NOT_SET_default'),
        'config/crash_vehicle_done': env_config.get('crash_vehicle_done'),
        'config/crash_object_done': env_config.get('crash_object_done'),
        'config/cost_to_reward': env_config.get('cost_to_reward'),
        'config/crash_vehicle_penalty': env_config.get('crash_vehicle_penalty'),
        'config/crash_object_penalty': env_config.get('crash_object_penalty'),
        'config/out_of_road_penalty': env_config.get('out_of_road_penalty'),
        'config/start_seed': env_config.get('start_seed'),
        'config/num_scenarios': env_config.get('num_scenarios'),
        'config/horizon': env_config.get('horizon'),
        'config/stack_size': env_config.get('stack_size'),
        'config/out_of_route_done': env_config.get('out_of_route_done', 'NOT_SET'),
        'config/driving_reward': env_config.get('driving_reward', 'NOT_SET'),
        'config/speed_reward': env_config.get('speed_reward', 'NOT_SET'),
        'config/use_lateral_reward': env_config.get('use_lateral_reward', 'NOT_SET'),
        'config/model_type': args.model,
        'config/checkpoint': args.checkpoint if args.checkpoint else 'N/A',
        'config/num_seeds': len(seeds),
    }
    
    print(f"\n[2] Environment config (FULL):")
    for key, val in env_config_log.items():
        print(f"    {key.replace('config/', '')}={val}")
    
    print(f"\n[3] Starting evaluation on {len(seeds)} training seeds...")
    start_time = time.time()
    results = evaluate_sequential(model, seeds, env_config, model_name)
    eval_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} RESULTS on {len(results)} {seed_type} SEEDS")
    print(f"{'='*60}")
    
    rewards = [r['reward'] for r in results]
    success_rates = [r['success_rate'] for r in results]
    success_no_bad = [r['success_no_bad_event_rate'] for r in results]
    route_completion = [r['route_completion'] for r in results]
    route_completion_no_bad = [r['route_completion_no_bad'] for r in results]
    crash_vehicle = [r['crash_vehicle_rate'] for r in results]
    any_bad = [r['any_bad_event_rate'] for r in results]
    out_of_road = [r['out_of_road_rate'] for r in results]
    episode_costs = [r['episode_cost'] for r in results]
    crash_vehicle_counts = [r['crash_vehicle_count'] for r in results]
    crash_object_counts = [r['crash_object_count'] for r in results]
    out_of_road_counts = [r['out_of_road_count'] for r in results]
    
    print(f"Reward: mean={np.mean(rewards):.1f}, std={np.std(rewards):.1f}")
    print(f"Route Completion: {np.mean(route_completion)*100:.1f}%")
    print(f"Route Completion (no bad): {np.mean(route_completion_no_bad)*100:.1f}%")
    print(f"Success Rate: {np.mean(success_rates)*100:.1f}%")
    print(f"Success Rate (no bad event): {np.mean(success_no_bad)*100:.1f}%")
    print(f"--- Safety Metrics ---")
    print(f"Crash Vehicle Rate: {np.mean(crash_vehicle)*100:.1f}%")
    print(f"Out of Road Rate: {np.mean(out_of_road)*100:.1f}%")
    print(f"Any Bad Event Rate: {np.mean(any_bad)*100:.1f}%")
    print(f"--- Episode Cost ---")
    print(f"Episode Cost: mean={np.mean(episode_costs):.2f}, std={np.std(episode_costs):.2f}")
    print(f"Crash Vehicle Count: mean={np.mean(crash_vehicle_counts):.2f}")
    print(f"Crash Object Count: mean={np.mean(crash_object_counts):.2f}")
    print(f"Out of Road Count: mean={np.mean(out_of_road_counts):.2f}")
    print(f"--- Reward Decomposition ---")
    avg_velocity = np.mean([r['avg_velocity'] for r in results])
    total_crash_penalty = np.mean([r['total_crash_penalty'] for r in results])
    total_oor_penalty = np.mean([r['total_out_of_road_penalty'] for r in results])
    total_driving_reward = np.mean([r['estimated_driving_reward'] for r in results])
    print(f"Avg Velocity: {avg_velocity:.2f} m/s")
    print(f"Crash Penalty Mean: {total_crash_penalty:.1f}")
    print(f"Out of Road Penalty Mean: {total_oor_penalty:.1f}")
    print(f"Estimated Driving Reward Mean: {total_driving_reward:.1f}")
    print(f"\nTotal evaluation time: {eval_time/60:.1f} minutes")
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.use_simple_seeds:
        seeds_suffix = f"simple_seeds_{args.start_seed}"
    elif args.use_test_seeds:
        seeds_suffix = "test_hard_seeds"
    else:
        seeds_suffix = "train_hard_seeds"
    result_file = output_path / f"{model_name}_{seeds_suffix}_{args.num_seeds}.json"
    with open(result_file, 'w') as f:
        json.dump({
            'model': model_name,
            'checkpoint': args.checkpoint,
            'num_seeds': len(seeds),
            'daytime': args.daytime,
            'seeds_type': seed_type.lower(),
            'summary': {
                'reward_mean': float(np.mean([r['reward'] for r in results])),
                'success_rate': float(np.mean([r['success_rate'] for r in results])),
                'success_no_bad_event_rate': float(np.mean([r['success_no_bad_event_rate'] for r in results])),
                'route_completion': float(np.mean([r['route_completion'] for r in results])),
                'any_bad_event_rate': float(np.mean([r['any_bad_event_rate'] for r in results])),
            },
            'per_seed_results': results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {result_file}")
    
    # Log to wandb if enabled
    if args.wandb:
        try:
            import wandb as wandb_module
            from pvp.utils.utils import get_time_str
            import uuid
            
            # Merge env config with results for wandb
            wandb_config = {**env_config_log, 'model': model_name, 'checkpoint': args.checkpoint}
            
            trial_name = f"{args.exp_name}_{model_name}_{seeds_suffix}_{get_time_str()}_{uuid.uuid4().hex[:8]}"
            wandb_run = wandb_module.init(
                project=args.wandb_project,
                entity="victorique",
                name=trial_name,
                config=wandb_config,
            )
            
            # Log summary metrics
            wandb_run.log({
                'eval/reward_mean': float(np.mean(rewards)),
                'eval/reward_std': float(np.std(rewards)),
                'eval/success_rate': float(np.mean(success_rates)),
                'eval/success_rate_no_bad': float(np.mean(success_no_bad)),
                'eval/route_completion': float(np.mean(route_completion)),
                'eval/route_completion_no_bad': float(np.mean(route_completion_no_bad)),
                'eval/crash_vehicle_rate': float(np.mean(crash_vehicle)),
                'eval/out_of_road_rate': float(np.mean(out_of_road)),
                'eval/any_bad_event_rate': float(np.mean(any_bad)),
                'eval/episode_cost_mean': float(np.mean(episode_costs)),
                'eval/num_seeds': len(seeds),
                'eval/seeds_type': seed_type,
                # Reward component breakdown
                'reward/avg_velocity': float(np.mean([r['avg_velocity'] for r in results])),
                'reward/total_step_rewards_mean': float(np.mean([r['total_step_rewards'] for r in results])),
                'reward/crash_penalty_mean': float(np.mean([r['total_crash_penalty'] for r in results])),
                'reward/out_of_road_penalty_mean': float(np.mean([r['total_out_of_road_penalty'] for r in results])),
                'reward/estimated_driving_reward_mean': float(np.mean([r['estimated_driving_reward'] for r in results])),
            })
            
            wandb_run.finish()
            print(f"Wandb logged: {trial_name}")
        except Exception as e:
            print(f"WARNING: Failed to log to wandb: {e}")
    
    print("Done!")


if __name__ == "__main__":
    main()
