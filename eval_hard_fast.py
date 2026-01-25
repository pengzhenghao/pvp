#!/usr/bin/env python
"""
Fast parallel evaluation using multiprocessing.Pool.
Each worker process evaluates multiple seeds sequentially with reset(seed=seed).
This avoids the slow SubprocVecEnv creation overhead.
"""
import os
import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count

# Set environment variables BEFORE any imports
os.environ['SDL_VIDEODRIVER'] = 'offscreen'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['DISPLAY'] = ''

# Hard scenarios (top 200)
HARD_SEEDS = [
    1832, 1683, 1786, 1081, 1175, 1111, 1946, 1839, 1821, 1802,
    1213, 1466, 1604, 1911, 1612, 1650, 1831, 1497, 1364, 1886,
    1569, 1047, 1897, 1467, 1504, 1399, 1303, 1793, 1258, 1189,
    1324, 1306, 1248, 1341, 1720, 1568, 1394, 1244, 1711, 1222,
    1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009,
    1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019,
    1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029,
    1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039,
    1040, 1041, 1042, 1043, 1044, 1045, 1046, 1048, 1049, 1050,
    1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060,
    1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070,
    1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080,
    1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091,
    1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101,
    1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1112,
    1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122,
    1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132,
    1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142,
    1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152,
    1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162,
]


def evaluate_seeds_worker(args):
    """
    Worker function: evaluate a batch of seeds in a single process.
    Creates one environment, evaluates all seeds using reset(seed=seed).
    """
    worker_id, seeds, model_path, model_type, use_image = args
    
    # Import here to avoid multiprocessing issues
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    from pvp.sb3.td3.policies import TD3Policy
    from pvp.sb3.sac.our_features_extractor import OurFeaturesExtractorCNN as OurFeaturesExtractor
    from pvp.sb3.common.save_util import load_from_zip_file
    
    # Create environment config - use fixed range to cover all seeds
    # Use 1000 as start_seed and 1000 scenarios to cover seeds 1000-1999
    if use_image:
        from metadrive.component.sensors.rgb_camera import RGBCamera
        sensor_size = (84, 84)
        config = dict(
            use_render=False,
            manual_control=False,
            start_seed=1000,
            num_scenarios=1000,
            horizon=1500,
            crash_vehicle_done=False,
            crash_object_done=False,
            image_observation=True,
            vehicle_config=dict(image_source="rgb_camera"),
            sensors={"rgb_camera": (RGBCamera, *sensor_size)},
            stack_size=3,
        )
    else:
        config = dict(
            use_render=False,
            manual_control=False,
            start_seed=1000,
            num_scenarios=1000,
            horizon=1500,
        )
    
    # Create environment
    env = HumanInTheLoopEnv(config=config)
    
    # Load model
    policy_kwargs = dict(
        features_extractor_class=OurFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=275),
        share_features_extractor=False,
        net_arch=[256,]
    )
    
    if model_type == "iql":
        from pvp.sb3.td3.iql import IQL
        model = IQL(policy=TD3Policy, env=env, policy_kwargs=policy_kwargs, verbose=0, device="cuda")
    else:
        from pvp.sb3.td3.td3 import TD3
        model = TD3(policy=TD3Policy, env=env, policy_kwargs=policy_kwargs, verbose=0, device="cuda")
    
    data, params, _ = load_from_zip_file(model_path, device=model.device, print_system_info=False)
    model.set_parameters(params, exact_match=True, device=model.device)
    
    # Evaluate each seed
    results = []
    for seed in seeds:
        obs = env.reset(seed=seed)
        done = False
        episode_reward = 0
        episode_length = 0
        route_completion = 0
        had_crash_vehicle = False
        had_crash_object = False
        had_out_of_road = False
        arrive_dest = False
        min_vehicle_dist = float('inf')
        total_close_encounters = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            route_completion = max(route_completion, info.get('route_completion', 0.0))
            
            if info.get('crash_vehicle', False):
                had_crash_vehicle = True
            if info.get('crash_object', False):
                had_crash_object = True
            if info.get('out_of_road', False):
                had_out_of_road = True
            if info.get('arrive_dest', False):
                arrive_dest = True
            
            vehicle_dist = info.get('min_vehicle_distance', -1)
            if vehicle_dist > 0 and vehicle_dist < min_vehicle_dist:
                min_vehicle_dist = vehicle_dist
            
            close_count = info.get('close_vehicle_count', 0)
            if close_count > 0:
                total_close_encounters += close_count
        
        had_bad_event = had_crash_vehicle or had_crash_object or had_out_of_road
        
        result = {
            'seed': seed,
            'reward': float(episode_reward),
            'length': int(episode_length),
            'route_completion': float(route_completion),
            'crash_vehicle_rate': float(had_crash_vehicle),
            'crash_object_rate': float(had_crash_object),
            'out_of_road_rate': float(had_out_of_road),
            'any_bad_event_rate': float(had_bad_event),
            'success_rate': float(arrive_dest),
            'min_vehicle_distance': float(min_vehicle_dist) if min_vehicle_dist != float('inf') else -1,
            'total_close_encounters': float(total_close_encounters),
        }
        results.append(result)
        
        print(f"  [Worker {worker_id}] seed={seed}: reward={episode_reward:.1f}, "
              f"route={route_completion:.1%}, success={arrive_dest}")
    
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Fast parallel evaluation")
    parser.add_argument("--model", type=str, required=True, choices=["iql", "td3", "both"],
                        help="Which model to evaluate")
    parser.add_argument("--num_seeds", type=int, default=200,
                        help="Number of seeds to evaluate")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of parallel workers")
    parser.add_argument("--iql_checkpoint", type=str, default="IQLBEST1.zip",
                        help="Path to IQL checkpoint")
    parser.add_argument("--td3_checkpoint", type=str, default="TD3BCBEST2.zip",
                        help="Path to TD3 checkpoint")
    parser.add_argument("--output", type=str, default="./results/hard_scenario_eval",
                        help="Output directory")
    parser.add_argument("--use_image", action="store_true", default=True,
                        help="Use RGB image observation")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    seeds = HARD_SEEDS[:args.num_seeds]
    print(f"Fast parallel evaluation on {len(seeds)} scenarios")
    print(f"Using {args.num_workers} parallel workers")
    
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    models_to_eval = {}
    if args.model in ["iql", "both"]:
        models_to_eval["iql"] = (script_dir / args.iql_checkpoint, "iql")
    if args.model in ["td3", "both"]:
        models_to_eval["td3bc2"] = (script_dir / args.td3_checkpoint, "td3")
    
    all_results = {}
    
    for model_name, (model_path, model_type) in models_to_eval.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.upper()} with {args.num_workers} workers")
        print(f"{'='*60}")
        
        if not model_path.exists():
            print(f"ERROR: {model_path} not found")
            continue
        
        # Distribute seeds across workers
        seeds_per_worker = np.array_split(seeds, args.num_workers)
        worker_args = [
            (i, list(seeds_per_worker[i]), str(model_path), model_type, args.use_image)
            for i in range(args.num_workers)
            if len(seeds_per_worker[i]) > 0
        ]
        
        print(f"  Distributing {len(seeds)} seeds across {len(worker_args)} workers")
        for i, (wid, wseeds, _, _, _) in enumerate(worker_args):
            print(f"    Worker {wid}: {len(wseeds)} seeds")
        
        # Run in parallel
        model_start = time.time()
        with Pool(processes=len(worker_args)) as pool:
            worker_results = pool.map(evaluate_seeds_worker, worker_args)
        
        # Flatten results
        model_results = []
        for wr in worker_results:
            model_results.extend(wr)
        
        model_time = time.time() - model_start
        print(f"\n{model_name.upper()} completed in {model_time/60:.1f} minutes")
        
        # Print summary
        rewards = [r['reward'] for r in model_results]
        success_rates = [r['success_rate'] for r in model_results]
        route_completions = [r['route_completion'] for r in model_results]
        crash_rates = [r['crash_vehicle_rate'] for r in model_results]
        
        print(f"\n{model_name.upper()} Summary:")
        print(f"  Reward: mean={np.mean(rewards):.1f}, std={np.std(rewards):.1f}")
        print(f"  Route Completion: {np.mean(route_completions)*100:.1f}%")
        print(f"  Success Rate: {np.mean(success_rates)*100:.1f}%")
        print(f"  Crash Vehicle Rate: {np.mean(crash_rates)*100:.1f}%")
        
        all_results[model_name] = model_results
    
    # Save results
    with open(output_path / "hard_scenario_results_fast.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
