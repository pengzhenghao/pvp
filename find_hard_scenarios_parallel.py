"""
Find hard scenarios for pretrained model evaluation (Parallel version for SLURM).
Tests scenarios 1000-1999 and ranks them by difficulty (bad event probability).
Uses SubprocVecEnv for parallel evaluation.

Usage:
    python find_hard_scenarios_parallel.py --num_envs 25 --num_trials 5 --output hard_scenarios.json
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

from metadrive.component.sensors.rgb_camera import RGBCamera


def evaluate_scenarios_parallel(model, make_env_fn, scenario_seeds, num_trials=5, num_envs=25):
    """
    Evaluate multiple scenarios in parallel using SubprocVecEnv.
    Each scenario is tested num_trials times.
    
    Returns:
        List of result dicts for each scenario
    """
    from pvp.sb3.common.vec_env import SubprocVecEnv
    
    all_results = []
    
    # Process scenarios in batches
    # Each batch: num_envs scenarios, each tested num_trials times
    # Total episodes per batch = num_envs * num_trials
    
    for batch_start in range(0, len(scenario_seeds), num_envs):
        batch_seeds = scenario_seeds[batch_start:batch_start + num_envs]
        batch_size = len(batch_seeds)
        
        # Initialize results for this batch
        batch_results = {seed: {
            'crash_vehicle': 0,
            'crash_object': 0,
            'out_of_road': 0,
            'arrive_dest': 0,
            'any_bad_event': 0,
            'total_reward': 0,
            'total_length': 0,
            'trials_completed': 0,
        } for seed in batch_seeds}
        
        # Run num_trials for each scenario in the batch
        for trial in range(num_trials):
            # Create parallel envs for this trial
            env_fns = [make_env_fn(seed) for seed in batch_seeds]
            vec_env = SubprocVecEnv(env_fns)
            
            obs = vec_env.reset()
            dones = np.zeros(batch_size, dtype=bool)
            
            # Track per-env statistics
            episode_rewards = np.zeros(batch_size)
            episode_lengths = np.zeros(batch_size)
            episode_crash_vehicle = np.zeros(batch_size, dtype=bool)
            episode_crash_object = np.zeros(batch_size, dtype=bool)
            episode_out_of_road = np.zeros(batch_size, dtype=bool)
            episode_arrive_dest = np.zeros(batch_size, dtype=bool)
            
            while not np.all(dones):
                actions, _ = model.predict(obs, deterministic=True)
                obs, rewards, new_dones, infos = vec_env.step(actions)
                
                for i, (done, info) in enumerate(zip(new_dones, infos)):
                    if not dones[i]:  # Only update if not already done
                        episode_rewards[i] += rewards[i]
                        episode_lengths[i] += 1
                        
                        if isinstance(info, dict):
                            if info.get('crash_vehicle', False):
                                episode_crash_vehicle[i] = True
                            if info.get('crash_object', False):
                                episode_crash_object[i] = True
                            if info.get('out_of_road', False):
                                episode_out_of_road[i] = True
                            if info.get('arrive_dest', False):
                                episode_arrive_dest[i] = True
                
                dones = dones | new_dones
            
            vec_env.close()
            
            # Update batch results
            for i, seed in enumerate(batch_seeds):
                if episode_crash_vehicle[i]:
                    batch_results[seed]['crash_vehicle'] += 1
                if episode_crash_object[i]:
                    batch_results[seed]['crash_object'] += 1
                if episode_out_of_road[i]:
                    batch_results[seed]['out_of_road'] += 1
                if episode_arrive_dest[i]:
                    batch_results[seed]['arrive_dest'] += 1
                if episode_crash_vehicle[i] or episode_crash_object[i] or episode_out_of_road[i]:
                    batch_results[seed]['any_bad_event'] += 1
                batch_results[seed]['total_reward'] += episode_rewards[i]
                batch_results[seed]['total_length'] += episode_lengths[i]
                batch_results[seed]['trials_completed'] += 1
        
        # Convert batch results to final format
        for seed in batch_seeds:
            r = batch_results[seed]
            n = r['trials_completed']
            all_results.append({
                'scenario_seed': seed,
                'num_trials': n,
                'bad_event_rate': r['any_bad_event'] / n if n > 0 else 0,
                'crash_vehicle_rate': r['crash_vehicle'] / n if n > 0 else 0,
                'crash_object_rate': r['crash_object'] / n if n > 0 else 0,
                'out_of_road_rate': r['out_of_road'] / n if n > 0 else 0,
                'success_rate': r['arrive_dest'] / n if n > 0 else 0,
                'avg_episode_reward': r['total_reward'] / n if n > 0 else 0,
                'avg_episode_length': r['total_length'] / n if n > 0 else 0,
            })
    
    return all_results


def save_results(all_results, output_file):
    """Save sorted results to file."""
    # Sort by difficulty (bad_event_rate descending, then crash_vehicle_rate descending)
    sorted_results = sorted(
        all_results, 
        key=lambda x: (x['bad_event_rate'], x['crash_vehicle_rate']),
        reverse=True
    )
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'total_scenarios_tested': len(sorted_results),
            'sorted_by': 'bad_event_rate (descending), then crash_vehicle_rate (descending)',
            'scenarios': sorted_results
        }, f, indent=2)
    
    # Also save a simple text summary
    txt_file = str(output_file).replace('.json', '_summary.txt')
    with open(txt_file, 'w') as f:
        f.write("Hard Scenarios Summary (sorted by difficulty)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Rank':<6}{'Seed':<8}{'BadEvent%':<12}{'CrashVeh%':<12}{'CrashObj%':<12}{'OutRoad%':<12}{'Success%':<12}\n")
        f.write("-" * 80 + "\n")
        for i, r in enumerate(sorted_results[:100]):  # Top 100
            f.write(f"{i+1:<6}{r['scenario_seed']:<8}"
                    f"{r['bad_event_rate']*100:<12.1f}"
                    f"{r['crash_vehicle_rate']*100:<12.1f}"
                    f"{r['crash_object_rate']*100:<12.1f}"
                    f"{r['out_of_road_rate']*100:<12.1f}"
                    f"{r['success_rate']*100:<12.1f}\n")
    
    print(f"Results saved to {output_file} and {txt_file}")
    
    # Print top 10 hardest
    print("\nTop 10 Hardest Scenarios:")
    print("-" * 60)
    for i, r in enumerate(sorted_results[:10]):
        print(f"  {i+1}. Seed {r['scenario_seed']}: "
              f"BadEvent={r['bad_event_rate']*100:.0f}%, "
              f"CrashVeh={r['crash_vehicle_rate']*100:.0f}%, "
              f"Success={r['success_rate']*100:.0f}%")


def main():
    parser = argparse.ArgumentParser(description="Find hard scenarios for pretrained model (Parallel)")
    parser.add_argument("--start_seed", type=int, default=1000, help="Start scenario seed")
    parser.add_argument("--num_scenarios", type=int, default=1000, help="Number of scenarios to test")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials per scenario")
    parser.add_argument("--num_envs", type=int, default=25, help="Number of parallel environments")
    parser.add_argument("--output", type=str, default="./results/hard_scenarios.json", help="Output file (relative path)")
    parser.add_argument("--save_interval", type=int, default=50, help="Save results every N scenarios")
    parser.add_argument("--checkpoint", type=str, default="./pretrained.zip", 
                        help="Path to pretrained model checkpoint (relative path)")
    # Penalty parameters
    parser.add_argument("--crash_vehicle_penalty", type=float, default=5.0)
    parser.add_argument("--crash_object_penalty", type=float, default=5.0)
    parser.add_argument("--out_of_road_penalty", type=float, default=5.0)
    args = parser.parse_args()
    
    start_time = time.time()
    
    # ===== Setup =====
    sensor_size = (84, 84)
    
    # Store args in a way accessible by lambda
    crash_vehicle_penalty = args.crash_vehicle_penalty
    crash_object_penalty = args.crash_object_penalty
    out_of_road_penalty = args.out_of_road_penalty
    
    def make_env_fn(scenario_seed):
        """Return a function that creates environment for a specific scenario."""
        def _init():
            eval_env_config = dict(
                use_render=False,
                manual_control=False,
                start_seed=scenario_seed,
                num_scenarios=1,  # Only this specific scenario
                horizon=1500,
                image_observation=True,
                vehicle_config=dict(image_source="rgb_camera"),
                sensors={"rgb_camera": (RGBCamera, *sensor_size)},
                stack_size=3,
                interface_panel=["rgb_camera", "dashboard"],
                daytime="08:30",
                crash_vehicle_done=False,
                crash_object_done=False,
                cost_to_reward=False,
                crash_vehicle_penalty=crash_vehicle_penalty,
                crash_object_penalty=crash_object_penalty,
                out_of_road_penalty=out_of_road_penalty,
            )
            from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
            return HumanInTheLoopEnv(config=eval_env_config)
        return _init
    
    # ===== Load model =====
    print("Loading pretrained model...")
    from pvp.sb3.common.vec_env import DummyVecEnv
    from pvp.sb3.td3.policies import TD3Policy
    from pvp.sb3.td3.td3 import TD3
    from pvp.sb3.haco import HACOReplayBuffer
    from pvp.sb3.sac.our_features_extractor import OurFeaturesExtractorCNN as OurFeaturesExtractor
    from pvp.sb3.common.save_util import load_from_zip_file
    
    # Create a temporary env for model initialization
    temp_env = DummyVecEnv([make_env_fn(args.start_seed)])
    
    policy_kwargs = dict(
        features_extractor_class=OurFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=275),
        share_features_extractor=False,
        net_arch=[256,]
    )
    
    model = TD3(
        policy=TD3Policy,
        replay_buffer_class=HACOReplayBuffer,
        replay_buffer_kwargs=dict(),
        policy_kwargs=policy_kwargs,
        env=temp_env,
        learning_rate=1e-4,
        optimize_memory_usage=True,
        learning_starts=0,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=None,
        policy_delay=2,
        create_eval_env=False,
        verbose=0,
        device="auto",
        buffer_size=1000,
    )
    
    # Load weights
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint {ckpt_path} not found!")
        return
    
    print(f"Loading checkpoint from {ckpt_path}...")
    data, params, pytorch_variables = load_from_zip_file(ckpt_path, device=model.device, print_system_info=False)
    model.set_parameters(params, exact_match=False, device=model.device)
    print("Model loaded successfully!")
    
    temp_env.close()
    
    # ===== Evaluate scenarios in batches =====
    all_results = []
    scenario_seeds = list(range(args.start_seed, args.start_seed + args.num_scenarios))
    
    print(f"\nEvaluating scenarios {args.start_seed} to {args.start_seed + args.num_scenarios - 1}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Trials per scenario: {args.num_trials}")
    print(f"Save interval: every {args.save_interval} scenarios")
    print("=" * 60)
    
    # Process in chunks for intermediate saving
    for chunk_start in range(0, len(scenario_seeds), args.save_interval):
        chunk_end = min(chunk_start + args.save_interval, len(scenario_seeds))
        chunk_seeds = scenario_seeds[chunk_start:chunk_end]
        
        print(f"\nProcessing scenarios {chunk_seeds[0]} to {chunk_seeds[-1]} "
              f"({chunk_start + 1}-{chunk_end}/{len(scenario_seeds)})...")
        
        try:
            chunk_results = evaluate_scenarios_parallel(
                model, make_env_fn, chunk_seeds, 
                num_trials=args.num_trials, 
                num_envs=args.num_envs
            )
            all_results.extend(chunk_results)
            
            # Print chunk statistics
            bad_count = sum(1 for r in chunk_results if r['bad_event_rate'] > 0)
            print(f"  Completed. Scenarios with bad events: {bad_count}/{len(chunk_results)}")
            
            # Save intermediate results
            print(f"  Saving intermediate results ({len(all_results)} scenarios total)...")
            save_results(all_results, args.output)
            
        except Exception as e:
            print(f"Error processing chunk: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    elapsed_time = time.time() - start_time
    print(f"\n\n{'='*60}")
    print("FINAL RESULTS")
    print("=" * 60)
    save_results(all_results, args.output)
    
    # Statistics summary
    bad_event_scenarios = [r for r in all_results if r['bad_event_rate'] > 0]
    crash_vehicle_scenarios = [r for r in all_results if r['crash_vehicle_rate'] > 0]
    
    print(f"\nSummary:")
    print(f"  Total scenarios tested: {len(all_results)}")
    print(f"  Scenarios with any bad event: {len(bad_event_scenarios)} ({len(bad_event_scenarios)/len(all_results)*100:.1f}%)")
    print(f"  Scenarios with crash vehicle: {len(crash_vehicle_scenarios)} ({len(crash_vehicle_scenarios)/len(all_results)*100:.1f}%)")
    print(f"  Total time: {elapsed_time/60:.1f} minutes")
    print(f"  Time per scenario: {elapsed_time/len(all_results):.2f} seconds")
    
    if bad_event_scenarios:
        avg_bad_event_rate = np.mean([r['bad_event_rate'] for r in bad_event_scenarios])
        print(f"  Average bad event rate (among hard scenarios): {avg_bad_event_rate*100:.1f}%")


if __name__ == "__main__":
    main()
