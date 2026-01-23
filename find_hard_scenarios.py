"""
Find hard scenarios for pretrained model evaluation.
Tests scenarios 1000-1999 and ranks them by difficulty (bad event probability).
Saves intermediate results every 50 scenarios.

Usage:
    python find_hard_scenarios.py --num_trials 5 --output hard_scenarios.json
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

import gymnasium
import gymnasium.spaces
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
gymnasium.spaces.space = gymnasium.spaces

from metadrive.component.sensors.rgb_camera import RGBCamera


def evaluate_scenario(model, env_maker, scenario_seed, num_trials=5):
    """
    Evaluate a single scenario multiple times and return statistics.
    
    Returns:
        dict with keys: 
            - bad_event_rate: probability of any bad event
            - crash_vehicle_rate: probability of crash vehicle
            - crash_object_rate: probability of crash object  
            - out_of_road_rate: probability of out of road
            - success_rate: probability of success (arrive_dest without bad events)
            - avg_episode_reward: average episode reward
            - avg_episode_length: average episode length
    """
    results = {
        'crash_vehicle': 0,
        'crash_object': 0,
        'out_of_road': 0,
        'arrive_dest': 0,
        'any_bad_event': 0,
        'total_reward': 0,
        'total_length': 0,
    }
    
    for trial in range(num_trials):
        env = env_maker(scenario_seed)
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        # Track bad events for this episode
        episode_crash_vehicle = False
        episode_crash_object = False
        episode_out_of_road = False
        episode_arrive_dest = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Check for bad events in info
            if isinstance(info, dict):
                if info.get('crash_vehicle', False):
                    episode_crash_vehicle = True
                if info.get('crash_object', False):
                    episode_crash_object = True
                if info.get('out_of_road', False):
                    episode_out_of_road = True
                if info.get('arrive_dest', False):
                    episode_arrive_dest = True
        
        # Update results
        if episode_crash_vehicle:
            results['crash_vehicle'] += 1
        if episode_crash_object:
            results['crash_object'] += 1
        if episode_out_of_road:
            results['out_of_road'] += 1
        if episode_arrive_dest:
            results['arrive_dest'] += 1
        if episode_crash_vehicle or episode_crash_object or episode_out_of_road:
            results['any_bad_event'] += 1
        
        results['total_reward'] += episode_reward
        results['total_length'] += episode_length
        
        env.close()
    
    return {
        'scenario_seed': scenario_seed,
        'num_trials': num_trials,
        'bad_event_rate': results['any_bad_event'] / num_trials,
        'crash_vehicle_rate': results['crash_vehicle'] / num_trials,
        'crash_object_rate': results['crash_object'] / num_trials,
        'out_of_road_rate': results['out_of_road'] / num_trials,
        'success_rate': results['arrive_dest'] / num_trials,
        'avg_episode_reward': results['total_reward'] / num_trials,
        'avg_episode_length': results['total_length'] / num_trials,
    }


def save_results(all_results, output_file):
    """Save sorted results to file."""
    # Sort by difficulty (bad_event_rate descending, then crash_vehicle_rate descending)
    sorted_results = sorted(
        all_results, 
        key=lambda x: (x['bad_event_rate'], x['crash_vehicle_rate']),
        reverse=True
    )
    
    with open(output_file, 'w') as f:
        json.dump({
            'total_scenarios_tested': len(sorted_results),
            'sorted_by': 'bad_event_rate (descending), then crash_vehicle_rate (descending)',
            'scenarios': sorted_results
        }, f, indent=2)
    
    # Also save a simple text summary
    txt_file = output_file.replace('.json', '_summary.txt')
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
    parser = argparse.ArgumentParser(description="Find hard scenarios for pretrained model")
    parser.add_argument("--start_seed", type=int, default=1000, help="Start scenario seed")
    parser.add_argument("--num_scenarios", type=int, default=1000, help="Number of scenarios to test")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials per scenario")
    parser.add_argument("--output", type=str, default="hard_scenarios.json", help="Output file")
    parser.add_argument("--save_interval", type=int, default=50, help="Save results every N scenarios")
    parser.add_argument("--checkpoint", type=str, default="/home/caihy/pvp/pretrained.zip", 
                        help="Path to pretrained model checkpoint")
    # Penalty parameters
    parser.add_argument("--crash_vehicle_penalty", type=float, default=5.0)
    parser.add_argument("--crash_object_penalty", type=float, default=5.0)
    parser.add_argument("--out_of_road_penalty", type=float, default=5.0)
    args = parser.parse_args()
    
    # ===== Setup =====
    sensor_size = (84, 84)
    
    def make_env(scenario_seed):
        """Create environment for a specific scenario."""
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
            crash_vehicle_penalty=args.crash_vehicle_penalty,
            crash_object_penalty=args.crash_object_penalty,
            out_of_road_penalty=args.out_of_road_penalty,
        )
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        return HumanInTheLoopEnv(config=eval_env_config)
    
    # ===== Load model =====
    print("Loading pretrained model...")
    from pvp.sb3.common.vec_env import DummyVecEnv
    from pvp.sb3.td3.policies import TD3Policy
    from pvp.sb3.td3.td3 import TD3
    from pvp.sb3.haco import HACOReplayBuffer
    from pvp.sb3.sac.our_features_extractor import OurFeaturesExtractorCNN as OurFeaturesExtractor
    from pvp.sb3.common.save_util import load_from_zip_file
    
    # Create a temporary env for model initialization
    temp_env = DummyVecEnv([lambda: make_env(args.start_seed)])
    
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
    
    # ===== Evaluate scenarios =====
    all_results = []
    end_seed = args.start_seed + args.num_scenarios
    
    print(f"\nEvaluating scenarios {args.start_seed} to {end_seed-1}")
    print(f"Trials per scenario: {args.num_trials}")
    print(f"Save interval: every {args.save_interval} scenarios")
    print("=" * 60)
    
    for i, seed in enumerate(range(args.start_seed, end_seed)):
        print(f"\rTesting scenario {seed} ({i+1}/{args.num_scenarios})...", end="", flush=True)
        
        try:
            result = evaluate_scenario(model, make_env, seed, args.num_trials)
            all_results.append(result)
            
            # Print brief status
            if result['bad_event_rate'] > 0:
                print(f" BadEvent={result['bad_event_rate']*100:.0f}%", end="")
        except Exception as e:
            print(f"\nError evaluating scenario {seed}: {e}")
            continue
        
        # Save intermediate results
        if (i + 1) % args.save_interval == 0:
            print(f"\n\n--- Saving intermediate results after {i+1} scenarios ---")
            save_results(all_results, args.output)
            print()
    
    # Final save
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
    
    if bad_event_scenarios:
        avg_bad_event_rate = np.mean([r['bad_event_rate'] for r in bad_event_scenarios])
        print(f"  Average bad event rate (among hard scenarios): {avg_bad_event_rate*100:.1f}%")


if __name__ == "__main__":
    main()
