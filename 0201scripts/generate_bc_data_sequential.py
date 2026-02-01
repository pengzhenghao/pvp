"""
Generate BC data with SEQUENTIAL storage per environment.

Key difference from parallel version:
- Each environment's data is stored CONSECUTIVELY in the batch file
- This ensures obs[idx+1] is the TRUE next observation from the same trajectory
- Episode boundaries are marked with done=True

Storage format:
- Batch files contain transitions from multiple envs, but grouped by env
- Within each env's data block, transitions are sequential in time
- This allows using optimize_memory (next_obs = obs[idx+1]) correctly

Usage:
    python generate_bc_data_sequential.py --num_envs 10 --total_timesteps 100000 --batch_size 10240
"""

import os
os.environ["SDL_VIDEODRIVER"] = "offscreen"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import json
import numpy as np
import time
from pathlib import Path
from collections import defaultdict

# Suppress logging
import logging
logging.getLogger('metadrive').setLevel(logging.WARNING)
logging.getLogger('panda3d').setLevel(logging.WARNING)

import sys
sys.path.insert(0, '/p0/user/caihy/pvp')

# Top 200 hardest scenarios for the lidar PPO expert (in [0, 1000) range)
HARD_200_SEEDS = [
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


def save_batch_sequential(env_buffers, transitions_dir, batch_idx):
    """
    Save transitions from multiple envs, grouped sequentially by env.
    
    This ensures that within each env's data block, transitions are consecutive in time.
    So obs[idx+1] within the same env is the true next observation.
    
    We also save env_start_indices so the loader knows where each env's data starts.
    """
    all_obs_images = []
    all_obs_states = []
    all_actions = []
    all_rewards = []
    all_dones = []
    env_boundaries = [0]  # Start indices for each env's data
    
    for env_id in sorted(env_buffers.keys()):
        buffer = env_buffers[env_id]
        if len(buffer) == 0:
            continue
            
        for trans in buffer:
            obs = trans['obs']
            if isinstance(obs, dict):
                obs_img = obs.get('image', obs.get('default'))
                obs_st = obs.get('state', np.zeros(19, dtype=np.float32))
            else:
                obs_img = obs
                obs_st = np.zeros(19, dtype=np.float32)
            
            all_obs_images.append(obs_img)
            all_obs_states.append(obs_st)
            all_actions.append(trans['action'])
            all_rewards.append(trans['reward'])
            all_dones.append(trans['done'])
        
        env_boundaries.append(len(all_obs_images))
    
    if len(all_obs_images) == 0:
        return 0
    
    save_dict = {
        'obs_image': np.stack(all_obs_images).astype(np.float32),
        'obs_state': np.stack(all_obs_states).astype(np.float32),
        'action': np.stack(all_actions).astype(np.float32),
        'reward': np.array(all_rewards, dtype=np.float32),
        'done': np.array(all_dones, dtype=np.bool_),
        'size': len(all_obs_images),
        'env_boundaries': np.array(env_boundaries, dtype=np.int32),
        'sequential_storage': True,  # Flag to indicate sequential storage
    }
    
    file_path = transitions_dir / f"batch_{batch_idx:05d}.npz"
    np.savez(file_path, **save_dict)
    
    return len(all_obs_images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/caihy/bc_data_sequential")
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--num_envs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10240, 
                        help="Transitions per batch file (should be divisible by num_envs)")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--save_validation_buffer", action="store_true",
                        help="Save full buffer as npz for validation (toy mode only)")
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--wandb_project", type=str, default="bc-data-gen")
    parser.add_argument("--exp_name", type=str, default="bc-data-gen")
    args = parser.parse_args()
    
    if args.toy:
        args.total_timesteps = 5000
        args.batch_size = 1000
        args.num_envs = 5
        args.save_validation_buffer = True
        print("TOY MODE: 5000 timesteps, 5 envs, 1000 batch size", flush=True)
    
    print("=" * 80, flush=True)
    print("Sequential BC Data Generation", flush=True)
    print("=" * 80, flush=True)
    print(f"Data dir: {args.data_dir}", flush=True)
    print(f"Total timesteps: {args.total_timesteps}", flush=True)
    print(f"Num envs: {args.num_envs}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Save validation buffer: {args.save_validation_buffer}", flush=True)
    
    # Import after setting env vars
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from pvp.sb3.common.monitor import Monitor
    
    # Note: We use the built-in expert from FakeHumanEnv (disable_expert=False)
    # The expert action is provided in infos['raw_action']
    print("Using built-in expert from FakeHumanEnv", flush=True)
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    transitions_dir = data_dir / "batches"
    transitions_dir.mkdir(parents=True, exist_ok=True)
    
    # Environment config - use built-in expert via FakeHumanEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from pvp.experiments.metadrive.egpo.fakehuman_env import FakeHumanEnv
    
    sensor_size = (84, 84)
    
    # CORRECT CONFIG: Do NOT set traffic_density or random_traffic
    # This uses env default (traffic_density=0.06)
    # Matches the evaluation configuration
    env_config = dict(
        image_observation=True,
        vehicle_config=dict(image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, *sensor_size)},
        horizon=1500,
        # Safety settings - match eval config
        crash_vehicle_done=False,
        crash_object_done=False,
        crash_vehicle_penalty=5.0,
        crash_object_penalty=5.0,
        out_of_road_penalty=5.0,
        # Reward settings
        driving_reward=1.0,
        speed_reward=0.1,
        use_lateral_reward=False,
        # Other settings
        daytime="08:30",
        use_render=False,
        disable_expert=False,
        start_seed=0,
        num_scenarios=1000,
        # NOTE: traffic_density and random_traffic are NOT SET
        # This uses env default (traffic_density=0.06)
    )
    print("IMPORTANT: Using env default traffic_density (0.06), NOT 0.3!", flush=True)
    
    # Create wrapper that selects from hard seeds (same as parallel version)
    class HardSeedFakeHumanEnv:
        """Wrapper that creates FakeHumanEnv with random hard seeds."""
        def __init__(self, env_config, hard_seeds):
            self.hard_seeds = hard_seeds
            self.env_config = env_config
            self.current_seed = None
            self.env = FakeHumanEnv(self.env_config)
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
            self.metadata = getattr(self.env, 'metadata', {'render.modes': []})
            self.reward_range = getattr(self.env, 'reward_range', (-float('inf'), float('inf')))
            self.spec = getattr(self.env, 'spec', None)
        
        def reset(self, **kwargs):
            self.current_seed = int(np.random.choice(self.hard_seeds))
            kwargs['seed'] = self.current_seed
            result = self.env.reset(**kwargs)
            if isinstance(result, tuple):
                return result[0]
            return result
        
        def step(self, action):
            return self.env.step(action)
        
        def render(self, mode='human'):
            return self.env.render(mode)
        
        def close(self):
            return self.env.close()
        
        def seed(self, seed=None):
            pass
    
    def make_env(rank):
        def _init():
            env = HardSeedFakeHumanEnv(env_config, HARD_200_SEEDS)
            env = Monitor(env)
            return env
        return _init
    
    print(f"Creating {args.num_envs} subprocess environments...", flush=True)
    vec_env = SubprocVecEnv([make_env(i) for i in range(args.num_envs)])
    print("VecEnv created!", flush=True)
    
    # Build comprehensive env config for logging
    env_config_log = {
        'env/image_observation': env_config.get('image_observation'),
        'env/traffic_density': 'NOT_SET_default_0.06',  # Explicitly note we're using default
        'env/random_traffic': 'NOT_SET_default',
        'env/daytime': env_config.get('daytime'),
        'env/crash_vehicle_done': env_config.get('crash_vehicle_done'),
        'env/crash_object_done': env_config.get('crash_object_done'),
        'env/crash_vehicle_penalty': env_config.get('crash_vehicle_penalty'),
        'env/crash_object_penalty': env_config.get('crash_object_penalty'),
        'env/out_of_road_penalty': env_config.get('out_of_road_penalty'),
        'env/driving_reward': env_config.get('driving_reward'),
        'env/speed_reward': env_config.get('speed_reward'),
        'env/use_lateral_reward': env_config.get('use_lateral_reward'),
        'env/horizon': env_config.get('horizon'),
        'env/start_seed': env_config.get('start_seed'),
        'env/num_scenarios': env_config.get('num_scenarios'),
        'env/disable_expert': env_config.get('disable_expert'),
        'data/hard_seeds_count': len(HARD_200_SEEDS),
        'data/hard_seeds_sample': str(HARD_200_SEEDS[:10]),
    }
    
    print("\nEnvironment config (FULL):", flush=True)
    for key, val in env_config_log.items():
        print(f"    {key}={val}", flush=True)
    
    # Initialize wandb
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            from pvp.utils.utils import get_time_str
            import uuid
            trial_name = f"{args.exp_name}_{get_time_str()}_{uuid.uuid4().hex[:8]}"
            # Merge args and env_config for comprehensive wandb config
            wandb_config = {**vars(args), **env_config_log}
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity="victorique",
                name=trial_name,
                config=wandb_config,
            )
            print(f"Wandb initialized: {trial_name}", flush=True)
        except Exception as e:
            print(f"WARNING: Failed to init wandb: {e}", flush=True)
    
    # Per-environment buffers for sequential storage
    env_buffers = defaultdict(list)
    
    # Validation buffer (for toy mode)
    validation_buffer = [] if args.save_validation_buffer else None
    
    batch_counter = 0
    total_timesteps = 0
    
    print("Initial reset...", flush=True)
    obs = vec_env.reset()
    
    start_time = time.time()
    
    # Track per-env steps for validation
    env_step_counters = [0] * args.num_envs
    
    # =========== EVALUATION METRICS TRACKING ===========
    # Per-episode tracking (per env)
    episode_rewards = np.zeros(args.num_envs)
    episode_lengths = np.zeros(args.num_envs, dtype=int)
    route_completions = np.zeros(args.num_envs)
    had_crash_vehicle = np.zeros(args.num_envs, dtype=bool)
    had_crash_object = np.zeros(args.num_envs, dtype=bool)
    had_out_of_road = np.zeros(args.num_envs, dtype=bool)
    arrive_dests = np.zeros(args.num_envs, dtype=bool)
    
    # Reward component tracking (per env)
    total_velocities = np.zeros(args.num_envs)
    total_crash_penalties = np.zeros(args.num_envs)
    total_out_of_road_penalties = np.zeros(args.num_envs)
    
    # Aggregate metrics
    completed_episodes = []
    eval_log_interval = 50  # Log every N episodes
    # =================================================
    
    # Dummy actions (env will use expert internally)
    dummy_actions = np.zeros((args.num_envs, 2), dtype=np.float32)
    
    while total_timesteps < args.total_timesteps:
        # Step environments with dummy actions - expert action will be in infos['raw_action']
        next_obs, rewards, dones, infos = vec_env.step(dummy_actions)
        
        # Store transitions per environment
        for i in range(args.num_envs):
            if isinstance(obs, dict):
                obs_i = {k: v[i].copy() for k, v in obs.items()}
            elif isinstance(obs[i], dict):
                obs_i = {k: v.copy() for k, v in obs[i].items()}
            else:
                obs_i = obs[i].copy()
            
            # Get expert action from infos
            if isinstance(infos, dict):
                raw_action = infos.get('raw_action', None)
                if raw_action is not None:
                    expert_action = raw_action[i]
                else:
                    expert_action = dummy_actions[i]
                info_i = {k: (v[i] if hasattr(v, '__getitem__') else v) for k, v in infos.items()}
            elif isinstance(infos, (list, tuple)):
                info_i = infos[i]
                expert_action = info_i.get('raw_action', dummy_actions[i])
            else:
                info_i = {}
                expert_action = dummy_actions[i]
            
            expert_action = np.asarray(expert_action, dtype=np.float32).flatten()
            if expert_action.shape != (2,):
                expert_action = np.zeros(2, dtype=np.float32)
            
            trans = {
                'obs': obs_i,
                'action': expert_action.copy(),
                'reward': float(rewards[i]),
                'done': bool(dones[i]),
            }
            
            env_buffers[i].append(trans)
            total_timesteps += 1
            env_step_counters[i] += 1
            
            # =========== EVALUATION METRICS TRACKING ===========
            episode_rewards[i] += rewards[i]
            episode_lengths[i] += 1
            route_completions[i] = max(route_completions[i], info_i.get('route_completion', 0.0))
            
            # Track velocity
            velocity = info_i.get('velocity', 0.0)
            if isinstance(velocity, (list, np.ndarray)):
                velocity = np.linalg.norm(velocity)
            total_velocities[i] += velocity
            
            if info_i.get('crash_vehicle', False):
                had_crash_vehicle[i] = True
                total_crash_penalties[i] += 5.0  # crash_vehicle_penalty
            if info_i.get('crash_object', False):
                had_crash_object[i] = True
                total_crash_penalties[i] += 5.0  # crash_object_penalty
            if info_i.get('out_of_road', False):
                had_out_of_road[i] = True
                total_out_of_road_penalties[i] += 5.0  # out_of_road_penalty
            if info_i.get('arrive_dest', False):
                arrive_dests[i] = True
            
            # Episode done - record metrics
            if dones[i]:
                had_bad = had_crash_vehicle[i] or had_crash_object[i] or had_out_of_road[i]
                rc_no_bad = route_completions[i] if not had_bad else 0.0
                
                avg_velocity = total_velocities[i] / episode_lengths[i] if episode_lengths[i] > 0 else 0.0
                completed_episodes.append({
                    'reward': float(episode_rewards[i]),
                    'length': int(episode_lengths[i]),
                    'route_completion': float(route_completions[i]),
                    'route_completion_no_bad': float(rc_no_bad),
                    'success': float(arrive_dests[i]),
                    'success_no_bad': float(arrive_dests[i] and not had_bad),
                    'crash_vehicle': float(had_crash_vehicle[i]),
                    'crash_object': float(had_crash_object[i]),
                    'out_of_road': float(had_out_of_road[i]),
                    'any_bad_event': float(had_bad),
                    # Reward components
                    'avg_velocity': float(avg_velocity),
                    'total_crash_penalty': float(total_crash_penalties[i]),
                    'total_out_of_road_penalty': float(total_out_of_road_penalties[i]),
                })
                
                # Reset per-episode tracking
                episode_rewards[i] = 0
                episode_lengths[i] = 0
                route_completions[i] = 0
                had_crash_vehicle[i] = False
                had_crash_object[i] = False
                had_out_of_road[i] = False
                arrive_dests[i] = False
                total_velocities[i] = 0
                total_crash_penalties[i] = 0
                total_out_of_road_penalties[i] = 0
            # =================================================
            
            # Save to validation buffer with env_id for verification
            if validation_buffer is not None:
                validation_buffer.append({
                    'env_id': i,
                    'step_in_env': env_step_counters[i] - 1,
                    **trans
                })
        
        obs = next_obs
        
        # Save batch when buffer is large enough
        total_in_buffers = sum(len(b) for b in env_buffers.values())
        if total_in_buffers >= args.batch_size:
            saved = save_batch_sequential(env_buffers, transitions_dir, batch_counter)
            print(f"Saved batch {batch_counter}: {saved} transitions", flush=True)
            batch_counter += 1
            env_buffers = defaultdict(list)
        
        # Log evaluation metrics periodically
        if len(completed_episodes) >= eval_log_interval and len(completed_episodes) % eval_log_interval == 0:
            recent = completed_episodes[-eval_log_interval:]
            eval_metrics = {
                'eval/reward_mean': np.mean([e['reward'] for e in recent]),
                'eval/route_completion': np.mean([e['route_completion'] for e in recent]),
                'eval/route_completion_no_bad': np.mean([e['route_completion_no_bad'] for e in recent]),
                'eval/success_rate': np.mean([e['success'] for e in recent]),
                'eval/success_no_bad': np.mean([e['success_no_bad'] for e in recent]),
                'eval/crash_vehicle_rate': np.mean([e['crash_vehicle'] for e in recent]),
                'eval/out_of_road_rate': np.mean([e['out_of_road'] for e in recent]),
                'eval/any_bad_event_rate': np.mean([e['any_bad_event'] for e in recent]),
                'eval/total_episodes': len(completed_episodes),
                # Reward component breakdown
                'reward/avg_velocity': np.mean([e['avg_velocity'] for e in recent]),
                'reward/crash_penalty_mean': np.mean([e['total_crash_penalty'] for e in recent]),
                'reward/out_of_road_penalty_mean': np.mean([e['total_out_of_road_penalty'] for e in recent]),
                # Estimated driving reward (reward + penalties)
                'reward/estimated_driving_reward': np.mean([e['reward'] + e['total_crash_penalty'] + e['total_out_of_road_penalty'] for e in recent]),
            }
            
            if wandb_run:
                wandb_run.log(eval_metrics, step=total_timesteps)
            
            print(f"  [Eval] Ep={len(completed_episodes)} R={eval_metrics['eval/reward_mean']:.1f} "
                  f"Succ={eval_metrics['eval/success_rate']*100:.0f}% "
                  f"SuccNoBad={eval_metrics['eval/success_no_bad']*100:.0f}% "
                  f"RC={eval_metrics['eval/route_completion']*100:.0f}% "
                  f"RCNoBad={eval_metrics['eval/route_completion_no_bad']*100:.0f}% "
                  f"Crash={eval_metrics['eval/crash_vehicle_rate']*100:.0f}%", flush=True)
        
        # Progress
        if total_timesteps % 1000 == 0:
            elapsed = time.time() - start_time
            rate = total_timesteps / elapsed
            eta = (args.total_timesteps - total_timesteps) / rate / 60
            print(f"[{total_timesteps}/{args.total_timesteps}] Rate={rate:.0f}/s ETA={eta:.1f}min Ep={len(completed_episodes)}", 
                  flush=True)
    
    # Save remaining
    if sum(len(b) for b in env_buffers.values()) > 0:
        saved = save_batch_sequential(env_buffers, transitions_dir, batch_counter)
        print(f"Saved final batch {batch_counter}: {saved} transitions", flush=True)
        batch_counter += 1
    
    vec_env.close()
    
    # Save metadata
    metadata = {
        'total_transitions': total_timesteps,
        'num_batch_files': batch_counter,
        'batch_size': args.batch_size,
        'num_envs': args.num_envs,
        'sequential_storage': True,
        'hard_seeds': HARD_200_SEEDS[:20],  # Sample
        'daytime': '08:30',
    }
    
    with open(data_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to {data_dir / 'metadata.json'}", flush=True)
    
    # Save validation buffer
    if validation_buffer is not None:
        print(f"\nSaving validation buffer ({len(validation_buffer)} transitions)...", flush=True)
        
        val_obs_images = []
        val_obs_states = []
        val_actions = []
        val_rewards = []
        val_dones = []
        val_env_ids = []
        val_step_in_envs = []
        
        for trans in validation_buffer:
            obs = trans['obs']
            if isinstance(obs, dict):
                val_obs_images.append(obs.get('image', obs.get('default')))
                val_obs_states.append(obs.get('state', np.zeros(19)))
            else:
                val_obs_images.append(obs)
                val_obs_states.append(np.zeros(19))
            val_actions.append(trans['action'])
            val_rewards.append(trans['reward'])
            val_dones.append(trans['done'])
            val_env_ids.append(trans['env_id'])
            val_step_in_envs.append(trans['step_in_env'])
        
        np.savez(
            data_dir / "validation_buffer.npz",
            obs_image=np.stack(val_obs_images).astype(np.float32),
            obs_state=np.stack(val_obs_states).astype(np.float32),
            action=np.stack(val_actions).astype(np.float32),
            reward=np.array(val_rewards, dtype=np.float32),
            done=np.array(val_dones, dtype=np.bool_),
            env_id=np.array(val_env_ids, dtype=np.int32),
            step_in_env=np.array(val_step_in_envs, dtype=np.int32),
        )
        print(f"Validation buffer saved to {data_dir / 'validation_buffer.npz'}", flush=True)
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} min", flush=True)
    print(f"Total transitions: {total_timesteps}", flush=True)
    print(f"Total batch files: {batch_counter}", flush=True)
    
    # =========== FINAL EVALUATION METRICS ===========
    if len(completed_episodes) > 0:
        print("\n" + "=" * 80, flush=True)
        print("LIDAR EXPERT EVALUATION DURING DATA GENERATION", flush=True)
        print("=" * 80, flush=True)
        
        final_metrics = {
            'eval/reward_mean': np.mean([e['reward'] for e in completed_episodes]),
            'eval/reward_std': np.std([e['reward'] for e in completed_episodes]),
            'eval/route_completion': np.mean([e['route_completion'] for e in completed_episodes]),
            'eval/route_completion_no_bad': np.mean([e['route_completion_no_bad'] for e in completed_episodes]),
            'eval/success_rate': np.mean([e['success'] for e in completed_episodes]),
            'eval/success_no_bad': np.mean([e['success_no_bad'] for e in completed_episodes]),
            'eval/crash_vehicle_rate': np.mean([e['crash_vehicle'] for e in completed_episodes]),
            'eval/crash_object_rate': np.mean([e['crash_object'] for e in completed_episodes]),
            'eval/out_of_road_rate': np.mean([e['out_of_road'] for e in completed_episodes]),
            'eval/any_bad_event_rate': np.mean([e['any_bad_event'] for e in completed_episodes]),
            'eval/total_episodes': len(completed_episodes),
            'eval/total_transitions': total_timesteps,
        }
        
        print(f"Total Episodes: {len(completed_episodes)}", flush=True)
        print(f"Reward: mean={final_metrics['eval/reward_mean']:.1f}, std={final_metrics['eval/reward_std']:.1f}", flush=True)
        print(f"Route Completion: {final_metrics['eval/route_completion']*100:.1f}%", flush=True)
        print(f"Route Completion (no bad): {final_metrics['eval/route_completion_no_bad']*100:.1f}%", flush=True)
        print(f"Success Rate: {final_metrics['eval/success_rate']*100:.1f}%", flush=True)
        print(f"Success Rate (no bad event): {final_metrics['eval/success_no_bad']*100:.1f}%", flush=True)
        print(f"--- Safety Metrics ---", flush=True)
        print(f"Crash Vehicle Rate: {final_metrics['eval/crash_vehicle_rate']*100:.1f}%", flush=True)
        print(f"Crash Object Rate: {final_metrics['eval/crash_object_rate']*100:.1f}%", flush=True)
        print(f"Out of Road Rate: {final_metrics['eval/out_of_road_rate']*100:.1f}%", flush=True)
        print(f"Any Bad Event Rate: {final_metrics['eval/any_bad_event_rate']*100:.1f}%", flush=True)
        print("=" * 80, flush=True)
        
        # Expected metrics from lidar expert on training seeds (traffic_density=0.06)
        # From eval_lidar_train_13152.out: R=309.2, RC=83.6%, Succ=58%, SuccNoBad=16%
        # If our metrics are VERY different, there's a config problem!
        print("\n[VALIDATION] Comparing with expected Lidar Expert performance:", flush=True)
        print(f"  Expected (traffic_density=0.06): SuccNoBad~16%, R~309, RC~84%, Crash~59%", flush=True)
        print(f"  Actual:                          SuccNoBad={final_metrics['eval/success_no_bad']*100:.0f}%, "
              f"R={final_metrics['eval/reward_mean']:.0f}, RC={final_metrics['eval/route_completion']*100:.0f}%, "
              f"Crash={final_metrics['eval/crash_vehicle_rate']*100:.0f}%", flush=True)
        
        # Check for large deviations
        succ_no_bad_actual = final_metrics['eval/success_no_bad']
        if succ_no_bad_actual < 0.10 or succ_no_bad_actual > 0.25:
            print(f"  WARNING: SuccNoBad={succ_no_bad_actual*100:.0f}% is outside expected range [10%, 25%]!", flush=True)
        else:
            print(f"  OK: SuccNoBad is within expected range", flush=True)
        
        if wandb_run:
            wandb_run.log(final_metrics, step=total_timesteps)
            wandb_run.summary.update(final_metrics)
            wandb_run.finish()
            print("\nWandb finished!", flush=True)
    # =================================================


if __name__ == "__main__":
    main()
