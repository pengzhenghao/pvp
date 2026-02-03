"""
Train IQL from batched data format (optimize_memory style).

Key: next_obs = obs[index+1], NOT separately stored.
- Episode boundaries (done=True) require special handling
- Caches loaded batch files to reduce I/O
- Online rollout collection during training (logs rollout/* metrics to wandb)

Usage:
    python train_iql_from_batches.py --data_dir /data/caihy/bc_data_1M --log_dir /data/caihy/iql_training
"""

import sys
import gymnasium
import gymnasium.spaces
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
gymnasium.spaces.space = gymnasium.spaces  # metadrive uses gym.spaces.space.Space

import os
_gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_id
os.environ["SDL_VIDEODRIVER"] = "offscreen"
os.environ["PYOPENGL_PLATFORM"] = "egl"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import argparse
import numpy as np
from pathlib import Path
import time
import json
import torch
import torch.nn.functional as F
import psutil
import hashlib
from collections import deque, defaultdict

# CuPy for GPU image acceleration (optional)
_cupy_available = False
try:
    import cupy as cp
    from torch.utils.dlpack import from_dlpack
    _cupy_available = True
except ImportError:
    pass  # cupy not installed, will use CPU path


# Top 200 hardest scenarios for the lidar PPO expert (in [0, 1000) range)
# Same as generate_bc_data_sequential.py
# Top 200 hardest scenarios for the lidar PPO expert (in [0, 1000) range)
# MUST match generate_bc_data_sequential.py and train_bc_from_batches.py EXACTLY
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


class SimpleMonitor:
    """
    Simplified Monitor that wraps an env and tracks episode statistics.
    Avoids compatibility issues with gym/gymnasium.
    """
    def __init__(self, env):
        self.env = env
        self.rewards = []
        self.episode_infos = defaultdict(list)
        self.t_start = time.time()
        self._last_info = {}
        
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def unwrapped(self):
        return self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
    
    def reset(self, **kwargs):
        self.rewards = []
        self.episode_infos = defaultdict(list)
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        self._last_info = info
        
        # Track all info keys
        for key, val in info.items():
            if isinstance(val, (int, float, np.number)):
                self.episode_infos[key].append(val)
        
        if done:
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            
            # Add final info values
            for key, val in info.items():
                if isinstance(val, (int, float, np.number, bool)):
                    ep_info[key] = val
                    # Also add episode mean for accumulated values
                    if key in self.episode_infos and len(self.episode_infos[key]) > 0:
                        ep_info["ep_{}".format(key)] = np.mean(self.episode_infos[key])
            
            info["episode"] = ep_info
        
        return obs, reward, done, info
    
    def close(self):
        self.env.close()


def _make_rollout_env_subprocess(env_id, daytime, hard_seeds):
    """
    Factory function for creating a single rollout environment.
    This is defined at module level to be pickle-friendly for SubprocVecEnv.
    """
    from pvp.experiments.metadrive.egpo.fakehuman_env import FakeHumanEnv
    from pvp.sb3.common.monitor import Monitor
    from metadrive.component.sensors.rgb_camera import RGBCamera
    
    sensor_size = (84, 84)
    
    env_config = dict(
        image_observation=True,
        image_on_cuda=False,  # Disabled for SubprocVecEnv
        vehicle_config=dict(image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, *sensor_size)},
        horizon=1500,
        crash_vehicle_done=False,
        crash_object_done=False,
        crash_vehicle_penalty=5.0,
        crash_object_penalty=5.0,
        out_of_road_penalty=5.0,
        driving_reward=1.0,
        speed_reward=0.1,
        use_lateral_reward=False,
        daytime=daytime,
        use_render=False,
        disable_expert=True,
        start_seed=0,
        num_scenarios=1000,
    )
    
    # Create wrapper that cycles through hard seeds
    # Each subprocess env gets its own seed counter starting at env_id
    class HardSeedSubprocEnv(FakeHumanEnv):
        """Wrapper that cycles through hard seeds for this subprocess."""
        def __init__(self, config, env_id, seeds):
            super().__init__(config)
            self.hard_seeds = seeds
            self.env_id = env_id
            # Start at different offset for each env to avoid all using same seed
            self.seed_idx = env_id
            self.seeds_used = []
        
        def reset(self, **kwargs):
            seed = self.hard_seeds[self.seed_idx % len(self.hard_seeds)]
            self.seed_idx += 1
            kwargs['seed'] = seed
            self.seeds_used.append(seed)
            return super().reset(**kwargs)
    
    env = HardSeedSubprocEnv(config=env_config, env_id=env_id, seeds=hard_seeds)
    env = Monitor(env=env)
    return env


def create_rollout_envs(num_envs=20, daytime="06:10", force_no_cuda_image=False):
    """
    Create N parallel rollout environments for online evaluation.
    Uses SubprocVecEnv for N>1 (MetaDrive only allows one engine per process).
    Uses DummyVecEnv for N=1 (simpler, no subprocess overhead).
    
    Args:
        num_envs: Number of parallel environments
        daytime: Daytime setting for the environment
        force_no_cuda_image: If True, disable CUDA image even if cupy is available
    
    Returns:
        vec_env: VecEnv with N environments
        seeds_hash: Hash of the seeds for verification
        actual_env_config: Actual config values read from the environment
        use_cuda_image: Whether CUDA image acceleration is enabled
        num_envs: Number of environments (for reference)
    """
    from pvp.sb3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from functools import partial
    
    # Compute seeds hash for verification
    seeds_str = ','.join(map(str, sorted(HARD_200_SEEDS)))
    seeds_hash = hashlib.md5(seeds_str.encode()).hexdigest()[:16]
    
    # image_on_cuda is always False for parallel envs (multiprocessing limitation)
    use_cuda_image = False
    
    if num_envs == 1:
        # For N=1, use DummyVecEnv (simpler, no subprocess overhead)
        def make_single_env():
            return _make_rollout_env_subprocess(0, daytime, HARD_200_SEEDS)
        vec_env = DummyVecEnv([make_single_env])
    else:
        # For N>1, use SubprocVecEnv (each env in separate subprocess)
        # Use partial to bind arguments for pickle compatibility
        env_fns = [partial(_make_rollout_env_subprocess, i, daytime, HARD_200_SEEDS) for i in range(num_envs)]
        vec_env = SubprocVecEnv(env_fns)
    
    # Get actual config (need to get from first env, method differs by VecEnv type)
    if num_envs == 1:
        first_env = vec_env.envs[0]
        actual_env_config = {
            'traffic_density': first_env.unwrapped.config.get('traffic_density', 0.06),
            'daytime': first_env.unwrapped.config.get('daytime', daytime),
            'out_of_road_penalty': first_env.unwrapped.config.get('out_of_road_penalty', 5.0),
            'crash_vehicle_penalty': first_env.unwrapped.config.get('crash_vehicle_penalty', 5.0),
            'crash_object_penalty': first_env.unwrapped.config.get('crash_object_penalty', 5.0),
            'driving_reward': first_env.unwrapped.config.get('driving_reward', 1.0),
            'speed_reward': first_env.unwrapped.config.get('speed_reward', 0.1),
            'horizon': first_env.unwrapped.config.get('horizon', 1500),
            'image_on_cuda': use_cuda_image,
            'num_envs': num_envs,
        }
    else:
        # For SubprocVecEnv, we can't easily access individual env configs
        # Use default values (they're the same as what we set in _make_rollout_env_subprocess)
        actual_env_config = {
            'traffic_density': 0.06,  # env default
            'daytime': daytime,
            'out_of_road_penalty': 5.0,
            'crash_vehicle_penalty': 5.0,
            'crash_object_penalty': 5.0,
            'driving_reward': 1.0,
            'speed_reward': 0.1,
            'horizon': 1500,
            'image_on_cuda': use_cuda_image,
            'num_envs': num_envs,
        }
    
    return vec_env, seeds_hash, actual_env_config, use_cuda_image, num_envs


def safe_mean(arr):
    """Compute mean of a list, returning 0 if empty."""
    return np.mean(arr) if len(arr) > 0 else 0.0


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class BatchedDataLoaderIQL:
    """
    Data loader using optimize_memory approach.
    - next_obs = obs[index+1] within same batch
    - Caches a batch file and samples from it multiple times
    - Reloads batch file every `samples_per_file` samples
    """
    
    def __init__(self, data_dir: str, batch_size: int = 1024, device: str = "auto",
                 samples_per_file: int = 20):
        """
        Args:
            samples_per_file: Number of times to sample from each loaded file before switching
        """
        self.data_dir = Path(data_dir)
        self.batches_dir = self.data_dir / "batches"
        self.batch_size = batch_size
        self.samples_per_file = samples_per_file
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load metadata if exists, otherwise infer from batch files
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.transitions_per_file = self.metadata.get('batch_size', 20480)
            self.optimize_memory = self.metadata.get('optimize_memory', False)
        else:
            self.metadata = {}
            self.transitions_per_file = 20480
            self.optimize_memory = True
        
        # Dynamically scan for batch files (supports growing dataset)
        self._rescan_batch_files()
        
        self.rescan_interval = 100  # Rescan every N samples
        self.total_samples = 0
        
        # Caching state
        self.cached_data = None
        self.cached_file_idx = -1
        self.sample_count = 0
        
        print(f"BatchedDataLoaderIQL initialized:")
        print(f"  Data dir: {self.data_dir}")
        print(f"  Batch files: {self.num_batch_files}")
        print(f"  Transitions per file: {self.transitions_per_file}")
        print(f"  Total transitions: {self.total_transitions}")
        print(f"  Training batch size: {self.batch_size}")
        print(f"  Samples per file before reload: {self.samples_per_file}")
        print(f"  optimize_memory mode: {self.optimize_memory}")
        print(f"  Device: {self.device}")
    
    def _rescan_batch_files(self):
        """Rescan for batch files (supports growing dataset during training)."""
        self.batch_files = sorted(list(self.batches_dir.glob("batch_*.npz")))
        self.num_batch_files = len(self.batch_files)
        # Estimate total transitions
        self.total_transitions = self.num_batch_files * self.transitions_per_file
    
    def _load_batch_file(self, file_idx: int):
        """Load a batch file into cache with env_boundaries for correct next_obs."""
        data = np.load(self.batch_files[file_idx])
        self.cached_data = {
            'obs_image': data['obs_image'],
            'obs_state': data['obs_state'],
            'action': data['action'],
            'reward': data['reward'],
            'done': data['done'],
            'size': int(data['size']),
            'env_boundaries': data['env_boundaries'],  # Key for sequential storage
        }
        
        # Build valid indices (cannot sample last index of each env block)
        env_boundaries = self.cached_data['env_boundaries']
        valid_indices = []
        for i in range(len(env_boundaries) - 1):
            start = int(env_boundaries[i])
            end = int(env_boundaries[i + 1])
            # Valid: [start, end-2] because next_obs = obs[idx+1] and idx+1 < end
            for idx in range(start, end - 1):
                valid_indices.append(idx)
        
        self.cached_data['valid_indices'] = np.array(valid_indices, dtype=np.int64)
        self.cached_file_idx = file_idx
        self.sample_count = 0
    
    def sample(self):
        """
        Sample a batch with next_obs computed as obs[index+1].
        
        Uses env_boundaries to ensure next_obs is from same environment.
        """
        self.total_samples += 1
        
        # Periodically rescan for new batch files (supports growing dataset)
        if self.total_samples % self.rescan_interval == 0:
            self._rescan_batch_files()
        
        # Reload file if needed
        if self.cached_data is None or self.sample_count >= self.samples_per_file:
            file_idx = np.random.randint(0, self.num_batch_files)
            self._load_batch_file(file_idx)
        
        self.sample_count += 1
        
        valid_indices = self.cached_data['valid_indices']
        
        # Sample from valid indices only
        num_to_sample = min(self.batch_size, len(valid_indices))
        sampled_valid_idx = np.random.choice(len(valid_indices), size=num_to_sample, replace=False)
        indices = valid_indices[sampled_valid_idx]
        
        # Current obs
        obs_image = self.cached_data['obs_image'][indices]
        obs_state = self.cached_data['obs_state'][indices]
        
        # Next obs = obs[indices + 1] - guaranteed to be in same env block
        next_indices = indices + 1
        next_obs_image = self.cached_data['obs_image'][next_indices]
        next_obs_state = self.cached_data['obs_state'][next_indices]
        
        actions = self.cached_data['action'][indices]
        rewards = self.cached_data['reward'][indices]
        dones = self.cached_data['done'][indices]
        
        # For done transitions, next_obs is still valid (it's the terminal observation)
        # But TD target should use (1-done) * V(next_obs), so done=True makes V(next_obs)=0
        
        # Convert to tensors
        obs_image_t = torch.as_tensor(obs_image, device=self.device, dtype=torch.float32)
        obs_state_t = torch.as_tensor(obs_state, device=self.device, dtype=torch.float32)
        next_obs_image_t = torch.as_tensor(next_obs_image, device=self.device, dtype=torch.float32)
        next_obs_state_t = torch.as_tensor(next_obs_state, device=self.device, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        rewards_t = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        dones_t = torch.as_tensor(dones, device=self.device, dtype=torch.float32)
        
        observations = {'image': obs_image_t, 'state': obs_state_t}
        next_observations = {'image': next_obs_image_t, 'state': next_obs_state_t}
        
        return observations, next_observations, actions_t, rewards_t, dones_t


def main():
    parser = argparse.ArgumentParser(description="Train IQL from batched data")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--training_steps", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--save_freq", type=int, default=10000)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--samples_per_file", type=int, default=20,
                        help="How many samples from each file before reloading")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="",
                        help="Main checkpoint for IQL components (V, Q, critic)")
    parser.add_argument("--actor_ckpt", type=str, default="",
                        help="Separate checkpoint for actor network only (e.g., from BC training)")
    parser.add_argument("--v_ckpt", type=str, default="", 
                        help="Separate checkpoint for V network (value_features_extractor + value_mlp)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="iql-hard-seeds")
    parser.add_argument("--wandb_team", type=str, default="victorique")
    parser.add_argument("--exp_name", type=str, default="iql-1M")
    parser.add_argument("--iql_tau", type=float, default=0.7)
    parser.add_argument("--iql_beta", type=float, default=3.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--reward_normalize", action="store_true")
    parser.add_argument("--adv_normalize", action="store_true",
                        help="Normalize advantages before computing weights (default: False for vanilla IQL)")
    parser.add_argument("--weight_normalize", action="store_true",
                        help="Normalize weights to have mean 1 (default: False for vanilla IQL)")
    parser.add_argument("--clip_score", type=float, default=100.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (default 1.0)")
    parser.add_argument("--daytime", type=str, default="06:10",
                        help="Daytime for rollout env (should match data generation)")
    parser.add_argument("--rollout_log_freq", type=int, default=100,
                        help="How often to log rollout metrics (in training steps)")
    parser.add_argument("--rollout_step_freq", type=int, default=1,
                        help="Do rollout every N training steps (1=every step, 5=every 5 steps)")
    parser.add_argument("--no_rollout", action="store_true",
                        help="Disable online rollout collection")
    parser.add_argument("--no_cuda_image", action="store_true",
                        help="Disable CUDA image acceleration (use numpy instead of cupy)")
    parser.add_argument("--num_rollout_envs", type=int, default=20,
                        help="Number of parallel rollout environments (default=20)")
    args = parser.parse_args()
    
    if args.toy:
        print("=" * 80, flush=True)
        print("TOY MODE ENABLED", flush=True)
        print("=" * 80, flush=True)
        args.training_steps = 50
        args.batch_size = 256
        args.save_freq = 20
        args.log_freq = 10
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = Path("/data/caihy/iql_training")
    
    from pvp.utils.utils import get_time_str
    import uuid
    trial_name = f"{args.exp_name}_tau{args.iql_tau}_beta{args.iql_beta}_{get_time_str()}_{uuid.uuid4().hex[:8]}"
    trial_dir = log_dir / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80, flush=True)
    print("IQL Training from Batched Data (optimize_memory)", flush=True)
    print("=" * 80, flush=True)
    print(f"Data directory: {data_dir}", flush=True)
    print(f"Log directory: {trial_dir}", flush=True)
    print(f"Training steps: {args.training_steps}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"IQL tau: {args.iql_tau}", flush=True)
    print(f"IQL beta: {args.iql_beta}", flush=True)
    print(f"Samples per file: {args.samples_per_file}", flush=True)
    print(f"Initial memory: {get_memory_usage():.1f} MB", flush=True)
    print("=" * 80, flush=True)
    
    print("\n[1] Creating data loader...", flush=True)
    data_loader = BatchedDataLoaderIQL(
        str(data_dir), 
        batch_size=args.batch_size,
        samples_per_file=args.samples_per_file
    )
    print(f"    Memory after data loader: {get_memory_usage():.1f} MB", flush=True)
    
    print("\n[2] Testing data loading...", flush=True)
    obs, next_obs, actions, rewards, dones = data_loader.sample()
    print(f"    obs['image'] shape: {obs['image'].shape}", flush=True)
    print(f"    next_obs['image'] shape: {next_obs['image'].shape}", flush=True)
    print(f"    actions shape: {actions.shape}", flush=True)
    print(f"    rewards shape: {rewards.shape}", flush=True)
    print(f"    dones shape: {dones.shape}", flush=True)
    print(f"    rewards mean: {rewards.mean():.2f}, std: {rewards.std():.2f}", flush=True)
    print(f"    dones mean: {dones.mean():.4f}", flush=True)
    print(f"    Memory after sample: {get_memory_usage():.1f} MB", flush=True)
    
    print("\n[3] Creating IQL model...", flush=True)
    from pvp.sb3.td3.iql import IQL
    from pvp.sb3.td3.policies import TD3Policy
    from pvp.sb3.sac.our_features_extractor import OurFeaturesExtractorCNN as OurFeaturesExtractor
    
    import gymnasium as gym
    obs_space = gym.spaces.Dict({
        'image': gym.spaces.Box(low=0, high=1, shape=(84, 84, 3, 3), dtype=np.float32),
        'state': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32),
    })
    action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    
    class DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = obs_space
            self.action_space = action_space
            self.num_envs = 1
            self.metadata = {'render.modes': []}
        def reset(self, seed=None, options=None):
            return {'image': np.zeros((84, 84, 3, 3), dtype=np.float32), 'state': np.zeros(19, dtype=np.float32)}, {}
        def step(self, action):
            return {'image': np.zeros((84, 84, 3, 3), dtype=np.float32), 'state': np.zeros(19, dtype=np.float32)}, 0.0, False, False, {}
        def render(self):
            pass
        def close(self):
            pass
    
    dummy_env = DummyEnv()
    
    policy_kwargs = dict(
        features_extractor_class=OurFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=275),
        share_features_extractor=False,
        net_arch=[256,]
    )
    
    model = IQL(
        policy=TD3Policy,
        env=dummy_env,
        learning_rate=args.learning_rate,
        buffer_size=1000,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        tensorboard_log=str(trial_dir),
        verbose=2,
        seed=args.seed,
        device="auto",
        policy_kwargs=policy_kwargs,
        iql_tau=args.iql_tau,
        iql_beta=args.iql_beta,
        clip_score=args.clip_score,
        reward_normalize=args.reward_normalize,
        normalize_advantage=args.adv_normalize,
        max_grad_norm=args.max_grad_norm,
    )
    
    dummy_env.close()
    print(f"    Model device: {model.device}", flush=True)
    print(f"    Memory after model: {get_memory_usage():.1f} MB", flush=True)
    
    print("\n[4] Loading initial checkpoint...", flush=True)
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    from pvp.sb3.common.save_util import load_from_zip_file
    import zipfile
    import io
    import json
    
    # #region agent log - Debug logging helper
    def _debug_log(hypothesis_id, location, message, data=None):
        log_entry = {"hypothesisId": hypothesis_id, "location": location, "message": message, "data": data or {}, "timestamp": time.time(), "sessionId": "debug-session"}
        with open("/p0/user/caihy/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    # #endregion
    
    # #region agent log - Hypothesis A/D: Log initial value network state hash
    v_fe_hash_before = sum(p.sum().item() for p in model.value_features_extractor.parameters())
    v_mlp_hash_before = sum(p.sum().item() for p in model.value_mlp.parameters())
    critic_hash_before = sum(p.sum().item() for p in model.critic.parameters())
    _debug_log("A", "before_any_load", "Initial model state hashes", {"v_fe": v_fe_hash_before, "v_mlp": v_mlp_hash_before, "critic": critic_hash_before})
    # #endregion
    
    # Step 1: Load actor from separate checkpoint if provided
    if args.actor_ckpt:
        actor_ckpt_path = Path(args.actor_ckpt)
        if actor_ckpt_path.exists():
            print(f"    Loading ACTOR ONLY from: {actor_ckpt_path}", flush=True)
            
            # #region agent log - Hypothesis A: Log state BEFORE actor loading
            v_fe_before_actor = sum(p.sum().item() for p in model.value_features_extractor.parameters())
            critic_before_actor = sum(p.sum().item() for p in model.critic.parameters())
            actor_before = sum(p.sum().item() for p in model.actor.parameters())
            _debug_log("A", "before_actor_load", "State before actor load", {"v_fe": v_fe_before_actor, "critic": critic_before_actor, "actor": actor_before})
            # #endregion
            
            # FIX: Directly read policy.pth from the zip file to get actor weights
            with zipfile.ZipFile(actor_ckpt_path, 'r') as zf:
                # #region agent log - Hypothesis D: Log files in BC checkpoint
                _debug_log("D", "actor_ckpt_files", "Files in BC checkpoint", {"files": zf.namelist()})
                # #endregion
                
                if 'policy.pth' in zf.namelist():
                    with zf.open('policy.pth') as f:
                        policy_state = torch.load(io.BytesIO(f.read()), map_location=model.device)
                    
                    # #region agent log - Hypothesis D: Log all keys in policy.pth
                    all_keys = list(policy_state.keys())
                    _debug_log("D", "policy_pth_keys", "Keys in policy.pth", {"keys": all_keys[:30], "total": len(all_keys)})
                    # #endregion
                    
                    # Extract only actor weights (not actor_target, not critic)
                    actor_state = {k: v for k, v in policy_state.items() 
                                   if k.startswith('actor.') and not k.startswith('actor_target.')}
                    
                    # #region agent log - Hypothesis D: Log selected actor keys
                    _debug_log("D", "actor_keys_selected", "Actor keys selected for loading", {"keys": list(actor_state.keys()), "count": len(actor_state)})
                    # #endregion
                    
                    if actor_state:
                        # #region agent log - Hypothesis H: Check key matching between checkpoint and model
                        model_actor_keys = list(model.actor.state_dict().keys())
                        _debug_log("H", "actor_key_comparison", "Comparing checkpoint keys vs model keys", {
                            "ckpt_keys": list(actor_state.keys())[:10],
                            "model_keys": model_actor_keys[:10],
                            "ckpt_key_count": len(actor_state),
                            "model_key_count": len(model_actor_keys)
                        })
                        # #endregion
                        
                        # Check if keys match - need to strip 'actor.' prefix from checkpoint keys
                        # because model.actor.state_dict() keys don't have 'actor.' prefix
                        actor_state_remapped = {}
                        for k, v in actor_state.items():
                            # Remove 'actor.' prefix if present
                            new_key = k[6:] if k.startswith('actor.') else k
                            actor_state_remapped[new_key] = v
                        
                        # #region agent log - Hypothesis H: Log remapped keys
                        _debug_log("H", "actor_remapped_keys", "Remapped keys for loading", {
                            "remapped_keys": list(actor_state_remapped.keys())[:10]
                        })
                        # #endregion
                        
                        model.actor.load_state_dict(actor_state_remapped, strict=False)
                        print(f"    Actor loaded from BC checkpoint! ({len(actor_state_remapped)} keys)", flush=True)
                    else:
                        print(f"    WARNING: No actor keys found in policy.pth!", flush=True)
                else:
                    print(f"    WARNING: policy.pth not found in BC checkpoint!", flush=True)
            
            # #region agent log - Hypothesis A: Log state AFTER actor loading - verify V/Q unchanged
            v_fe_after_actor = sum(p.sum().item() for p in model.value_features_extractor.parameters())
            critic_after_actor = sum(p.sum().item() for p in model.critic.parameters())
            actor_after = sum(p.sum().item() for p in model.actor.parameters())
            v_fe_changed = abs(v_fe_after_actor - v_fe_before_actor) > 1e-6
            critic_changed = abs(critic_after_actor - critic_before_actor) > 1e-6
            actor_changed = abs(actor_after - actor_before) > 1e-6
            _debug_log("A", "after_actor_load", "State after actor load", {
                "v_fe": v_fe_after_actor, "critic": critic_after_actor, "actor": actor_after,
                "v_fe_CHANGED": v_fe_changed, "critic_CHANGED": critic_changed, "actor_CHANGED": actor_changed
            })
            # #endregion
        else:
            print(f"    WARNING: Actor checkpoint not found: {actor_ckpt_path}", flush=True)
    
    # Step 2: Load IQL components (V, Q, critic) from main checkpoint
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = script_dir / "domainAexpertBC.zip"
    
    if ckpt_path.exists():
        print(f"    Loading IQL components (V, Q, critic) from: {ckpt_path}", flush=True)
        
        with zipfile.ZipFile(ckpt_path, 'r') as zf:
            # Load value_features_extractor
            if 'value_features_extractor.pth' in zf.namelist():
                with zf.open('value_features_extractor.pth') as f:
                    v_fe_state = torch.load(io.BytesIO(f.read()), map_location=model.device)
                    model.value_features_extractor.load_state_dict(v_fe_state)
                    print(f"    value_features_extractor loaded!", flush=True)
            
            # Load value_mlp
            if 'value_mlp.pth' in zf.namelist():
                with zf.open('value_mlp.pth') as f:
                    v_mlp_state = torch.load(io.BytesIO(f.read()), map_location=model.device)
                    model.value_mlp.load_state_dict(v_mlp_state)
                    print(f"    value_mlp loaded!", flush=True)
            
            # Load critic (Q) from policy.pth
            if 'policy.pth' in zf.namelist():
                with zf.open('policy.pth') as f:
                    policy_state = torch.load(io.BytesIO(f.read()), map_location=model.device)
                    # Extract critic weights
                    critic_keys = [k for k in policy_state.keys() if 'critic' in k]
                    if critic_keys:
                        critic_state = {k: policy_state[k] for k in critic_keys}
                        model.critic.load_state_dict(critic_state, strict=False)
                        print(f"    critic (Q) loaded!", flush=True)
                    
                    # If no separate actor_ckpt, also load actor from here
                    if not args.actor_ckpt:
                        actor_keys = [k for k in policy_state.keys() if 'actor' in k or 'features_extractor' in k or 'mlp_extractor' in k]
                        if actor_keys:
                            actor_state = {k: policy_state[k] for k in actor_keys}
                            model.actor.load_state_dict(actor_state, strict=False)
                            print(f"    actor loaded (no separate actor_ckpt)!", flush=True)
            
            # FIX: Load optimizer states to prevent training instability
            # value_optimizer state
            if 'value_optimizer.pth' in zf.namelist():
                with zf.open('value_optimizer.pth') as f:
                    v_opt_state = torch.load(io.BytesIO(f.read()), map_location=model.device)
                    model.value_optimizer.load_state_dict(v_opt_state)
                    print(f"    value_optimizer loaded!", flush=True)
            
            # critic.optimizer state
            if 'critic.optimizer.pth' in zf.namelist():
                with zf.open('critic.optimizer.pth') as f:
                    critic_opt_state = torch.load(io.BytesIO(f.read()), map_location=model.device)
                    model.critic.optimizer.load_state_dict(critic_opt_state)
                    print(f"    critic.optimizer loaded!", flush=True)
        
        # #region agent log - Hypothesis B: Log final state after IQL components loaded
        v_fe_final = sum(p.sum().item() for p in model.value_features_extractor.parameters())
        v_mlp_final = sum(p.sum().item() for p in model.value_mlp.parameters())
        critic_final = sum(p.sum().item() for p in model.critic.parameters())
        actor_final = sum(p.sum().item() for p in model.actor.parameters())
        _debug_log("B", "after_iql_load", "Final state after IQL checkpoint load", {"v_fe": v_fe_final, "v_mlp": v_mlp_final, "critic": critic_final, "actor": actor_final, "actor_ckpt_used": bool(args.actor_ckpt)})
        # #endregion
        
        print(f"    IQL components loaded from {ckpt_path}!", flush=True)
    else:
        print(f"    WARNING: Checkpoint not found: {ckpt_path}", flush=True)
    
    # Load V network (and optionally Q network) from separate checkpoint if provided
    if args.v_ckpt:
        v_ckpt_path = Path(args.v_ckpt)
        if v_ckpt_path.exists():
            print(f"    Loading V network from: {v_ckpt_path}", flush=True)
            import zipfile
            import io
            
            with zipfile.ZipFile(v_ckpt_path, 'r') as zf:
                # Load value_features_extractor
                if 'value_features_extractor.pth' in zf.namelist():
                    with zf.open('value_features_extractor.pth') as f:
                        v_fe_state = torch.load(io.BytesIO(f.read()), map_location=model.device)
                        model.value_features_extractor.load_state_dict(v_fe_state)
                        print(f"    value_features_extractor loaded!", flush=True)
                
                # Load value_mlp
                if 'value_mlp.pth' in zf.namelist():
                    with zf.open('value_mlp.pth') as f:
                        v_mlp_state = torch.load(io.BytesIO(f.read()), map_location=model.device)
                        model.value_mlp.load_state_dict(v_mlp_state)
                        print(f"    value_mlp loaded!", flush=True)
                
                # Also load critic (Q) from same checkpoint to ensure Q and V are consistent
                if 'policy.pth' in zf.namelist():
                    with zf.open('policy.pth') as f:
                        policy_state = torch.load(io.BytesIO(f.read()), map_location=model.device)
                        # Extract critic weights
                        critic_keys = [k for k in policy_state.keys() if 'critic' in k]
                        if critic_keys:
                            critic_state = {k: policy_state[k] for k in critic_keys}
                            model.critic.load_state_dict(critic_state, strict=False)
                            print(f"    critic (Q) loaded from same checkpoint!", flush=True)
            
            print(f"    V & Q networks loaded from {v_ckpt_path}!", flush=True)
        else:
            print(f"    WARNING: V checkpoint not found: {v_ckpt_path}", flush=True)
    
    if args.reward_normalize:
        print("\n[5] Computing reward normalization stats...", flush=True)
        all_rewards = []
        for _ in range(100):
            _, _, _, r, _ = data_loader.sample()
            all_rewards.append(r.cpu().numpy())
        all_rewards = np.concatenate(all_rewards)
        model._reward_mean = float(np.mean(all_rewards))
        model._reward_std = float(np.std(all_rewards)) + 1e-8
        print(f"    Reward mean: {model._reward_mean:.4f}", flush=True)
        print(f"    Reward std: {model._reward_std:.4f}", flush=True)
    
    # Build comprehensive training config for wandb
    # NOTE: These should match the data generation config!
    training_env_config = {
        'env/image_observation': True,
        'env/traffic_density': 'NOT_SET_default_0.06',  # Must match data generation
        'env/random_traffic': 'NOT_SET_default',
        'env/daytime': '06:10',  # Must match data generation (BC uses 06:10)
        'env/crash_vehicle_done': False,
        'env/crash_object_done': False,
        'env/crash_vehicle_penalty': 5.0,
        'env/crash_object_penalty': 5.0,
        'env/out_of_road_penalty': 5.0,
        'env/horizon': 1500,
        'env/start_seed': 0,
        'env/num_scenarios': 1000,
        'data/hard_seeds_used': 'HARD_200_SEEDS_in_[0,1000)',
        'data/data_dir': args.data_dir,
        'model/iql_tau': args.iql_tau,
        'model/iql_beta': args.iql_beta,
        'model/learning_rate': args.learning_rate,
        'model/batch_size': args.batch_size,
        'model/reward_normalize': args.reward_normalize,
        'model/adv_normalize': args.adv_normalize,
        'model/weight_normalize': args.weight_normalize,
        'model/clip_score': args.clip_score,
        'model/max_grad_norm': args.max_grad_norm,
    }
    
    print("\n[6] Training Environment Config (should match data generation):", flush=True)
    for key, val in training_env_config.items():
        print(f"    {key}={val}", flush=True)
    
    wandb_run = None
    if args.wandb:
        print("\n[7] Initializing wandb...", flush=True)
        try:
            import wandb as wandb_module
            wandb_config = {**vars(args), **training_env_config}
            wandb_run = wandb_module.init(
                project=args.wandb_project,
                entity=args.wandb_team,
                name=trial_name,
                config=wandb_config,
            )
            print("    Wandb initialized!", flush=True)
        except Exception as e:
            print(f"    WARNING: Failed to initialize wandb: {e}", flush=True)
            args.wandb = False
    
    # Create rollout environments for online evaluation (N parallel envs)
    rollout_env = None
    rollout_obs = None
    ep_info_buffer = deque(maxlen=100)  # Store recent episode infos
    ep_extended_buffer = deque(maxlen=100)  # Store extended metrics (success_no_bad, etc.)
    rollout_seeds_hash = None
    actual_rollout_config = None
    rollout_num_envs = args.num_rollout_envs
    
    # Rollout episode-level tracking - now arrays for N parallel envs
    rollout_episode_count = 0  # Total episodes completed across all envs
    rollout_image_on_cuda = False  # Track if using CUDA image acceleration
    
    def init_rollout_tracking_arrays(n_envs):
        """Initialize per-env tracking arrays."""
        return {
            'step_in_episode': np.zeros(n_envs, dtype=np.int32),
            'had_crash_vehicle': np.zeros(n_envs, dtype=np.bool_),
            'had_crash_object': np.zeros(n_envs, dtype=np.bool_),
            'had_out_of_road': np.zeros(n_envs, dtype=np.bool_),
            'had_bad_event': np.zeros(n_envs, dtype=np.bool_),
            'arrive_dest': np.zeros(n_envs, dtype=np.bool_),
            'route_completion': np.zeros(n_envs, dtype=np.float32),
            'route_at_first_bad': np.full(n_envs, -1.0, dtype=np.float32),  # -1 means no bad event yet
            'total_crash_penalty': np.zeros(n_envs, dtype=np.float32),
            'total_out_of_road_penalty': np.zeros(n_envs, dtype=np.float32),
            'total_velocity': np.zeros(n_envs, dtype=np.float32),
            'min_vehicle_distance': np.full(n_envs, float('inf'), dtype=np.float32),
            'close_encounters': np.zeros(n_envs, dtype=np.int32),
            'safe_passes': np.zeros(n_envs, dtype=np.int32),
            'dangerous_close': np.zeros(n_envs, dtype=np.int32),
            'current_seed': np.zeros(n_envs, dtype=np.int32),
        }
    
    def reset_env_tracking(tracking, env_idx):
        """Reset tracking for a single env after episode end."""
        tracking['step_in_episode'][env_idx] = 0
        tracking['had_crash_vehicle'][env_idx] = False
        tracking['had_crash_object'][env_idx] = False
        tracking['had_out_of_road'][env_idx] = False
        tracking['had_bad_event'][env_idx] = False
        tracking['arrive_dest'][env_idx] = False
        tracking['route_completion'][env_idx] = 0.0
        tracking['route_at_first_bad'][env_idx] = -1.0  # -1 means no bad event yet
        tracking['total_crash_penalty'][env_idx] = 0.0
        tracking['total_out_of_road_penalty'][env_idx] = 0.0
        tracking['total_velocity'][env_idx] = 0.0
        tracking['min_vehicle_distance'][env_idx] = float('inf')
        tracking['close_encounters'][env_idx] = 0
        tracking['safe_passes'][env_idx] = 0
        tracking['dangerous_close'][env_idx] = 0
        tracking['current_seed'][env_idx] = 0
    
    rollout_tracking = None  # Will be initialized after env creation
    
    if not args.no_rollout:
        print(f"\n[8] Creating {rollout_num_envs} parallel rollout environments...", flush=True)
        try:
            rollout_env, rollout_seeds_hash, actual_rollout_config, rollout_image_on_cuda, rollout_num_envs = create_rollout_envs(
                num_envs=rollout_num_envs,
                daytime=args.daytime, 
                force_no_cuda_image=args.no_cuda_image
            )
            # VecEnv.reset() returns stacked obs: {'image': (N, H, W, C, S), 'state': (N, D)}
            rollout_obs = rollout_env.reset()
            
            # Initialize per-env tracking arrays
            rollout_tracking = init_rollout_tracking_arrays(rollout_num_envs)
            
            print(f"    Rollout envs created: {rollout_num_envs} parallel environments!", flush=True)
            print(f"    Seeds hash: {rollout_seeds_hash}", flush=True)
            print(f"    image_on_cuda: {rollout_image_on_cuda} (cupy available: {_cupy_available})", flush=True)
            print(f"    Obs shapes: image={rollout_obs['image'].shape}, state={rollout_obs['state'].shape}", flush=True)
            print(f"    Actual env config:", flush=True)
            for k, v in actual_rollout_config.items():
                print(f"      {k}: {v}", flush=True)
            
            # Log env config to wandb
            if args.wandb and wandb_run:
                daytime_numeric = int(args.daytime.replace(":", "")) if args.daytime else 0
                env_cfg_metrics = {
                    'env_cfg/traffic_density': actual_rollout_config['traffic_density'],
                    'env_cfg/daytime_numeric': daytime_numeric,
                    'env_cfg/out_of_road_penalty': actual_rollout_config['out_of_road_penalty'],
                    'env_cfg/crash_vehicle_penalty': actual_rollout_config['crash_vehicle_penalty'],
                    'env_cfg/crash_object_penalty': actual_rollout_config['crash_object_penalty'],
                    'env_cfg/driving_reward': actual_rollout_config['driving_reward'],
                    'env_cfg/speed_reward': actual_rollout_config['speed_reward'],
                    'env_cfg/horizon': actual_rollout_config['horizon'],
                    'env_cfg/num_seeds': len(HARD_200_SEEDS),
                    'env_cfg/seeds_hash_numeric': int(rollout_seeds_hash[:8], 16) % 1000000,
                    'env_cfg/num_rollout_envs': rollout_num_envs,
                }
                wandb_run.log(env_cfg_metrics)
                wandb_run.summary['seeds_hash'] = rollout_seeds_hash
                wandb_run.summary['daytime'] = args.daytime
                wandb_run.summary['traffic_density'] = actual_rollout_config['traffic_density']
                wandb_run.summary['num_rollout_envs'] = rollout_num_envs
                print(f"    Logged env_cfg to wandb!", flush=True)
        except Exception as e:
            print(f"    WARNING: Failed to create rollout envs: {e}", flush=True)
            import traceback
            traceback.print_exc()
            rollout_env = None
    else:
        print("\n[8] Rollout collection disabled (--no_rollout)", flush=True)
    
    print("\n" + "=" * 80, flush=True)
    print("Starting IQL Training Loop", flush=True)
    print("=" * 80, flush=True)
    
    # #region agent log - Hypothesis I: Log exact weights BEFORE training starts
    # This will help compare if value/critic weights are identical between runs
    v_fe_hash_train_start = sum(p.sum().item() for p in model.value_features_extractor.parameters())
    v_mlp_hash_train_start = sum(p.sum().item() for p in model.value_mlp.parameters())
    critic_hash_train_start = sum(p.sum().item() for p in model.critic.parameters())
    actor_hash_train_start = sum(p.sum().item() for p in model.actor.parameters())
    
    # Also check first weight values for exact matching
    v_fe_first_weight = next(model.value_features_extractor.parameters()).flatten()[:5].tolist()
    critic_first_weight = next(model.critic.parameters()).flatten()[:5].tolist()
    
    _debug_log("I", "training_start_state", "Model state at training start", {
        "v_fe_hash": v_fe_hash_train_start, 
        "v_mlp_hash": v_mlp_hash_train_start, 
        "critic_hash": critic_hash_train_start,
        "actor_hash": actor_hash_train_start,
        "v_fe_first_5_values": v_fe_first_weight,
        "critic_first_5_values": critic_first_weight,
        "actor_ckpt_used": bool(args.actor_ckpt)
    })
    print(f"    [DEBUG] V_FE hash at train start: {v_fe_hash_train_start:.6f}", flush=True)
    print(f"    [DEBUG] Critic hash at train start: {critic_hash_train_start:.6f}", flush=True)
    # #endregion
    
    from pvp.sb3.common.utils import polyak_update
    
    start_time = time.time()
    log_freq = args.log_freq
    
    value_losses = []
    critic_losses = []
    actor_losses = []
    bc_losses = []
    
    for step in range(1, args.training_steps + 1):
        obs, next_obs, actions, rewards, dones = data_loader.sample()
        
        if args.reward_normalize and model._reward_mean is not None:
            rewards = (rewards - model._reward_mean) / model._reward_std
        
        # #region agent log - Hypothesis I: Log first 3 steps to detect early divergence
        if step <= 3:
            obs_hash = obs['image'].sum().item() if isinstance(obs, dict) else obs.sum().item()
            actions_hash = actions.sum().item()
            rewards_hash = rewards.sum().item()
            _debug_log("I", f"step_{step}_data", f"Training data at step {step}", {
                "obs_hash": obs_hash, "actions_hash": actions_hash, "rewards_hash": rewards_hash
            })
        # #endregion
        
        # ========== IQL Update ==========
        # 1. Value function update
        with torch.no_grad():
            q1, q2 = model.critic(obs, actions)
            q_min = torch.min(q1, q2)
        
        value_features = model.value_features_extractor(obs)
        v_pred = model.value_mlp(value_features)
        
        # Clip q_min to prevent explosion propagation
        q_min_clipped = torch.clamp(q_min.detach(), -1000, 1000)
        
        diff = q_min_clipped - v_pred
        weight = torch.where(diff > 0, args.iql_tau, 1 - args.iql_tau)
        value_loss = (weight * (diff ** 2)).mean()
        
        # Detect explosion early
        if value_loss.item() > 1e6:
            print(f"    [WARNING] Value loss exploded: {value_loss.item():.2e}", flush=True)
        
        model.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.value_features_extractor.parameters()) + 
                                        list(model.value_mlp.parameters()), model.max_grad_norm)
        model.value_optimizer.step()
        value_losses.append(value_loss.item())
        
        # 2. Critic update with V(s') as target
        with torch.no_grad():
            next_value_features = model.value_features_extractor(next_obs)
            next_v = model.value_mlp(next_value_features)
            # Clip next_v to prevent explosion (reasonable value range for normalized rewards)
            next_v = torch.clamp(next_v, -1000, 1000)
            # For done transitions, next_v should be 0
            target_q = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * args.gamma * next_v
            # Clip target_q as well
            target_q = torch.clamp(target_q, -1000, 1000)
        
        current_q1, current_q2 = model.critic(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Detect explosion early
        if critic_loss.item() > 1e6:
            print(f"    [WARNING] Critic loss exploded: {critic_loss.item():.2e}", flush=True)
        
        model.critic.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.critic.parameters(), model.max_grad_norm)
        model.critic.optimizer.step()
        critic_losses.append(critic_loss.item())
        
        # 3. Actor update (advantage-weighted regression)
        with torch.no_grad():
            q1, q2 = model.critic(obs, actions)
            q_min = torch.min(q1, q2)
            v = model.value_mlp(model.value_features_extractor(obs))
            advantage = q_min - v
            
            # Normalize advantage by std only (following original IQL implementation)
            # DO NOT subtract mean - this preserves relative advantage ordering
            adv_std = advantage.std()
            if adv_std > 1e-8:
                normalized_advantage = advantage / adv_std
            else:
                normalized_advantage = advantage
            
            # Compute weights: exp(β * A) with clipping for stability
            exp_advantage = torch.exp(args.iql_beta * normalized_advantage)
            exp_advantage = torch.clamp(exp_advantage, max=args.clip_score)
            # Normalize weights to have mean 1 for stable gradients (optional)
            if args.weight_normalize:
                exp_advantage = exp_advantage / exp_advantage.mean()
        
        policy_actions = model.actor(obs)
        bc_loss_per_sample = ((policy_actions - actions) ** 2).sum(dim=-1)
        actor_loss = (exp_advantage.squeeze() * bc_loss_per_sample).mean()
        
        model.actor.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), model.max_grad_norm)
        model.actor.optimizer.step()
        actor_losses.append(actor_loss.item())
        bc_losses.append(bc_loss_per_sample.mean().item())
        
        polyak_update(model.critic.parameters(), model.critic_target.parameters(), model.tau)
        polyak_update(model.actor.parameters(), model.actor_target.parameters(), model.tau)
        
        model._n_updates += 1
        
        # ============ ONLINE ROLLOUT COLLECTION (N PARALLEL ENVS) ============
        # Collect one step of rollout data from N parallel envs using current policy
        # Only do rollout every N steps for speedup
        if rollout_env is not None and rollout_obs is not None and rollout_tracking is not None and step % args.rollout_step_freq == 0:
            # Increment step counter for all envs
            rollout_tracking['step_in_episode'] += 1
            
            with torch.no_grad():
                # Convert observation to tensor - NOW BATCHED (N, H, W, C, S)
                if rollout_image_on_cuda and _cupy_available:
                    if hasattr(rollout_obs['image'], 'toDlpack'):
                        image_tensor = from_dlpack(rollout_obs['image'].toDlpack())
                    else:
                        image_tensor = torch.as_tensor(rollout_obs['image'], device=model.device)
                    if image_tensor.dtype == torch.uint8:
                        image_tensor = image_tensor.float() / 255.0
                    else:
                        image_tensor = image_tensor.float()
                    rollout_obs_tensor = {
                        'image': image_tensor,  # Already (N, H, W, C, S)
                        'state': torch.as_tensor(rollout_obs['state'], device=model.device, dtype=torch.float32),
                    }
                else:
                    # Standard numpy -> torch conversion for batched obs
                    rollout_obs_tensor = {
                        'image': torch.as_tensor(rollout_obs['image'], device=model.device, dtype=torch.float32),
                        'state': torch.as_tensor(rollout_obs['state'], device=model.device, dtype=torch.float32),
                    }
                # Get actions from policy for all N envs - returns (N, 2)
                model.actor.eval()
                policy_actions = model.actor(rollout_obs_tensor).cpu().numpy()  # (N, 2)
                model.actor.train()
            
            # Step all N environments with their respective actions
            # VecEnv.step returns: obs, rewards (N,), dones (N,), infos (list of N dicts)
            # Use different variable names to avoid overwriting training batch data
            new_obs, rollout_rewards, rollout_dones, infos = rollout_env.step(policy_actions)
            
            # === PER-ENV METRIC TRACKING ===
            for i in range(rollout_num_envs):
                info = infos[i]
                
                # Get current seed on first step of episode for this env
                if rollout_tracking['step_in_episode'][i] == 1:
                    rollout_tracking['current_seed'][i] = info.get('env_seed', 0)
                
                # Track velocity for this env
                velocity = info.get('velocity', 0.0)
                if isinstance(velocity, (list, np.ndarray)):
                    velocity = np.linalg.norm(velocity)
                rollout_tracking['total_velocity'][i] += velocity
                
                # Track crash events for this env
                if info.get('crash_vehicle', False):
                    rollout_tracking['had_crash_vehicle'][i] = True
                    rollout_tracking['total_crash_penalty'][i] += 5.0
                if info.get('crash_object', False):
                    rollout_tracking['had_crash_object'][i] = True
                    rollout_tracking['total_crash_penalty'][i] += 5.0
                if info.get('out_of_road', False):
                    rollout_tracking['had_out_of_road'][i] = True
                    rollout_tracking['total_out_of_road_penalty'][i] += 5.0
                
                # Track bad event for this env - record route at FIRST bad event
                if rollout_tracking['had_crash_vehicle'][i] or rollout_tracking['had_crash_object'][i] or rollout_tracking['had_out_of_road'][i]:
                    if not rollout_tracking['had_bad_event'][i]:
                        # First bad event - record route completion at this moment
                        rollout_tracking['route_at_first_bad'][i] = info.get('route_completion', 0.0)
                    rollout_tracking['had_bad_event'][i] = True
                
                # Track success for this env
                if info.get('arrive_dest', False):
                    rollout_tracking['arrive_dest'][i] = True
                
                # Track route completion for this env
                rollout_tracking['route_completion'][i] = max(
                    rollout_tracking['route_completion'][i], 
                    info.get('route_completion', 0.0)
                )
                
                # Track traffic proximity for this env
                min_dist = info.get('min_vehicle_distance', float('inf'))
                if min_dist < float('inf'):
                    rollout_tracking['min_vehicle_distance'][i] = min(
                        rollout_tracking['min_vehicle_distance'][i], min_dist
                    )
                if min_dist < 5.0:  # Close encounter threshold
                    rollout_tracking['close_encounters'][i] += 1
                    step_crash = info.get('crash_vehicle', False) or info.get('crash_object', False)
                    if step_crash:
                        rollout_tracking['dangerous_close'][i] += 1
                    else:
                        rollout_tracking['safe_passes'][i] += 1
                
                # === EPISODE END FOR ENV i ===
                if rollout_dones[i]:
                    rollout_episode_count += 1
                    
                    # Compute extended metrics for this env
                    step_count = rollout_tracking['step_in_episode'][i]
                    success_no_bad = rollout_tracking['arrive_dest'][i] and not rollout_tracking['had_bad_event'][i]
                    rc_no_bad = rollout_tracking['route_completion'][i] if not rollout_tracking['had_bad_event'][i] else 0.0
                    # Route until first bad event (or full route if no bad event)
                    route_at_bad = rollout_tracking['route_at_first_bad'][i]
                    route_until_bad = route_at_bad if route_at_bad >= 0 else rollout_tracking['route_completion'][i]
                    avg_velocity = rollout_tracking['total_velocity'][i] / step_count if step_count > 0 else 0.0
                    close_enc = rollout_tracking['close_encounters'][i]
                    safe_pass_rate = rollout_tracking['safe_passes'][i] / close_enc if close_enc > 0 else 1.0
                    close_enc_crash_rate = rollout_tracking['dangerous_close'][i] / close_enc if close_enc > 0 else 0.0
                    
                    # Store basic ep_info from Monitor (VecEnv stores it in 'episode' key)
                    if 'episode' in info:
                        ep_info_buffer.append(info['episode'])
                    
                    # Store extended metrics
                    ep_extended = {
                        'seed': int(rollout_tracking['current_seed'][i]),
                        'success_no_bad': float(success_no_bad),
                        'success_rate': float(rollout_tracking['arrive_dest'][i]),
                        'route_completion': float(rollout_tracking['route_completion'][i]),
                        'route_completion_no_bad': float(rc_no_bad),
                        'route_until_bad': float(route_until_bad),
                        'crash_vehicle_rate': float(rollout_tracking['had_crash_vehicle'][i]),
                        'crash_object_rate': float(rollout_tracking['had_crash_object'][i]),
                        'out_of_road_rate': float(rollout_tracking['had_out_of_road'][i]),
                        'any_bad_event_rate': float(rollout_tracking['had_bad_event'][i]),
                        'crash_penalty': float(rollout_tracking['total_crash_penalty'][i]),
                        'out_of_road_penalty': float(rollout_tracking['total_out_of_road_penalty'][i]),
                        'avg_velocity': float(avg_velocity),
                        'min_vehicle_distance': float(rollout_tracking['min_vehicle_distance'][i]) if rollout_tracking['min_vehicle_distance'][i] < float('inf') else -1,
                        'close_encounters': int(rollout_tracking['close_encounters'][i]),
                        'safe_passes': int(rollout_tracking['safe_passes'][i]),
                        'dangerous_close': int(rollout_tracking['dangerous_close'][i]),
                        'safe_pass_rate': float(safe_pass_rate),
                        'close_enc_crash_rate': float(close_enc_crash_rate),
                        'episode_length': int(step_count),
                        'env_id': i,
                    }
                    ep_extended_buffer.append(ep_extended)
                    
                    # Reset tracking for this env only (VecEnv auto-resets)
                    reset_env_tracking(rollout_tracking, i)
                    
                    # Verification print for first few episodes
                    if rollout_episode_count <= 5:
                        print(f"[Rollout] Ep={rollout_episode_count} env={i} seed={ep_extended['seed']} "
                              f"success_no_bad={ep_extended['success_no_bad']:.0f} "
                              f"route={ep_extended['route_completion']:.2f} "
                              f"route_until_bad={ep_extended['route_until_bad']:.2f} "
                              f"len={ep_extended['episode_length']}", flush=True)
            
            # Update obs for next step (VecEnv already auto-reset finished envs)
            rollout_obs = new_obs
        # ====================================================
        
        if step % log_freq == 0 or step == 1:
            elapsed = time.time() - start_time
            rate = step / elapsed
            eta_hours = (args.training_steps - step) / rate / 3600 if rate > 0 else 0
            
            avg_value_loss = np.mean(value_losses[-log_freq:])
            avg_critic_loss = np.mean(critic_losses[-log_freq:])
            avg_actor_loss = np.mean(actor_losses[-log_freq:])
            avg_bc_loss = np.mean(bc_losses[-log_freq:])
            
            log_msg = f"[{step}/{args.training_steps}] V={avg_value_loss:.4f} Q={avg_critic_loss:.4f} A={avg_actor_loss:.4f} BC={avg_bc_loss:.4f} | {rate:.1f}/s ETA={eta_hours:.1f}h"
            print(log_msg, flush=True)
            
            if args.wandb and wandb_run:
                # ================================================================
                # IQL Diagnostic Metrics: Is IQL != BC?
                # ================================================================
                adv_flat = advantage.flatten()
                
                # 1. Advantage percentiles
                adv_p10 = torch.quantile(adv_flat, 0.1).item()
                adv_p25 = torch.quantile(adv_flat, 0.25).item()
                adv_p50 = torch.quantile(adv_flat, 0.5).item()
                adv_p75 = torch.quantile(adv_flat, 0.75).item()
                adv_p90 = torch.quantile(adv_flat, 0.9).item()
                
                # 2. Weight entropy: measure of uniformity
                # High entropy = uniform = BC, Low entropy = selective = IQL
                weights_normalized = exp_advantage.flatten() / exp_advantage.sum()
                weight_entropy = -(weights_normalized * torch.log(weights_normalized + 1e-10)).sum()
                max_entropy = np.log(len(weights_normalized))
                entropy_ratio = weight_entropy.item() / max_entropy  # 1.0 = uniform = BC
                
                # 3. Coefficient of Variation (CV) of weights
                weight_cv = (exp_advantage.std() / (exp_advantage.mean() + 1e-10)).item()
                
                # 4. Reward-advantage correlation
                # rewards is from training batch (size 1024), not rollout
                rewards_flat = rewards.flatten()
                rewards_centered = rewards_flat - rewards_flat.mean()
                adv_centered = adv_flat - adv_flat.mean()
                corr_num = (rewards_centered * adv_centered).sum()
                corr_denom = torch.sqrt((rewards_centered ** 2).sum()) * torch.sqrt((adv_centered ** 2).sum()) + 1e-8
                reward_advantage_corr = (corr_num / corr_denom).item()
                
                # 5. Advantage gap (high reward vs low reward samples)
                reward_median = torch.median(rewards_flat)
                high_reward_mask = rewards_flat >= reward_median
                low_reward_mask = rewards_flat < reward_median
                
                if high_reward_mask.sum() > 0 and low_reward_mask.sum() > 0:
                    adv_high_reward = adv_flat[high_reward_mask].mean().item()
                    adv_low_reward = adv_flat[low_reward_mask].mean().item()
                    adv_gap = adv_high_reward - adv_low_reward
                    weight_high_reward = exp_advantage.flatten()[high_reward_mask].mean().item()
                    weight_low_reward = exp_advantage.flatten()[low_reward_mask].mean().item()
                    weight_ratio = weight_high_reward / (weight_low_reward + 1e-10)
                else:
                    adv_high_reward = 0
                    adv_low_reward = 0
                    adv_gap = 0
                    weight_high_reward = 1
                    weight_low_reward = 1
                    weight_ratio = 1
                
                # 6. IQL-BC divergence: Using original ESS-based formula
                # ESS = (sum(w))^2 / sum(w^2), normalized by n gives ESS_ratio in [0, 1]
                # ESS_ratio = 1.0 means uniform weights (IQL = BC)
                # ESS_ratio << 1.0 means weights are concentrated (IQL != BC)
                weights_for_ess = exp_advantage.flatten()
                weights_normalized = weights_for_ess / weights_for_ess.sum()
                ess = 1.0 / (weights_normalized ** 2).sum()
                ess_ratio = ess.item() / len(weights_normalized)
                iql_bc_divergence = (1.0 - ess_ratio) + weights_for_ess.std().item()
                
                log_dict = {
                    # Basic losses
                    'train/value_loss': avg_value_loss,
                    'train/critic_loss': avg_critic_loss,
                    'train/actor_loss': avg_actor_loss,
                    'train/bc_loss': avg_bc_loss,
                    
                    # Advantage statistics (raw, before normalization)
                    'train/advantage_mean': advantage.mean().item(),
                    'train/advantage_std': advantage.std().item(),
                    'train/advantage_min': advantage.min().item(),
                    'train/advantage_max': advantage.max().item(),
                    'train/advantage_range': advantage.max().item() - advantage.min().item(),
                    
                    # Advantage percentiles
                    'train/iql_advantage_p10': adv_p10,
                    'train/iql_advantage_p25': adv_p25,
                    'train/iql_advantage_p50': adv_p50,
                    'train/iql_advantage_p75': adv_p75,
                    'train/iql_advantage_p90': adv_p90,
                    
                    # Normalized advantage stats
                    'train/norm_advantage_mean': normalized_advantage.mean().item(),
                    'train/norm_advantage_std': normalized_advantage.std().item(),
                    
                    # Exp weights (after clipping)
                    'train/exp_weight_mean': exp_advantage.mean().item(),
                    'train/exp_weight_std': exp_advantage.std().item(),
                    'train/exp_weight_min': exp_advantage.min().item(),
                    'train/exp_weight_max': exp_advantage.max().item(),
                    'train/iql_weight_cv': weight_cv,
                    
                    # Weight entropy and uniformity
                    'train/iql_weight_entropy': weight_entropy.item(),
                    'train/iql_entropy_ratio': entropy_ratio,  # 1.0 = uniform = BC
                    
                    # Reward-advantage correlation (KEY METRIC!)
                    'train/iql_reward_advantage_corr': reward_advantage_corr,
                    
                    # Advantage gap (high vs low reward)
                    'train/iql_advantage_high_reward': adv_high_reward,
                    'train/iql_advantage_low_reward': adv_low_reward,
                    'train/iql_advantage_gap': adv_gap,
                    'train/iql_weight_high_reward': weight_high_reward,
                    'train/iql_weight_low_reward': weight_low_reward,
                    'train/iql_weight_ratio': weight_ratio,
                    
                    # IQL-BC divergence (ESS-based)
                    'train/iql_bc_divergence': iql_bc_divergence,
                    'train/iql_ess_ratio': ess_ratio,  # 1.0 = uniform = BC, <1.0 = selective = IQL
                    
                    # Value estimates
                    'train/v_value_mean': v.mean().item(),
                    'train/v_value_std': v.std().item(),
                    'train/q_value_mean': q_min.mean().item(),
                    'train/q_value_std': q_min.std().item(),
                    
                    # Training info
                    'train/iql_tau': args.iql_tau,
                    'train/iql_beta': args.iql_beta,
                    'train/n_updates': model._n_updates,
                    'train/memory_mb': get_memory_usage(),
                    'train/rate_per_sec': rate,
                    'train/step': step,
                    'train/eta_hours': eta_hours,
                }
                
                # ============ ROLLOUT METRICS (ENHANCED) ============
                # Log rollout metrics from ep_info_buffer + ep_extended_buffer
                if len(ep_info_buffer) > 0 and step % args.rollout_log_freq == 0:
                    # Core metrics from Monitor
                    log_dict['rollout/ep_rew_mean'] = safe_mean([ep_info["r"] for ep_info in ep_info_buffer])
                    log_dict['rollout/ep_len_mean'] = safe_mean([ep_info["l"] for ep_info in ep_info_buffer])
                    
                    # Log ALL environment-specific metrics from Monitor (same as _dump_logs)
                    first_ep_info = ep_info_buffer[-1]
                    for k, v in first_ep_info.items():
                        if k not in ["r", "l"] and type(v) is not str:
                            try:
                                log_dict["rollout/{}_mean".format(k)] = safe_mean(
                                    [ep_info[k] for ep_info in ep_info_buffer if k in ep_info]
                                )
                            except (TypeError, ValueError, KeyError):
                                pass
                    
                    # Also log "total_*" as sums
                    for k, v in first_ep_info.items():
                        if k.startswith("total"):
                            log_dict["rollout/{}_sum".format(k)] = ep_info_buffer[-1].get(k, 0)
                    
                    # ======== EXTENDED METRICS from ep_extended_buffer ========
                    if len(ep_extended_buffer) > 0:
                        # Success metrics
                        log_dict['rollout/success_no_bad'] = safe_mean([ep['success_no_bad'] for ep in ep_extended_buffer])
                        log_dict['rollout/success_rate'] = safe_mean([ep['success_rate'] for ep in ep_extended_buffer])
                        
                        # Route completion metrics
                        log_dict['rollout/route_completion_no_bad'] = safe_mean([ep['route_completion_no_bad'] for ep in ep_extended_buffer])
                        log_dict['rollout/route_until_bad'] = safe_mean([ep['route_until_bad'] for ep in ep_extended_buffer])
                        log_dict['rollout/route_completion'] = safe_mean([ep['route_completion'] for ep in ep_extended_buffer])
                        
                        # Safety metrics
                        log_dict['rollout/crash_vehicle_rate'] = safe_mean([ep['crash_vehicle_rate'] for ep in ep_extended_buffer])
                        log_dict['rollout/crash_object_rate'] = safe_mean([ep['crash_object_rate'] for ep in ep_extended_buffer])
                        log_dict['rollout/out_of_road_rate'] = safe_mean([ep['out_of_road_rate'] for ep in ep_extended_buffer])
                        log_dict['rollout/any_bad_event_rate'] = safe_mean([ep['any_bad_event_rate'] for ep in ep_extended_buffer])
                        
                        # Reward decomposition
                        log_dict['rollout/crash_penalty_mean'] = safe_mean([ep['crash_penalty'] for ep in ep_extended_buffer])
                        log_dict['rollout/out_of_road_penalty_mean'] = safe_mean([ep['out_of_road_penalty'] for ep in ep_extended_buffer])
                        log_dict['rollout/avg_velocity_mean'] = safe_mean([ep['avg_velocity'] for ep in ep_extended_buffer])
                        
                        # Traffic proximity
                        min_dists = [ep['min_vehicle_distance'] for ep in ep_extended_buffer if ep['min_vehicle_distance'] > 0]
                        log_dict['rollout/min_vehicle_distance'] = safe_mean(min_dists) if min_dists else -1
                        log_dict['rollout/close_encounters_mean'] = safe_mean([ep['close_encounters'] for ep in ep_extended_buffer])
                        log_dict['rollout/safe_pass_rate'] = safe_mean([ep['safe_pass_rate'] for ep in ep_extended_buffer])
                        log_dict['rollout/close_enc_crash_rate'] = safe_mean([ep['close_enc_crash_rate'] for ep in ep_extended_buffer])
                        
                        # Difficulty segment analysis (if enough episodes)
                        if len(ep_extended_buffer) >= 5:
                            sorted_eps = sorted([ep for ep in ep_extended_buffer if ep['seed'] is not None], 
                                               key=lambda x: HARD_200_SEEDS.index(x['seed']) if x['seed'] in HARD_200_SEEDS else 999)
                            if len(sorted_eps) >= 5:
                                num_per_seg = len(sorted_eps) // 5
                                segments = ['hardest', 'hard', 'medium', 'easy', 'easiest']
                                for seg_idx, seg_name in enumerate(segments):
                                    start_idx = seg_idx * num_per_seg
                                    end_idx = min(start_idx + num_per_seg, len(sorted_eps))
                                    if start_idx < len(sorted_eps):
                                        seg_eps = sorted_eps[start_idx:end_idx]
                                        log_dict[f'rollout/segment_{seg_name}_success'] = safe_mean([ep['success_rate'] for ep in seg_eps])
                                        log_dict[f'rollout/segment_{seg_name}_crash'] = safe_mean([ep['crash_vehicle_rate'] for ep in seg_eps])
                    # ===========================================================
                    
                    # Print rollout summary (enhanced)
                    if step % (args.rollout_log_freq * 5) == 0:
                        succ_no_bad = log_dict.get('rollout/success_no_bad', 0)
                        rc_no_bad = log_dict.get('rollout/route_completion_no_bad', 0)
                        route_until_bad = log_dict.get('rollout/route_until_bad', 0)
                        route_comp = log_dict.get('rollout/route_completion', 0)
                        crash = log_dict.get('rollout/crash_vehicle_rate', 0)
                        out_road = log_dict.get('rollout/out_of_road_rate', 0)
                        crash_pen = log_dict.get('rollout/crash_penalty_mean', 0)
                        safe_pass = log_dict.get('rollout/safe_pass_rate', 0)
                        print(f"    [Rollout] Ep={len(ep_extended_buffer)} R={log_dict.get('rollout/ep_rew_mean', 0):.1f} RC={route_comp*100:.0f}% RCUntilBad={route_until_bad*100:.0f}% "
                              f"SuccNoBad={succ_no_bad:.0%} RCNoBad={rc_no_bad:.0%} "
                              f"Crash={crash:.0%} OoR={out_road:.0%} CrashPen={crash_pen:.1f} SafePass={safe_pass:.0%}", flush=True)
                # =========================================
                
                wandb_run.log(log_dict, step=step)
        
        if step % args.save_freq == 0:
            ckpt_path = trial_dir / f"iql_step_{step:06d}.zip"
            model.save(str(ckpt_path))
            print(f"    [SAVE] Checkpoint saved: {ckpt_path}", flush=True)
    
    final_path = trial_dir / "iql_final.zip"
    model.save(str(final_path))
    print(f"\n[FINAL] Saved: {final_path}", flush=True)
    
    print("\n" + "=" * 80, flush=True)
    print("Training Summary", flush=True)
    print("=" * 80, flush=True)
    print(f"    Total steps: {args.training_steps}", flush=True)
    print(f"    Training time: {(time.time() - start_time)/60:.1f} min", flush=True)
    print(f"    Final memory: {get_memory_usage():.1f} MB", flush=True)
    print("=" * 80, flush=True)
    
    # Cleanup rollout env
    if rollout_env is not None:
        print("\n[CLEANUP] Closing rollout environment...", flush=True)
        rollout_env.close()
        print("    Rollout env closed!", flush=True)
    
    if args.wandb and wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
