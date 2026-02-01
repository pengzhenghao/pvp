"""
Train IQL from batched data format (optimize_memory style).

Key: next_obs = obs[index+1], NOT separately stored.
- Episode boundaries (done=True) require special handling
- Caches loaded batch files to reduce I/O

Usage:
    python train_iql_from_batches.py --data_dir /data/caihy/bc_data_1M --log_dir /data/caihy/iql_training
"""

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
import sys
import time
import json
import torch
import torch.nn.functional as F
import psutil
import gymnasium
sys.modules['gym'] = gymnasium


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
    parser.add_argument("--ckpt", type=str, default="")
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
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = script_dir / "domainAexpertBC.zip"
    
    if ckpt_path.exists():
        print(f"    Loading actor/critic from: {ckpt_path}", flush=True)
        from pvp.sb3.common.save_util import load_from_zip_file
        data, params, pytorch_variables = load_from_zip_file(ckpt_path, device=model.device, print_system_info=False)
        model.set_parameters(params, exact_match=False, device=model.device)
        print(f"    Actor/Critic loaded!", flush=True)
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
        'env/daytime': '08:30',  # Must match data generation
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
    
    print("\n" + "=" * 80, flush=True)
    print("Starting IQL Training Loop", flush=True)
    print("=" * 80, flush=True)
    
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
                
                wandb_run.log({
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
                }, step=step)
        
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
    
    if args.wandb and wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
