"""
Evaluate baseline models (pretrained or expert) on eval_env.
Uses the same EvalCallback as train_bc_metadrive_online.py for comprehensive metrics.
Uploads results to wandb.

Usage:
    python eval_baseline.py --model pretrained --n_eval_episodes 200
    python eval_baseline.py --model expert --n_eval_episodes 200
"""

import argparse
import os
import uuid
import numpy as np
import sys
import gymnasium
sys.modules['gym'] = gymnasium

from pathlib import Path

from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.vec_env import SubprocVecEnv
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.haco import HACOReplayBuffer
from pvp.sb3.td3.policies import TD3Policy
from pvp.sb3.td3.td3 import TD3
from pvp.sb3.sac.our_features_extractor import OurFeaturesExtractorCNN as OurFeaturesExtractor
from pvp.utils.utils import get_time_str
from metadrive.component.sensors.rgb_camera import RGBCamera


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models")
    parser.add_argument("--model", type=str, required=True, choices=["pretrained", "expert"],
                        help="Which model to evaluate: 'pretrained' (RGB) or 'expert' (lidar)")
    parser.add_argument("--n_eval_episodes", type=int, default=200,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=1000,
                        help="Environment seed for evaluation")
    parser.add_argument("--wandb_project", type=str, default="0122mainexpfull",
                        help="Wandb project name")
    parser.add_argument("--wandb_team", type=str, default="victorique",
                        help="Wandb team name")
    parser.add_argument("--log_dir", type=str, default="/home/caihy/pvp",
                        help="Log directory")
    # Penalty parameters (same as train_bc_metadrive_online.py)
    parser.add_argument("--crash_vehicle_penalty", type=float, default=5.0)
    parser.add_argument("--crash_object_penalty", type=float, default=5.0)
    parser.add_argument("--out_of_road_penalty", type=float, default=5.0)
    parser.add_argument("--crash_vehicle_cost", type=float, default=1.0)
    parser.add_argument("--crash_object_cost", type=float, default=1.0)
    parser.add_argument("--out_of_road_cost", type=float, default=1.0)
    args = parser.parse_args()
    
    # ===== Setup experiment naming =====
    experiment_batch_name = f"eval_{args.model}_baseline"
    trial_name = f"{experiment_batch_name}_{get_time_str()}_{uuid.uuid4().hex[:8]}"
    print(f"Trial name: {trial_name}")
    
    log_dir = args.log_dir
    experiment_dir = Path(log_dir) / Path("runs") / experiment_batch_name
    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=True)
    print(f"Logging to {trial_dir}")
    
    # ===== Environment config =====
    sensor_size = (84, 84)
    num_eval_envs = 10
    
    def _make_eval_env():
        eval_env_config = dict(
            use_render=False,
            manual_control=False,
            start_seed=args.seed,
            horizon=1500,
            image_observation=True,
            vehicle_config=dict(image_source="rgb_camera"),
            sensors={"rgb_camera": (RGBCamera, *sensor_size)},
            stack_size=3,
            interface_panel=["rgb_camera", "dashboard"],
            daytime="06:10",
            crash_vehicle_done=False,
            crash_object_done=False,
            cost_to_reward=False,
            crash_vehicle_penalty=args.crash_vehicle_penalty,
            crash_object_penalty=args.crash_object_penalty,
            out_of_road_penalty=args.out_of_road_penalty,
            crash_vehicle_cost=args.crash_vehicle_cost,
            crash_object_cost=args.crash_object_cost,
            out_of_road_cost=args.out_of_road_cost,
        )
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        eval_env = HumanInTheLoopEnv(config=eval_env_config)
        eval_env = Monitor(env=eval_env, filename=str(trial_dir))
        return eval_env
    
    # ===== Policy config (same as train_bc_metadrive_online.py) =====
    policy_kwargs = dict(
        features_extractor_class=OurFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=275),
        share_features_extractor=False,
        net_arch=[256,]
    )
    
    # ===== Create a temporary environment for model initialization =====
    print("Creating temporary environment for model initialization...")
    temp_eval_env = SubprocVecEnv([_make_eval_env] * 1)
    
    # ===== Create model =====
    trainer_config = dict(
        policy=TD3Policy,
        replay_buffer_class=HACOReplayBuffer,
        replay_buffer_kwargs=dict(),
        policy_kwargs=policy_kwargs,
        env=temp_eval_env,
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
        tensorboard_log=trial_dir,
        create_eval_env=False,
        verbose=2,
        seed=args.seed,
        device="auto",
        buffer_size=1000,
    )
    
    model = TD3(**trainer_config)
    
    # ===== Load model weights =====
    if args.model == "pretrained":
        print("=" * 80)
        print("Evaluating PRETRAINED model (RGB observations)")
        print("=" * 80)
        
        pretrained_ckpt = Path("/home/caihy/pvp/pretrained.zip")
        if pretrained_ckpt.exists():
            print(f"Loading pretrained model from {pretrained_ckpt}...")
            from pvp.sb3.common.save_util import load_from_zip_file
            data, params, pytorch_variables = load_from_zip_file(pretrained_ckpt, device=model.device, print_system_info=False)
            model.set_parameters(params, exact_match=False, device=model.device)
            print("Pretrained model loaded successfully!")
        else:
            print(f"ERROR: Pretrained checkpoint {pretrained_ckpt} not found!")
            return
    else:  # expert
        print("=" * 80)
        print("Evaluating EXPERT model behavior via pretrained policy wrapper")
        print("NOTE: Expert uses lidar obs internally, but we evaluate on RGB env for consistency")
        print("=" * 80)
        
        # For expert evaluation, we'll use a custom wrapper that internally uses the expert
        # but the evaluation framework expects RGB observations
        # We load a dummy pretrained model but override predict() to use expert
        pretrained_ckpt = Path("/home/caihy/pvp/pretrained.zip")
        if pretrained_ckpt.exists():
            from pvp.sb3.common.save_util import load_from_zip_file
            data, params, pytorch_variables = load_from_zip_file(pretrained_ckpt, device=model.device, print_system_info=False)
            model.set_parameters(params, exact_match=False, device=model.device)
        
        # Load expert and override predict method
        from pvp.experiments.metadrive.egpo.fakehuman_env import get_expert
        print("Loading expert model (PPO with lidar observations)...")
        expert = get_expert()
        print("Expert model loaded!")
        
        # Create a wrapper class that uses expert for prediction
        class ExpertWrapper:
            def __init__(self, expert_model, rgb_model):
                self.expert = expert_model
                self.rgb_model = rgb_model
                # Copy attributes needed by EvalCallback
                self.device = rgb_model.device
                self.policy = rgb_model.policy
                self.actor = rgb_model.actor
                self.critic = rgb_model.critic
                self.num_timesteps = 0
                self._n_updates = 0
                self.logger = rgb_model.logger
                self.env = rgb_model.env
                
            def predict(self, observation, state=None, episode_start=None, deterministic=True):
                # Expert expects lidar observation, but we receive RGB
                # Get lidar observation from env info if available
                # For now, just use the expert action based on RGB (will use random action)
                # This is a limitation - expert uses lidar, not RGB
                import torch
                # Return random action since expert can't use RGB
                action = np.random.uniform(-1, 1, size=(2,))
                return action, state
        
        # Note: Expert evaluation on RGB env is not meaningful
        # Let's create a separate expert evaluation with lidar env
        print("WARNING: Expert uses lidar observations, not RGB!")
        print("Creating lidar-based evaluation environment for expert...")
        
        # Close temp RGB env
        temp_eval_env.close()
        
        # Create lidar-based env for expert
        def _make_expert_eval_env():
            eval_env_config = dict(
                use_render=False,
                manual_control=False,
                start_seed=args.seed,
                horizon=1500,
                image_observation=False,  # Lidar observation for expert
                crash_vehicle_done=False,
                crash_object_done=False,
                cost_to_reward=False,
                crash_vehicle_penalty=args.crash_vehicle_penalty,
                crash_object_penalty=args.crash_object_penalty,
                out_of_road_penalty=args.out_of_road_penalty,
            )
            from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
            eval_env = HumanInTheLoopEnv(config=eval_env_config)
            eval_env = Monitor(env=eval_env, filename=str(trial_dir))
            return eval_env
        
        temp_eval_env = SubprocVecEnv([_make_expert_eval_env] * 1)
        model = expert  # Use expert directly
        model.env = temp_eval_env
    
    # Close temp env
    temp_eval_env.close()
    
    # ===== Create eval environment =====
    print(f"Creating {num_eval_envs} evaluation environments...")
    if args.model == "pretrained":
        eval_env = SubprocVecEnv([_make_eval_env] * num_eval_envs)
    else:
        # Expert uses lidar env
        def _make_expert_eval_env():
            eval_env_config = dict(
                use_render=False,
                manual_control=False,
                start_seed=args.seed,
                horizon=1500,
                image_observation=False,
                crash_vehicle_done=False,
                crash_object_done=False,
                cost_to_reward=False,
                crash_vehicle_penalty=args.crash_vehicle_penalty,
                crash_object_penalty=args.crash_object_penalty,
                out_of_road_penalty=args.out_of_road_penalty,
            )
            from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
            eval_env = HumanInTheLoopEnv(config=eval_env_config)
            eval_env = Monitor(env=eval_env, filename=str(trial_dir))
            return eval_env
        eval_env = SubprocVecEnv([_make_expert_eval_env] * num_eval_envs)
    
    model.env = eval_env
    
    # ===== Setup callbacks with wandb =====
    callbacks = []
    callbacks.append(
        WandbCallback(
            trial_name=trial_name,
            exp_name=experiment_batch_name,
            team_name=args.wandb_team,
            project_name=args.wandb_project,
            config={
                "model": args.model,
                "n_eval_episodes": args.n_eval_episodes,
                "seed": args.seed,
                "trainer_config": trainer_config if args.model == "pretrained" else {"model": "PPO_expert"},
            }
        )
    )
    callbacks = CallbackList(callbacks)
    
    # ===== Setup evaluation using _setup_learn (same as train_bc_metadrive_online.py) =====
    print(f"Setting up evaluation with {args.n_eval_episodes} episodes...")
    
    # Use _setup_learn to create EvalCallback with all the metrics
    total_timesteps = 1  # We only need one evaluation
    eval_freq = 1  # Evaluate immediately
    
    _, callback = model._setup_learn(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        callback=callbacks,
        eval_freq=eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        eval_log_path=str(trial_dir),
        reset_num_timesteps=True,
        tb_log_name=experiment_batch_name,
    )
    
    # Start training callbacks (initializes wandb, etc.)
    callback.on_training_start(locals(), globals())
    
    # ===== Run evaluation =====
    print("=" * 80)
    print(f"Running evaluation: {args.n_eval_episodes} episodes")
    print("=" * 80)
    
    # Trigger evaluation by calling on_step
    model.num_timesteps = 0
    callback.on_step()
    
    # End training
    callback.on_training_end()
    
    # ===== Cleanup =====
    eval_env.close()
    
    print("=" * 80)
    print("Evaluation completed!")
    print(f"Results uploaded to wandb project: {args.wandb_project}")
    print(f"Logs saved to: {trial_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
