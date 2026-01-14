import argparse
import os
import uuid
from pathlib import Path
import sys
import gymnasium
sys.modules['gym'] = gymnasium

from pvp.experiments.metadrive.egpo.fakehuman_env import FakeHumanEnv
from pvp.pvp_td3 import PVPTD3
from pvp.sb3.td3.td3 import TD3
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.vec_env import SubprocVecEnv
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.haco import HACOReplayBuffer
from pvp.sb3.td3.policies import TD3Policy
from pvp.utils.shared_control_monitor import SharedControlMonitor
from pvp.utils.utils import get_time_str
from pvp.sb3.sac.our_features_extractor import OurFeaturesExtractorCNN as OurFeaturesExtractor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", default="bc_metadrive_online", type=str, help="The name for this batch of experiments."
    )
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--learning_starts", default=0, type=int)
    parser.add_argument("--save_freq", default=50000000, type=int)
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="td3", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="victorique", help="The team name for wandb.")
    parser.add_argument("--log_dir", type=str, default="/home/caihy/pvp", help="Folder to store the logs.")
    parser.add_argument("--free_level", type=float, default=0.95)
    parser.add_argument("--ckpt", default="", type=str)
    parser.add_argument("--data_collection_timesteps", default=20000, type=int, help="Total timesteps for data collection.")
    parser.add_argument("--bc_training_timesteps", default=1000000000, type=int, help="Total timesteps for BC training (can be very large).")
    parser.add_argument("--train_freq", default=1, type=int, help="Train every N steps.")
    parser.add_argument("--gradient_steps", default=1, type=int, help="Number of gradient steps per training update.")
    parser.add_argument("--eval_freq", default=1000, type=int, help="Evaluate policy every N steps.")
    parser.add_argument("--n_eval_episodes", default=200, type=int, help="Number of episodes for evaluation.")
    parser.add_argument("--toy", action="store_true", help="Use toy/debug mode with small numbers.")
    args = parser.parse_args()
    
    # Apply toy mode settings if enabled
    if args.toy:
        print("=" * 80)
        print("TOY MODE ENABLED - Using small numbers for debugging")
        print("=" * 80)
        args.data_collection_timesteps = 2000
        args.bc_training_timesteps = 2000
        args.batch_size = 64
        args.eval_freq = 20
        args.n_eval_episodes = 4
        args.save_freq = 1000
        # num_envs will be set to 2 in the environment setup

    # ===== Set up some arguments =====
    experiment_batch_name = "{}_freelevel{}".format(args.exp_name, args.free_level)
    seed = args.seed
    trial_name = "{}_{}_{}".format(experiment_batch_name, get_time_str(), uuid.uuid4().hex[:8])
    print("Trial name is set to: ", trial_name)

    use_wandb = True
    project_name = args.wandb_project
    team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    log_dir = args.log_dir
    experiment_dir = Path(log_dir) / Path("runs") / experiment_batch_name

    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=False)  # Avoid overwritting old experiment
    print(f"We start logging training data into {trial_dir}")

    free_level = args.free_level
    from metadrive.component.sensors.rgb_camera import RGBCamera
    sensor_size = (84, 84)
    
    # ===== Setup the config =====
    # Shared policy config
    policy_kwargs = dict(
        features_extractor_class=OurFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=275),
        share_features_extractor=False, 
        net_arch=[256,]
    )
    
    # Environment config
    env_config = dict(
        free_level=free_level,
        image_observation=True, 
        vehicle_config=dict(image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, *sensor_size)},
        stack_size=3,
        interface_panel=["rgb_camera", "dashboard"],
        daytime="06:10",
        use_render=False,
        disable_expert=False,  # Ensure IDMPolicy is used
    )

    # ===== Setup the training environment =====
    num_envs = 1 if args.toy else 50
    num_eval_envs = 1 if args.toy else 10
    
    def _make_train_env():
        train_env = FakeHumanEnv(config=env_config)
        train_env = Monitor(env=train_env, filename=str(trial_dir))
        train_env = SharedControlMonitor(env=train_env, folder=trial_dir / "data", prefix=trial_name)
        return train_env
    
    train_env = SubprocVecEnv([_make_train_env] * num_envs)

    # ===== Setup eval environment function =====
    # Note: eval_env will be created in Phase 2, not here, to avoid unnecessary process creation
    def _make_eval_env():
        eval_env_config = dict(
            use_render=False,
            manual_control=False,
            start_seed=1000,  # Different seed for heldout test set
            horizon=1500,
            image_observation=True, 
            vehicle_config=dict(image_source="rgb_camera"),
            sensors={"rgb_camera": (RGBCamera, *sensor_size)},
            stack_size=3,
            interface_panel=["rgb_camera", "dashboard"],
            daytime="06:10",
        )
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        eval_env = HumanInTheLoopEnv(config=eval_env_config)
        eval_env = Monitor(env=eval_env, filename=str(trial_dir))
        return eval_env

    # ===== Setup PVPTD3 for data collection =====
    # PVPTD3 will only collect data (train() is commented in learn())
    pvp_config = dict(
        adaptive_batch_size="False",
        bc_loss_weight=1.0,
        only_bc_loss="True",
        with_human_proxy_value_loss="False",
        with_agent_proxy_value_loss="False",
        add_bc_loss="True",
        use_balance_sample=True,
        agent_data_ratio=1.0,
        policy=TD3Policy,
        replay_buffer_class=HACOReplayBuffer,
        replay_buffer_kwargs=dict(),
        policy_kwargs=policy_kwargs,
        env=train_env,
        learning_rate=1e-4,
        q_value_bound=1,
        optimize_memory_usage=True,
        buffer_size=args.data_collection_timesteps if not args.toy else 2000,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        action_noise=None,
        tensorboard_log=trial_dir,
        create_eval_env=False,
        verbose=2,
        seed=seed,
        device="auto",
    )
    
    data_collector = PVPTD3(**pvp_config)
    if args.ckpt:
        ckpt = Path(args.ckpt)
        print(f"Loading checkpoint from {ckpt}!")
        from pvp.sb3.common.save_util import load_from_zip_file
        data, params, pytorch_variables = load_from_zip_file(ckpt, device=data_collector.device, print_system_info=False)
        data_collector.set_parameters(params, exact_match=False, device=data_collector.device)

    # ===== Setup TD3 for BC training =====
    # TD3 will use the same policy and network structure
    td3_config = dict(
        policy=TD3Policy,
        replay_buffer_class=HACOReplayBuffer,
        replay_buffer_kwargs=dict(),
        policy_kwargs=policy_kwargs,
        env=train_env,
        learning_rate=1e-4,
        optimize_memory_usage=True,
        learning_starts=0,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=args.gradient_steps,
        action_noise=None,
        policy_delay=2,
        tensorboard_log=trial_dir,
        create_eval_env=False,
        verbose=2,
        seed=seed,
        device="auto",
        buffer_size=args.data_collection_timesteps if not args.toy else 200,
    )
    
    bc_trainer = TD3(**td3_config)
    
    # Load initial policy from checkpoint
    initial_ckpt = Path("/home/caihy/pvp/rgbsr70.zip")
    if initial_ckpt.exists():
        print(f"Loading initial policy for bc_trainer from {initial_ckpt}!")
        from pvp.sb3.common.save_util import load_from_zip_file
        data, params, pytorch_variables = load_from_zip_file(initial_ckpt, device=bc_trainer.device, print_system_info=False)
        bc_trainer.set_parameters(params, exact_match=False, device=bc_trainer.device)
    else:
        print(f"Warning: Initial checkpoint {initial_ckpt} not found! bc_trainer will start with random weights.")

    # ===== Setup callbacks =====
    # Phase 1: No wandb, only data collection
    save_freq = args.save_freq
    phase1_callbacks = [
        CheckpointCallback(name_prefix="rl_model", verbose=2, save_freq=save_freq, save_path=str(trial_dir / "models"))
    ]
    # Do NOT add WandbCallback for Phase 1 - wandb will be enabled in Phase 2
    callbacks = CallbackList(phase1_callbacks)

    # ===== Custom learn loop: collect data with PVPTD3 and train with TD3 =====
    # Phase 1: Data collection
    data_collection_timesteps = args.data_collection_timesteps
    bc_training_timesteps = args.bc_training_timesteps
    
    total_timesteps, callback = data_collector._setup_learn(
        data_collection_timesteps,
        None,  # eval_env for evaluation
        callbacks,
        -1,  # eval_freq
        args.n_eval_episodes,  # n_eval_episodes
        str(trial_dir),  # eval_log_path
        True,  # reset_num_timesteps
        experiment_batch_name,  # tb_log_name
    )
    
    callback.on_training_start(locals(), globals())
    
    print("=" * 80)
    print("Phase 1: Data Collection ONLY (No Training, No Evaluation)")
    print("=" * 80)
    print(f"PVPTD3 will collect {data_collection_timesteps} timesteps from FakeHumanEnv (IDMPolicy actions)")
    print("No training or evaluation will be performed during data collection")
    print("=" * 80)
    
    # Phase 1: Data collection ONLY - no training, no evaluation
    while data_collector.num_timesteps < data_collection_timesteps:
        # Collect rollouts using PVPTD3 (data goes to human_data_buffer)
        rollout = data_collector.collect_rollouts(
            data_collector.env,
            train_freq=data_collector.train_freq,
            action_noise=data_collector.action_noise,
            callback=callback,
            learning_starts=data_collector.learning_starts,
            replay_buffer=data_collector.replay_buffer,
            log_interval=1,
        )
        
        if rollout.continue_training is False:
            break
        
        # Log progress
        log_interval = 100 if args.toy else 1000
        if data_collector.num_timesteps % log_interval == 0:
            print(f"Data Collection: {data_collector.num_timesteps}/{data_collection_timesteps}, "
                  f"Buffer size: {data_collector.human_data_buffer.pos * num_envs}")
    
    print("=" * 80)
    print(f"Phase 1 completed! Collected {data_collector.human_data_buffer.pos} transitions")
    print("=" * 80)
    
    # Close train_env from Phase 1 to free resources
    print("Closing Phase 1 training environment...")
    train_env.close()
    
    # Phase 2: BC training from scratch on collected data
    print("=" * 80)
    print("Phase 2: BC Training from Scratch on Collected Data")
    print("=" * 80)
    print(f"Starting BC training from scratch for {bc_training_timesteps} timesteps using collected data")
    print(f"Evaluation will be performed every {args.eval_freq} steps via EvalCallback")
    print("=" * 80)
    
    # Create eval_env for Phase 2 (not needed in Phase 1)
    print("Creating eval environment for Phase 2...")
    eval_env = SubprocVecEnv([_make_eval_env] * num_eval_envs)
    
    # IMPORTANT: bc_trainer.env still points to train_env which was closed after Phase 1
    # _setup_learn() will call self.env.reset(), so we need to set bc_trainer.env to a valid env
    # Since BC training doesn't actually need train_env (only uses replay buffer), we can use eval_env
    # or create a new train_env. Using eval_env is simpler.
    print("Updating bc_trainer.env to use eval_env (BC training doesn't need train_env)...")
    bc_trainer.env = eval_env
    
    # Share the human_data_buffer between PVPTD3 and TD3
    # This ensures TD3 trains on the same data collected by PVPTD3
    bc_trainer.replay_buffer = data_collector.human_data_buffer
    
    # Setup callbacks for Phase 2 using bc_trainer (so EvalCallback model points to bc_trainer)
    # Create new callbacks for Phase 2
    phase2_callbacks = [
        CheckpointCallback(name_prefix="rl_model", verbose=2, save_freq=save_freq, save_path=str(trial_dir / "models"))
    ]
    if use_wandb:
        phase2_callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=experiment_batch_name,
                team_name=team_name,
                project_name=project_name,
                config={"pvp_config": pvp_config, "td3_config": td3_config}
            )
        )
    phase2_callbacks = CallbackList(phase2_callbacks)
    
    # Setup learn for Phase 2 - this will create EvalCallback and attach it to bc_trainer
    # reset_num_timesteps=True to start training from scratch (timesteps reset to 0)
    phase2_total_timesteps, phase2_callback = bc_trainer._setup_learn(
        bc_training_timesteps,
        eval_env,  # eval_env for evaluation (recreated for Phase 2)
        phase2_callbacks,
        args.eval_freq,  # eval_freq
        args.n_eval_episodes,  # n_eval_episodes
        str(trial_dir),  # eval_log_path
        True,  # reset_num_timesteps=True - train from scratch
        experiment_batch_name,  # tb_log_name
    )
    
    phase2_callback.on_training_start(locals(), globals())
    
    bc_training_current_timesteps = 0  # Start from 0 (train from scratch)
    last_phase2_eval_timesteps = 0  # Track last eval in Phase 2
    
    # Find EvalCallback in the callback list to manually trigger evaluation
    from pvp.sb3.common.callbacks import EvalCallback
    eval_callback = None
    if hasattr(phase2_callback, 'callbacks'):
        for cb in phase2_callback.callbacks:
            if isinstance(cb, EvalCallback):
                eval_callback = cb
                break
            elif hasattr(cb, 'callbacks'):  # Nested CallbackList
                for sub_cb in cb.callbacks:
                    if isinstance(sub_cb, EvalCallback):
                        eval_callback = sub_cb
                        break
                if eval_callback is not None:
                    break
    
    while bc_training_current_timesteps < bc_training_timesteps:
        # Only train, no data collection
        if True:
            bc_trainer.num_timesteps = bc_training_current_timesteps
            bc_trainer.train(batch_size=args.batch_size, gradient_steps=args.gradient_steps)
            bc_training_current_timesteps += args.train_freq
            
            # Call callback.on_step() for other callbacks (e.g., CheckpointCallback, WandbCallback)
            phase2_callback.on_step()
            
            # Manually trigger EvalCallback based on timesteps (not n_calls)
            # This ensures evaluation happens every eval_freq timesteps
            if (bc_training_current_timesteps - last_phase2_eval_timesteps >= args.eval_freq and 
                eval_callback is not None):
                # Temporarily adjust n_calls to trigger evaluation
                # EvalCallback checks: if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
                original_n_calls = eval_callback.n_calls
                # Calculate how many eval cycles we should have done by now in Phase 2
                eval_cycles = bc_training_current_timesteps // args.eval_freq
                # Set n_calls to trigger evaluation (it will be incremented by on_step() above)
                eval_callback.n_calls = eval_cycles * args.eval_freq - 1
                # Trigger evaluation (this will increment n_calls and check the condition)
                eval_callback._on_step()
                last_phase2_eval_timesteps = bc_training_current_timesteps
        
        # Log progress
        log_interval = 100 if args.toy else 10000
        if bc_training_current_timesteps % log_interval == 0:
            print(f"BC Training: {bc_training_current_timesteps}/{bc_training_timesteps}, "
                  f"Training updates: {bc_trainer._n_updates}")
    
    callback.on_training_end()
    
    print("=" * 80)
    print("Training completed!")
    print("=" * 80)
    print(f"Data collection: {data_collection_timesteps} timesteps")
    print(f"BC training: {bc_training_timesteps} timesteps")
    print(f"Final buffer size: {data_collector.human_data_buffer.pos}")
    print(f"Total training updates: {bc_trainer._n_updates}")
