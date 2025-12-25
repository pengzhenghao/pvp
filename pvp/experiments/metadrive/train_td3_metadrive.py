import argparse
import os
from pathlib import Path

from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.vec_env import SubprocVecEnv
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.td3.policies import TD3Policy
from pvp.sb3.td3.td3 import TD3, ReplayBuffer
from pvp.sb3.haco import HACOReplayBuffer
from pvp.utils.utils import get_time_str


def register_env(make_env_fn, env_name):
    from gym.envs.registration import register
    register(id=env_name, entry_point=make_env_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="td3_metadrive", type=str, help="The name for this batch of experiments.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="TD3Offline", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="victorique", help="The team name for wandb.")
    parser.add_argument("--ckpt", default="/home/caihy/pvp/runs/trainfreq5/trainfreq5_2025-12-19_00-13-10_faec4da3/models/rl_model_6370000_steps.zip", type=str, help="Path to previous checkpoint.")
    args = parser.parse_args()

    # ===== Set up some arguments =====
    import uuid

    # control_device = args.device
    # control_device = args.device
    experiment_batch_name = "{}".format(args.exp_name)
    seed = args.seed
    trial_name = "{}_{}_{}".format(experiment_batch_name, get_time_str(), uuid.uuid4().hex[:8])
    print("Trial name is set to: ", trial_name)

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    experiment_dir = Path("runs") / experiment_batch_name
    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=True)
    print(f"We start logging training data into {trial_dir}")

    # ===== Setup the config =====
    config = dict(
        # Environment config
        # env_config={"main_exp": False, "horizon": 1500},

        # Algorithm config
        algo=dict(
            policy=TD3Policy,
            replay_buffer_class=HACOReplayBuffer,  ###
            replay_buffer_kwargs=dict(),
            policy_kwargs=dict(net_arch=[256, 256]),
            env=None,
            learning_rate=1e-4,
            optimize_memory_usage=True,
            learning_starts=10000,
            batch_size=1024,
            tau=0.005,
            gamma=0.99,
            train_freq=5,
            gradient_steps=1,
            action_noise=None,
            # action_noise=NormalActionNoise(mean=np.zeros([2,]), sigma=0.15 * np.ones([2,])),
            # target_policy_noise=0,
            policy_delay=2,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
            buffer_size=1_000_000,
        ),

        # Experiment log
        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir)
    )

    # ===== Setup the training environment =====


    # ===== Also build the eval env =====
    # ===== Also build the eval env =====
    def _make_eval_env():
        eval_env_config = dict(
            use_render=False,  # Open the interface
            manual_control=False,  # Allow receiving control signal from external device
            start_seed=1000,
            horizon=1500,
        )
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        from pvp.sb3.common.monitor import Monitor
        eval_env = HumanInTheLoopEnv(config=eval_env_config)
        eval_env = Monitor(env=eval_env, filename=str(trial_dir))
        return eval_env


    eval_env = SubprocVecEnv([_make_eval_env] * 20)
    
    def make_train_env():
        env_config = dict(
            use_render=False,  # Open the interface
            manual_control=False,  # Allow receiving control signal from external device
            # controller=control_device,
            window_size=(1600, 1100),
        )

        train_env = HumanInTheLoopEnv(config=env_config)
        train_env = Monitor(env=train_env, filename=str(trial_dir))
        return train_env

    train_env = SubprocVecEnv([make_train_env])
    train_env.num_envs = 1
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Setup the callbacks =====
    callbacks = [
        CheckpointCallback(name_prefix="rl_model", verbose=1, save_freq=10000, save_path=str(trial_dir / "models"))
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=experiment_batch_name,
                team_name=team_name,
                project_name=project_name,
                config=config
            )
        )

    callbacks = CallbackList(callbacks)

    # if args.eval:
    #     # eval_env = SubprocVecEnv([])
    #     # eval_env = SubprocVecEnv([lambda: _make_eval_env(False)])
    #     config["algo"]["learning_rate"] = 0.0
    #     config["algo"]["train_freq"] = (1, "step")

    # ===== Setup the training algorithm =====
    if args.ckpt:
        model = TD3.load(args.ckpt, **config["algo"])
    else:
        model = TD3(**config["algo"])

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=1000_0000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=eval_env,
        # eval_freq=5000,
        eval_freq=10000,
        n_eval_episodes=200,
        eval_log_path=str(trial_dir),

        # logging
        tb_log_name=experiment_batch_name,  # Should place the algorithm name here!
        log_interval=1,
    )
