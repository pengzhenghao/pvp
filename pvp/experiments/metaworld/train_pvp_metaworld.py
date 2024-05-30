"""
Training script for training PVP in MetaDrive Safety Env.
"""
import argparse
import os
from pathlib import Path

from pvp.experiments.metaworld.metaworld_env import FakeHumanInTheLoopMetaWorld
from pvp.pvp_td3 import PVPTD3
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.haco import HACOReplayBuffer
from pvp.sb3.td3.policies import TD3Policy
from pvp.utils.shared_control_monitor import SharedControlMonitor
from pvp.utils.utils import get_time_str

os.environ["WANDB_START_METHOD"] = "thread" # for wandb to upload properly

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="pvp_metaworld", type=str, help="The name for this batch of experiments.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--log_dir", type=str, default="data/michaelsong/pvp", help="Folder to store the logs.")
    parser.add_argument("--wandb_project", type=str, default="", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="", help="The team name for wandb.")
    args = parser.parse_args()

    # ===== Set up some arguments =====
    experiment_batch_name = "{}".format(args.exp_name)
    seed = args.seed
    trial_name = "{}_{}".format(experiment_batch_name, get_time_str())

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    log_dir = args.log_dir
    experiment_dir = Path(log_dir) / Path("runs") / experiment_batch_name

    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=False)
    print(f"We start logging training data into {trial_dir}")

    # ===== Setup the config =====
    config = dict(

        # todo: setup an environment config for metaworld envs

        # Algorithm config
        algo=dict(
            use_balance_sample=True,
            policy=TD3Policy,
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(
                discard_reward=True,  # We run in reward-free manner!
            ),
            policy_kwargs=dict(net_arch=[256, 256]),
            env=None,
            learning_rate=1e-4,
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=50_000,  # We only conduct experiment less than 50K steps
            learning_starts=100,  # The number of steps before
            batch_size=128,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            action_noise=None,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
        ),

        # Experiment log
        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir)
    )

    # ===== Setup the training environment =====
    train_env = FakeHumanInTheLoopMetaWorld(env_name='button-press-v2')
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    # Store all shared control data to the files.
    train_env = SharedControlMonitor(env=train_env, folder=trial_dir / "data", prefix=trial_name)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None


    # ===== Setup the eval environment =====
    def _make_eval_env():
        from pvp.experiments.metaworld.metaworld_env import HumanInTheLoopEnv
        from pvp.sb3.common.monitor import Monitor
        eval_env = HumanInTheLoopEnv(env_name='button-press-v2')
        eval_env = Monitor(env=eval_env, filename=str(trial_dir))
        return eval_env

    eval_env = _make_eval_env()

    # ===== Setup the callbacks =====
    save_freq = 500  # Number of steps per model checkpoint
    callbacks = [
        CheckpointCallback(name_prefix="rl_model", verbose=1, save_freq=save_freq, save_path=str(trial_dir / "models"))
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

    # ===== Setup the training algorithm =====
    model = PVPTD3(**config["algo"])

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=50_000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=eval_env,
        eval_freq=1000,
        n_eval_episodes=10,
        eval_log_path=str(trial_dir),

        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
        save_buffer=False,
        load_buffer=False,
    )
