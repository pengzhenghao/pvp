import argparse
import os
import os.path as osp

# from pvp.train_metadrive.human_in_the_loop_env import HumanInTheLoopEnv
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.td3.td3 import TD3, ReplayBuffer
from pvp.sb3.td3.policies import TD3Policy
from pvp.sb3.common.vec_env.subproc_vec_env import SubprocVecEnv
from pvp.sb3.common.vec_env.dummy_vec_env import DummyVecEnv
# from drivingforce.human_in_the_loop.common import baseline_eval_config

from pvp.utils.utils import get_time_str



def make_train_env():
    from metadrive.envs.multigoal_intersection import MultiGoalIntersectionEnv
    from metadrive.envs.gym_wrapper import create_gym_wrapper

    env_config = dict(
        use_render=False,
        manual_control=False,
        vehicle_config=dict(show_lidar=False, show_navi_mark=True, show_line_to_navi_mark=True),
        accident_prob=0.0,
        decision_repeat=5,
        horizon=500,  # to speed up training
    )

    return create_gym_wrapper(MultiGoalIntersectionEnv)(env_config)


def make_eval_env():
    from metadrive.envs.multigoal_intersection import MultiGoalIntersectionEnv
    from metadrive.envs.gym_wrapper import create_gym_wrapper

    env_config = dict(
        use_render=False,
        manual_control=False,
        vehicle_config=dict(show_lidar=False, show_navi_mark=True, show_line_to_navi_mark=True),
        accident_prob=0.0,
        decision_repeat=5,
        horizon=500,  # to speed up training
    )

    return create_gym_wrapper(MultiGoalIntersectionEnv)(env_config)


# def make_eval_env(log_dir):
#     def _init():
#         env = Monitor(env=HumanInTheLoopEnv(config=baseline_eval_config), filename=os.path.join(log_dir, "eval"))
#         return env
#
#     return _init


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="TEST", type=str, help="The experiment name.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    args = parser.parse_args()

    # ===== Setup some meta information =====
    exp_name = args.exp_name
    seed = int(args.seed)
    use_wandb = args.wandb
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    project_name = "brandon"
    team_name = "drivingforce"

    trial_name = "{}_seed{}_{}".format(exp_name, seed, get_time_str())
    log_dir = osp.join("runs", exp_name, trial_name)
    os.makedirs(osp.join("runs", exp_name), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print("We start logging training data into {}".format(log_dir))

    # ===== Setup the config =====
    config = dict(
        # Environment config
        # env_config={"main_exp": False, "horizon": 1500},

        # Algorithm config
        algo=dict(
            policy=TD3Policy,
            replay_buffer_class=ReplayBuffer,  ###
            replay_buffer_kwargs=dict(),
            policy_kwargs=dict(net_arch=[256, 256]),
            env=None,
            learning_rate=1e-4,
            optimize_memory_usage=True,

            learning_starts=10000,  ###
            batch_size=100,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            # train_freq=1,
            # target_policy_noise=0,
            # policy_delay=1,

            action_noise=None,
            tensorboard_log=log_dir,
            create_eval_env=False,

            verbose=2,
            seed=seed,
            device="auto",
        ),

        # Meta data
        project_name=project_name,
        team_name=team_name,
        exp_name=exp_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=log_dir
    )

    # ===== Setup the training environment =====
    train_env = make_train_env()
    train_env = Monitor(env=train_env, filename=log_dir)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # eval_env = DummyVecEnv([make_eval_env])
    eval_env = None

    # ===== Setup the callbacks =====
    callbacks = [
        CheckpointCallback(
            name_prefix="rl_model",
            verbose=1,
            save_freq=10000,
            save_path=osp.join(log_dir, "models")
        )
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=exp_name,
                project_name=project_name,
                config=config
            )
        )
    callbacks = CallbackList(callbacks)

    # ===== Setup the training algorithm =====
    model = TD3(**config["algo"])

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=200_0000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=eval_env,
        # eval_freq=5000,
        eval_freq=1,
        n_eval_episodes=30,
        eval_log_path=log_dir,

        # logging
        tb_log_name=exp_name,  # Should place the algorithm name here!
        log_interval=4,
    )
