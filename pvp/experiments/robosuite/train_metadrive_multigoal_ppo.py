"""
Training script for training PPO in MetaDrive Safety Env.
"""
import argparse
import os
from pathlib import Path

from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.env_util import make_vec_env
from pvp.sb3.common.vec_env import DummyVecEnv, SubprocVecEnv
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.ppo import PPO
from pvp.sb3.ppo.policies import ActorCriticPolicy
from pvp.utils.utils import get_time_str

import argparse
import os
import os.path as osp

from metadrive.envs.multigoal_intersection import MultiGoalIntersectionEnv
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

from metadrive.envs.multigoal_intersection import MultiGoalIntersectionEnv
from metadrive.envs.gym_wrapper import create_gym_wrapper
import numpy as np


def register_env(make_env_fn, env_name):
    from gym.envs.registration import register
    register(id=env_name, entry_point=make_env_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", default="ppo_metadrive_multigoal", type=str, help="The name for this batch of experiments."
    )
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--lane_line_detector", default=0, type=int)
    parser.add_argument("--vehicle_detector", default=120, type=int)
    parser.add_argument("--traffic_density", default=0.2, type=float)
    parser.add_argument("--ckpt", default=None, type=str, help="Path to previous checkpoint.")
    parser.add_argument("--debug", action="store_true", help="Set to True when debugging.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="", help="The team name for wandb.")
    args = parser.parse_args()

    # FIXME: Remove this in future.
    # if args.wandb_team is None:
    #     args.wandb_team = "drivingforce"
    # if args.wandb_project is None:
    #     args.wandb_project = "pvp2024"

    project_name = "brandon"
    team_name = "drivingforce"

    # ===== Set up some arguments =====
    # control_device = args.device
    experiment_batch_name = "{}".format(args.exp_name)
    seed = args.seed
    trial_name = "{}_seed{}_{}".format(experiment_batch_name, seed, get_time_str())

    traffic_density = args.traffic_density

    use_wandb = args.wandb
    # project_name = args.wandb_project
    # team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    experiment_dir = Path("runs") / experiment_batch_name
    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=True)
    print(f"We start logging training data into {trial_dir}")

    # ===== Setup the config =====
    config = dict(
        # ===== Environment =====
        # env_config=dict(
        #     use_render=False,  # Open the interface
        #     manual_control=False,  # Allow receiving control signal from external device
        #     # controller=control_device,
        #     # window_size=(1600, 1100),
        #     horizon=1500,
        #
        #     use_multigoal_intersection=False,
        #     num_scenarios=1000,
        #     start_seed=1000,
        #
        # ),
        num_train_envs=32,

        # ===== Environment =====
        # eval_env_config=dict(
        #     use_render=False,  # Open the interface
        #     manual_control=False,  # Allow receiving control signal from external device
        #     start_seed=1000,
        #     horizon=1500,
        # ),
        num_eval_envs=1,

        # ===== Training =====
        algo=dict(
            policy=ActorCriticPolicy,
            n_steps=1024,  # n_steps * n_envs = total_batch_size
            n_epochs=20,
            learning_rate=5e-5,
            batch_size=256,
            clip_range=0.1,
            vf_coef=0.5,
            ent_coef=0.0,
            max_grad_norm=10.0,
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
    vec_env_cls = SubprocVecEnv
    if args.debug:
        config["num_train_envs"] = 1
        config["algo"]["n_steps"] = 64
        vec_env_cls = DummyVecEnv

    # ===== Setup the training environment =====
    # train_env_config = config["env_config"]

    # def _make_train_env():
    #     from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    #     from pvp.sb3.common.monitor import Monitor
    #     train_env = HumanInTheLoopEnv(config=train_env_config)
    #     train_env = Monitor(env=train_env, filename=str(trial_dir))
    #     return train_env

    def make_train_env():
        class MultiGoalWrapped(MultiGoalIntersectionEnv):
            current_goal = None

            def step(self, actions):
                o, r, tm, tc, i = super().step(actions)

                o = i['obs/goals/{}'.format(self.current_goal)]
                r = i['reward/goals/{}'.format(self.current_goal)]
                i['route_completion'] = i['route_completion/goals/{}'.format(self.current_goal)]
                i['arrive_dest'] = i['arrive_dest/goals/{}'.format(self.current_goal)]
                i['reward/goals/default'] = i['reward/goals/{}'.format(self.current_goal)]
                i['route_completion/goals/default'] = i['route_completion/goals/{}'.format(self.current_goal)]
                i['arrive_dest/goals/default'] = i['arrive_dest/goals/{}'.format(self.current_goal)]

                return o, r, tm, tc, i

            def reset(self, *args, **kwargs):
                o, i = super().reset(*args, **kwargs)

                # Sample a goal from the goal set
                if self.config["use_multigoal_intersection"]:
                    p = {"right_turn": 0.3, "left_turn": 0.3, "go_straight": 0.1, "u_turn": 0.3}
                    self.current_goal = np.random.choice(list(p.keys()), p=list(p.values()))

                else:
                    self.current_goal = "default"

                o = i['obs/goals/{}'.format(self.current_goal)]
                i['route_completion'] = i['route_completion/goals/{}'.format(self.current_goal)]
                i['arrive_dest'] = i['arrive_dest/goals/{}'.format(self.current_goal)]
                i['reward/goals/default'] = i['reward/goals/{}'.format(self.current_goal)]
                i['route_completion/goals/default'] = i['route_completion/goals/{}'.format(self.current_goal)]
                i['arrive_dest/goals/default'] = i['arrive_dest/goals/{}'.format(self.current_goal)]

                # print("CURRENT RANDOM SEED: ", self.current_seed)

                return o, i

        env_config = dict(
            use_render=False,
            manual_control=False,
            vehicle_config=dict(
                show_navi_mark=True,
                show_line_to_navi_mark=True,
                show_lidar=False,
                show_side_detector=True,
                show_lane_line_detector=True,
                lane_line_detector=dict(num_lasers=args.lane_line_detector),
                lidar=dict(num_lasers=args.vehicle_detector),
            ),
            # accident_prob=0.0,
            traffic_density=traffic_density,
            # decision_repeat=5,
            # horizon=500,  # to speed up training
            #
            # out_of_route_penalty=1,
            use_multigoal_intersection=False,
            num_scenarios=1000,
            start_seed=1000,
        )

        env_config.update(
            {
                "use_render": bool(args.ckpt),
                "num_scenarios": 100,
                "accident_prob": 0.8,
                # "traffic_density": 0.05,
                "crash_vehicle_done": False,
                "crash_object_done": False,
                # "cost_to_reward": False,
                "out_of_route_done": True,  # Raise done if out of route.
                "num_scenarios": 50,  # There are totally 50 possible maps.
                "start_seed": 100,  # We will use the map 100~150 as the default training environment.
                # "traffic_density": 0.06,
            }
        )

        return create_gym_wrapper(MultiGoalWrapped)(env_config)

    # def make_eval_env():
    #
    #     env_config = dict(
    #         use_render=False,
    #         manual_control=False,
    #         vehicle_config=dict(
    #             show_navi_mark = True,
    #             show_line_to_navi_mark = True,
    #             show_lidar = False,
    #             show_side_detector = True,
    #             show_lane_line_detector = True,
    #         ),
    #         accident_prob=0.0,
    #         traffic_density=0.0,
    #         decision_repeat=5,
    #         horizon=500,  # to speed up training
    #
    #         out_of_route_penalty=1,
    #     )
    #
    #     return create_gym_wrapper(MultiGoalIntersectionEnv)(env_config)

    # train_env_name = "metadrive_train-v0"
    # register_env(_make_train_env, train_env_name)
    train_env = make_vec_env(make_train_env, n_envs=config["num_train_envs"], vec_env_cls=vec_env_cls)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Also build the eval env =====
    # eval_env_config = config["eval_env_config"]

    # def _make_eval_env():
    #     from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    #     from pvp.sb3.common.monitor import Monitor
    #     eval_env = HumanInTheLoopEnv(config=eval_env_config)
    #     eval_env = Monitor(env=eval_env, filename=str(trial_dir))
    #     return eval_env

    # eval_env_name = "metadrive_eval-v0"
    # register_env(_make_eval_env, eval_env_name)
    # eval_env = make_vec_env(make_eval_env, n_envs=config["num_eval_envs"], vec_env_cls=vec_env_cls)
    eval_env = None

    # ===== Setup the callbacks =====
    save_freq = 100_000  # Number of steps per model checkpoint
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

    if args.ckpt:

        config["algo"]["learning_rate"] = 0.0
        model = PPO(**config["algo"])

        ckpt = Path(args.ckpt)
        print(f"Loading checkpoint from {ckpt}!")
        from pvp.sb3.common.save_util import load_from_zip_file
        data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)

        model.set_parameters(params, exact_match=True, device=model.device)

    else:
        model = PPO(**config["algo"])

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=10_000_000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=eval_env,
        eval_freq=100_000,
        n_eval_episodes=100,
        eval_log_path=str(trial_dir),

        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
        # save_buffer=False,
        # load_buffer=False,
    )
