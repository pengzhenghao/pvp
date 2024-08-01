import argparse
import os
import os.path as osp
import numpy as np
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.sac import SAC
from pvp.sb3.sac.sac import ReplayBuffer
from pvp.sb3.sac.policies import SACPolicy
from pvp.sb3.common.vec_env.subproc_vec_env import SubprocVecEnv
from pvp.sb3.common.vec_env.dummy_vec_env import DummyVecEnv

from pvp.utils.utils import get_time_str

from metadrive.envs.multigoal_intersection import MultiGoalIntersectionEnv
from metadrive.envs.gym_wrapper import create_gym_wrapper
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


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
            p = {
                "right_turn": 0.3,
                "left_turn": 0.3,
                "go_straight": 0.1,
                "u_turn": 0.3
            }
            self.current_goal = np.random.choice(list(p.keys()), p=list(p.values()))

        else:
            self.current_goal = "default"

        o = i['obs/goals/{}'.format(self.current_goal)]
        i['route_completion'] = i['route_completion/goals/{}'.format(self.current_goal)]
        i['arrive_dest'] = i['arrive_dest/goals/{}'.format(self.current_goal)]
        i['reward/goals/default'] = i['reward/goals/{}'.format(self.current_goal)]
        i['route_completion/goals/default'] = i['route_completion/goals/{}'.format(self.current_goal)]
        i['arrive_dest/goals/default'] = i['arrive_dest/goals/{}'.format(self.current_goal)]

        return o, i


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="TEST", type=str, help="The experiment name.")
    parser.add_argument("--ckpt", default="", type=str)
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    # parser.add_argument("--penalty", default=2.0, type=float)
    # parser.add_argument("--driving_reward", default=1.0, type=float)
    args = parser.parse_args()

    # ===== Setup some meta information =====
    exp_name = args.exp_name
    seed = int(args.seed)
    use_wandb = args.wandb
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    project_name = "brandon"
    team_name = "drivingforce"

    experiment_batch_name = exp_name
    trial_name = "{}_seed{}_{}".format(exp_name, seed, get_time_str())
    log_dir = osp.join("runs", exp_name, trial_name)
    os.makedirs(osp.join("runs", exp_name), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print("We start logging training data into {}".format(log_dir))

    # ===== Setup the config =====
    config = dict(
        # Algorithm config
        algo=dict(
            policy=SACPolicy,
            replay_buffer_class=ReplayBuffer,  ###
            # replay_buffer_kwargs=dict(),
            policy_kwargs=dict(log_std_init=1e-3, net_arch=[400, 300]),
            env=None,

            # ===== Training =====
            learning_rate=dict(actor=1e-4, critic=1e-4, entropy=1e-4),

            use_sde_at_warmup=True,
            use_sde=True,
            sde_sample_freq=64,

            learning_starts=10000 if not args.eval else 0,  ###
            batch_size=256,

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

    def make_train_env(render=False):

        env_config = dict(
            use_render=render,
            manual_control=False,
            vehicle_config=dict(show_lidar=False, show_navi_mark=True, show_line_to_navi_mark=True,
                                show_line_to_dest=True, show_dest_mark=True),
            decision_repeat=5,
            horizon=500,  # to speed up training

            traffic_density=0.06,

            use_multigoal_intersection=True,
            out_of_route_done=False,  # Raise done if out of route.

            num_scenarios=1000,
            start_seed=100,
            accident_prob=0.8,
            crash_vehicle_done=False,
            crash_object_done=False,
        )

        wrapped = create_gym_wrapper(MultiGoalWrapped)

        return wrapped(env_config)


    train_env = make_train_env(render=args.eval)
    train_env = Monitor(env=train_env, filename=log_dir)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    eval_env = None

    # ===== Setup the callbacks =====
    callbacks = [
        CheckpointCallback(
            name_prefix="rl_model",
            verbose=1,
            save_freq=1_0000,
            save_path=osp.join(log_dir, "models")
        )
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

    if args.eval:
        # eval_env = SubprocVecEnv([])
        # eval_env = SubprocVecEnv([lambda: _make_eval_env(False)])
        config["algo"]["learning_rate"] = 0.0
        config["algo"]["train_freq"] = (1, "step")

    # ===== Setup the training algorithm =====
    if args.ckpt:

        model = SAC(**config["algo"])

        ckpt = str(PROJECT_ROOT / args.ckpt)

        from pvp.sb3.common.save_util import load_from_zip_file

        data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)

        model.set_parameters(params, exact_match=True, device=model.device)

    else:
        model = SAC(**config["algo"])

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=300_0000,
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
