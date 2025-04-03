"""
Compared to original file:
1. use fakehumanenv
2. new config: free_level
3. buffer_size and total_timesteps set to 150_000
"""
import argparse
import os
from pathlib import Path
import uuid
import torch
import torch.nn as nn
from pvp.sb3.common.policies import BaseFeaturesExtractor

from pvp.experiments.metadrive.egpo.fakehuman_env import FakeHumanEnv
from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
from pvp.pvp_td3 import PVPTD3
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.haco import HACOReplayBuffer
from pvp.sb3.td3.policies import TD3Policy
from pvp.utils.shared_control_monitor import SharedControlMonitor
from pvp.utils.utils import get_time_str
from pvp.sb3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv

import math
from typing import Optional

import numpy as np
import torch
import torch as th
import torch.nn as nn

from pvp.sb3.common.policies import ContinuousCritic
from pvp.sb3.common.torch_layers import create_mlp
from pvp.sb3.sac.our_features_extractor import BaseFeaturesExtractor
from pvp.sb3.td3.policies import TD3Policy


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.embedding = nn.Embedding.from_pretrained(
            pe, freeze=False
        )  # Initialize embedding with positional encodings

    def forward(self, x, steps):
        pe = self.embedding(steps)  # Query positional encodings using steps
        return x + pe


class TinyTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, max_len=100):
        super(TinyTransformer, self).__init__()
        self.positional_encoding = PositionalEncoding(input_dim, max_len)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=0.0
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.class_token = nn.Parameter(torch.randn(1, 1, input_dim))

    def forward(self, x, steps):
        x = self.positional_encoding(x, steps)

        batch_size = x.size(0)
        class_token = self.class_token.expand(batch_size, -1, -1)  # Expand class token to batch size

        x = torch.cat((class_token, x), dim=1)  # Prepend class token
        x = x.permute(1, 0, 2)  # Transformer expects [sequence_length, batch_size, feature_dim]
        x = self.transformer_encoder(x)
        return x[0, :, :]  # Return the class token output


class LowStateNets(nn.Module):
    def __init__(self, history_len=50, input_dim=46, features_dim=256):
        super(LowStateNets, self).__init__()

        nn.Linear(input_dim, 128)

        # Define the convolutional layers to process the sequence data
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.fc_last_frame = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())

        # Define the final fully connected layers to merge the features
        self.fc_final = nn.Sequential(nn.Linear(128 + 128, 512), nn.ReLU(), nn.Linear(512, 256))

    def forward(self, low_level_states, last_frame_data):
        # low_level_states shape: (B, 50, 46)
        # last_frame_data shape: (B, 46)

        # Process the low level states with convolutional layers
        x = low_level_states.permute(0, 2, 1)  # Change shape to (B, 46, 50) for Conv1d
        x = self.conv1(x)  # Output shape: (B, 128, 50)
        x = nn.ReLU()(x)
        x = self.conv2(x)  # Output shape: (B, 256, 50)
        x = nn.ReLU()(x)

        # Apply global average pooling to get a fixed-size feature vector
        x = nn.AdaptiveAvgPool1d(1)(x)  # Output shape: (B, 256, 1)
        x = x.squeeze(-1)  # Output shape: (B, 256)

        # Process the last frame data with fully connected layers
        y = self.fc_last_frame(last_frame_data)  # Output shape: (B, 256)

        # Concatenate the processed sequence data and the last frame data
        combined = torch.cat((x, y), dim=1)  # Output shape: (B, 512)

        # Pass the combined features through the final fully connected layers
        features = self.fc_final(combined)  # Output shape: (B, 256)

        return features


class LowStateNetwork(nn.Module):
    """
    LowStateNetwork class for processing low-level states.
    """
    def __init__(
        self, input_channels: int = 46, embedding_channels: int = 32, history_length: int = 50, output_size: int = 8
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embedding_channels = embedding_channels
        self.history_length = history_length
        self.output_size = output_size

        self.embedding = nn.Sequential(nn.Linear(self.input_channels, self.embedding_channels), nn.ReLU(inplace=True))

        size_proj = self.history_length
        conv_layers = []
        for out_channels, kernel_size, stride in [
            (self.embedding_channels, 4, 3),
            (self.embedding_channels, 4, 2),
            (self.embedding_channels, 3, 1),
            (self.embedding_channels, 3, 1),
        ]:
            size_proj = int(np.floor((size_proj - kernel_size) / stride + 1))
            conv_layers.append(nn.Conv1d(self.embedding_channels, out_channels, kernel_size=kernel_size, stride=stride))
            conv_layers.append(nn.ReLU(inplace=True))

        self.conv = conv_layers
        self.conv = nn.Sequential(*self.conv)
        self.projection = nn.Sequential(
            nn.Linear(size_proj * self.embedding_channels + input_channels, self.output_size), nn.ReLU(inplace=True),
            nn.Linear(self.output_size, self.output_size)
        )

    def forward(self, x) -> torch.Tensor:
        # x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        B, T = x.shape[:2]

        assert x.ndim == 4
        # Collapse the batch and time dimensions first
        x = x.view(-1, self.history_length, self.input_channels)

        embedded = self.embedding(x)

        # conv_out = embedded.transpose(1, 2)
        # for c in self.conv:
        #     conv_out = c(conv_out)
        conv_out = self.conv(embedded.transpose(1, 2))
        output = self.projection(torch.cat([conv_out.flatten(1), x[:, -1]], dim=1))
        output = output.view(B, T, -1)
        return output


class BBoxFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        features_dim,
        transformer_hidden_dim=1024,
        horizon=10,
        num_transformer_layers=2,
        num_transformer_heads=2,
        net_arch=(256, ),
        use_continuous_action_space=False
    ):
        super().__init__(observation_space, features_dim)

        obs_flat_dim = observation_space["obs"].shape[1:]
        obs_flat_dim = np.prod(obs_flat_dim)
        self.nets = nn.Sequential(*create_mlp(input_dim=obs_flat_dim, output_dim=features_dim, net_arch=net_arch))

        if use_continuous_action_space:
            act_flat_dim = observation_space["action"].shape[1:]
            act_flat_dim = np.prod(act_flat_dim)
            self.action_embedding = nn.Sequential(
                *create_mlp(input_dim=act_flat_dim, output_dim=features_dim, net_arch=net_arch)
            )
        else:
            self.action_embedding = nn.Embedding(observation_space["action"][0].n, features_dim)

        # Define the tiny transformer
        self.transformer = TinyTransformer(
            input_dim=features_dim,
            num_heads=num_transformer_heads,
            num_layers=num_transformer_layers,
            hidden_dim=transformer_hidden_dim,
            max_len=horizon
        )

    def forward(self, observations):
        obs_features = self.nets(observations["obs"])
        obs_step = torch.arange(observations['obs'].shape[1]).to(observations['obs'].device)
        obs_step = obs_step.unsqueeze(0).expand(observations['obs'].shape[0], -1)
        # Offset 1 step for observation
        obs_step = obs_step + 1

        action_features = self.action_embedding(observations["action"])
        act_step = torch.arange(observations['action'].shape[1]).to(observations['action'].device)
        act_step = act_step.unsqueeze(0).expand(observations['action'].shape[0], -1)

        features = [obs_features, action_features]
        steps = [obs_step, act_step]

        time_dim = 1
        features = torch.cat(features, dim=time_dim)
        steps = torch.cat(steps, dim=time_dim)

        output_token = self.transformer(features, steps)  # Apply the transformer
        return output_token


class GRUFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        features_dim,
        gru_hidden_dim=128,
        horizon=10,
        net_arch=(256, ),
        use_continuous_action_space=False,
        transformer_hidden_dim=None,
        num_transformer_layers=None,
        num_transformer_heads=None
    ):
        super().__init__(observation_space, features_dim)

        obs_flat_dim = observation_space["obs"].shape[1:]
        obs_flat_dim = np.prod(obs_flat_dim)
        obs_flat_dim_with_action = obs_flat_dim + 2

        self.input_fc = nn.Sequential(
            *create_mlp(input_dim=obs_flat_dim_with_action, output_dim=features_dim, net_arch=[
                256,
                256,
            ])
        )
        self.gru = nn.GRU(input_size=256, hidden_size=features_dim, num_layers=2, batch_first=True)
        # self.output_fc = nn.Linear(gru_hidden_dim, features_dim)
        self.forward_fc = nn.Sequential(
            *create_mlp(input_dim=obs_flat_dim_with_action, output_dim=features_dim, net_arch=[
                256,
                256,
            ])
        )

    def forward(self, observations):

        # if observations['action'].shape[1] != observations['obs'].shape[1]:
        #     act = observations["action"][:, 1:]
        #     obs = torch.cat([act, observations["obs"]], dim=-1)
        # else:
        #     act = observations["action"][:, 1:]
        #     act = torch.cat([act, torch.zeros_like(act[:, :1])], dim=1)
        #     obs = torch.cat([act, observations["obs"]], dim=-1)
        #
        # obs_features = self.input_fc(obs)
        # gru_out, _ = self.gru(obs_features)
        # output = gru_out[:, -1, :]

        # Residual:
        if observations['action'].shape[1] != observations['obs'].shape[1]:
            forward_obs = torch.cat([observations["obs"][:, -1], observations["action"][:, -1]], dim=-1)
        else:
            act = observations["action"][:, 1:]
            forward_obs = torch.cat([observations["obs"][:, -1], torch.zeros_like(act[:, -1])], dim=-1)
        forward_output = self.forward_fc(forward_obs)
        # output = output + forward_output
        output = forward_output

        return output


class PVPCritic(ContinuousCritic):
    def __init__(self, *args, **kwargs):
        super(PVPCritic, self).__init__(*args, **kwargs)
        assert not self.share_features_extractor

    def add_actions_to_obs(self, obs, actions):
        new_obs = {k: v for k, v in obs.items()}
        new_obs['action'] = th.cat([obs['action'], actions.unsqueeze(dim=1)], dim=1)
        return new_obs

    def forward(self, obs: th.Tensor, actions: th.Tensor):
        new = self.add_actions_to_obs(obs, actions)
        features = self.extract_features(new)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        # TODO: Why we have no gradient here?
        # with th.no_grad():
        features = self.extract_features(self.add_actions_to_obs(obs, actions))
        return self.q_networks[0](th.cat([features, actions], dim=1))

    def q2_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        raise ValueError()
        with th.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[1](th.cat([features, actions], dim=1))


class PVPContinuousPolicy(TD3Policy):
    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None):
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return PVPCritic(**critic_kwargs).to(self.device)


"""
From: https://github.com/octo-models/octo/blob/5eaa5c6960398925ae6f52ed072d843d2f3ecb0b/octo/utils/gym_wrappers.py#L27
"""

import gym
import gym.spaces
import gymnasium
import numpy as np
from collections import deque


def stack_and_pad(history: deque, num_obs: int):
    """
    Converts a list of observation dictionaries (`history`) into a single observation dictionary
    by stacking the values. Adds a padding mask to the observation that denotes which timesteps
    represent padding based on the number of observations seen so far (`num_obs`).
    """
    horizon = len(history)
    full_obs = {k: np.stack([dic[k] for dic in history]) for k in history[0]}
    pad_length = horizon - min(num_obs, horizon)
    timestep_pad_mask = np.ones(horizon)
    timestep_pad_mask[:pad_length] = 0
    full_obs["timestep_pad_mask"] = timestep_pad_mask
    return full_obs


def space_stack(space: gym.Space, repeat: int):
    """
    Creates new Gym space that represents the original observation/action space
    repeated `repeat` times.
    """

    if isinstance(space, (gymnasium.spaces.Box, gym.spaces.Box)):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    elif isinstance(space, (gymnasium.spaces.Discrete, gym.spaces.Discrete)):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, (gymnasium.spaces.Dict, gym.spaces.Dict)):
        return gym.spaces.Dict({k: space_stack(v, repeat) for k, v in space.spaces.items()})
    else:
        raise ValueError(f"Space {space} is not supported by Gym wrappers.")


class HistoryWrapper(gym.Wrapper):
    """
    Accumulates the observation history into `horizon` size chunks. If the length of the history
    is less than the length of the horizon, we pad the history to the full horizon length.
    A `timestep_pad_mask` key is added to the final observation dictionary that denotes which timesteps
    are padding.
    """
    def __init__(self, env: gym.Env, horizon: int, include_first_frame=False):
        super().__init__(env)
        self.horizon = horizon

        self.history = deque(maxlen=self.horizon)
        self.action_history = deque(maxlen=self.horizon)
        self.num_obs = 0
        self.first_obs = None

        self.include_first_frame = include_first_frame
        self.should_capture_first_frame = False

        obs_horizon = self.horizon + 1 if include_first_frame else self.horizon

        obs_dict = {
            "obs": space_stack(self.env.observation_space, obs_horizon),
            "action": space_stack(self.env.action_space, obs_horizon),
            "timestep_pad_mask": gym.spaces.Box(shape=(obs_horizon, ), low=0, high=1, dtype=np.float32),
        }
        # if self.unwrapped.config.use_low_state_obs:
        #     # TODO: The shape is hardcoded here.
        #     obs_dict["low_state"] = space_stack(
        #         gymnasium.spaces.Box(shape=(50, 46,), low=float("-inf"), high=float("inf")), obs_horizon)

        self.observation_space = gym.spaces.Dict(obs_dict)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.num_obs += 1
        self.history.append(self.get_dict(obs, info))
        assert len(self.history) == self.horizon

        if self.should_capture_first_frame:
            # Capture the first frame.
            self.should_capture_first_frame = False

            a = self.get_default_action()
            self.first_obs = self.get_dict(obs, a)

        if self.include_first_frame:
            full_obs = stack_and_pad([self.first_obs] + list(self.history), self.num_obs + 1)
        else:
            full_obs = stack_and_pad(list(self.history), self.num_obs)

        return full_obs, reward, done, info

    def get_dict(self, obs, info):
        action = info["raw_action"]
        ret = {"obs": obs, "action": action}

        # if self.unwrapped.config.use_low_state_obs:
        #     ret["low_state"] = info["low_state"]

        return ret

    # def reverse_rgb_dict(self, obs):
    #     return obs['rgb']

    def get_default_info(self, o):
        if isinstance(self.env.action_space, (gym.spaces.Discrete, gymnasium.spaces.Discrete)):
            a = 0
        elif isinstance(self.env.action_space, (gym.spaces.Box, gymnasium.spaces.Box)):
            a = np.zeros(self.env.action_space.shape)
        else:
            raise NotImplementedError
        info = {"raw_action": a}
        # TODO: Maybe we can ask the env to return this?
        # if self.unwrapped.config.use_low_state_obs:
        #     info["low_state"] = np.zeros(46)
        return info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        info = self.get_default_info(obs)

        self.num_obs = 1
        self.history.extend([self.get_dict(obs, info)] * self.horizon)

        if self.include_first_frame:
            self.should_capture_first_frame = True
            full_obs = stack_and_pad([self.history[0]] + list(self.history), self.num_obs)

        else:
            full_obs = stack_and_pad(list(self.history), self.num_obs)

        self.first_obs = None

        return full_obs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", default="pvp_metadrive_fakehuman", type=str, help="The name for this batch of experiments."
    )
    parser.add_argument("--learning_starts", default=200, type=int)
    parser.add_argument("--save_freq", default=500, type=int)
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="", help="The team name for wandb.")
    parser.add_argument("--log_dir", type=str, default="/data/zhenghao/pvp", help="Folder to store the logs.")
    parser.add_argument("--free_level", type=float, default=0.95)
    parser.add_argument("--bc_loss_weight", type=float, default=0.0)

    # parser.add_argument(
    #     "--intervention_start_stop_td", default=True, type=bool, help="Whether to use intervention_start_stop_td."
    # )

    parser.add_argument("--adaptive_batch_size", default="False", type=str)
    parser.add_argument("--ckpt", default="", type=str)

    parser.add_argument("--toy_env", action="store_true", help="Whether to use a toy environment.")
    # parser.add_argument(
    #     "--device",
    #     required=True,
    #     choices=['wheel', 'gamepad', 'keyboard'],
    #     type=str,
    #     help="The control device, selected from [wheel, gamepad, keyboard]."
    # )
    args = parser.parse_args()

    # ===== Set up some arguments =====
    # control_device = args.device
    experiment_batch_name = "{}_freelevel{}".format(args.exp_name, args.free_level)
    seed = args.seed
    trial_name = "{}_{}_{}".format(experiment_batch_name, get_time_str(), uuid.uuid4().hex[:8])
    print("Trial name is set to: ", trial_name)

    use_wandb = args.wandb
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

    # ===== Setup the config =====
    config = dict(

        # Environment config
        env_config=dict(

            # Original real human exp env config:
            # use_render=True,  # Open the interface
            # manual_control=True,  # Allow receiving control signal from external device
            # controller=control_device,
            # window_size=(1600, 1100),

            # FakeHumanEnv config:
            free_level=free_level,
        ),

        # Algorithm config
        algo=dict(
            # intervention_start_stop_td=args.intervention_start_stop_td,
            adaptive_batch_size=args.adaptive_batch_size,
            bc_loss_weight=args.bc_loss_weight,
            only_bc_loss="False",
            add_bc_loss="True" if args.bc_loss_weight > 0.0 else "False",
            use_balance_sample=True,
            agent_data_ratio=1.0,
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(
                discard_reward=True,  # We run in reward-free manner!
            ),

            # policy=TD3Policy,
            # policy_kwargs=dict(net_arch=[256, 256]),
            policy=PVPContinuousPolicy,
            policy_kwargs=dict(
                features_extractor_class=GRUFeatureExtractor,
                features_extractor_kwargs=dict(
                    features_dim=256,
                    transformer_hidden_dim=512,
                    horizon=10 + 1,
                    use_continuous_action_space=True,
                    net_arch=[],
                ),
                net_arch=[],
                one_hot_discrete=False,
                share_features_extractor=False,
            ),
            env=None,
            learning_rate=1e-4,
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=50_000,  # We only conduct experiment less than 50K steps
            learning_starts=args.learning_starts,  # The number of steps before
            batch_size=128,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),
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
    if args.toy_env:
        config["env_config"].update(
            # Here we set num_scenarios to 1, remove all traffic, and fix the map to be a very simple one.
            num_scenarios=1,
            traffic_density=0.0,
            map="COT"
        )

    # ===== Setup the training environment =====
    train_env = FakeHumanEnv(config=config["env_config"], )

    train_env = HistoryWrapper(train_env, horizon=10, include_first_frame=False)

    train_env = Monitor(env=train_env, filename=str(trial_dir))
    # Store all shared control data to the files.
    train_env = SharedControlMonitor(env=train_env, folder=trial_dir / "data", prefix=trial_name)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

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

        eval_env = HistoryWrapper(eval_env, horizon=10, include_first_frame=False)

        eval_env = Monitor(env=eval_env, filename=str(trial_dir))
        return eval_env

    eval_env = SubprocVecEnv([_make_eval_env])

    # ===== Setup the callbacks =====
    save_freq = args.save_freq  # Number of steps per model checkpoint
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
    if args.ckpt:
        ckpt = Path(args.ckpt)
        print(f"Loading checkpoint from {ckpt}!")
        from pvp.sb3.common.save_util import load_from_zip_file
        data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
        model.set_parameters(params, exact_match=True, device=model.device)

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=50_000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        # eval_env=None,
        # eval_freq=-1,
        # n_eval_episodes=2,
        # eval_log_path=None,

        # eval
        eval_env=eval_env,
        eval_freq=500,
        n_eval_episodes=10,
        eval_log_path=str(trial_dir),

        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
        save_buffer=False,
        load_buffer=False,
    )
