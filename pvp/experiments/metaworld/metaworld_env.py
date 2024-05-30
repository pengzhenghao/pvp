"""
Simple wrapper for registering metaworld enviornments
properly with gym.
"""
import gym
import torch
import numpy as np
import copy
import time
from collections import deque
import pathlib
import math
from metadrive.utils.math import safe_clip



def trim_mw_obs(obs):
    # Remove the double robot observation from the environment.
    # Only stack two object observations
    # this helps keep everything more markovian
    return np.concatenate((obs[:18], obs[22:]), dtype=np.float32)


class MetaWorldSawyerEnv(gym.Env):
    def __init__(self, env_name, seed=False, randomize_hand=True, sparse: bool = False, horizon: int = 250):
        from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

        self.env_name = env_name
        self._env = ALL_V2_ENVIRONMENTS[env_name]()
        self._env._freeze_rand_vec = False
        self._env._partially_observable = False
        self._env._set_task_called = True
        self._seed = seed
        if self._seed:
            self._env.seed(0)  # Seed it at zero for now.
        self.randomize_hand = randomize_hand
        self.sparse = sparse
        assert self._env.observation_space.shape[0] == 39
        low, high = self._env.observation_space.low, self._env.observation_space.high
        self.observation_space = gym.spaces.Box(low=trim_mw_obs(low), high=trim_mw_obs(high), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self._max_episode_steps = min(horizon, self._env.max_path_length)

    def seed(self, seed=None):
        super().seed(seed=seed)
        if self._seed:
            self._env.seed(0)

    def step(self, action):
        self._episode_steps += 1
        obs, reward, done, info = self._env.step(action)
        if self._episode_steps == self._max_episode_steps:
            done = True
            info["discount"] = 1.0  # Ensure infinite boostrap.
        # Remove history from the observations. It makes it too hard to reset.
        if self.sparse:
            reward = float(info["success"])  # Reward is now if we succeed or fail.
        else:
            reward = reward / 10
        return trim_mw_obs(obs.astype(np.float32)), reward, done, info

    def _get_obs(self):
        return trim_mw_obs(self._env._get_obs())

    def get_state(self):
        joint_state, mocap_state = self._env.get_env_state()
        qpos, qvel = joint_state.qpos, joint_state.qvel
        mocap_pos, mocap_quat = mocap_state
        self._split_shapes = np.cumsum(
            np.array([qpos.shape[0], qvel.shape[0], mocap_pos.shape[1], mocap_quat.shape[1]])
        )
        return np.concatenate([qpos, qvel, mocap_pos[0], mocap_quat[0], self._env._last_rand_vec], axis=0)

    def set_state(self, state):
        joint_state = self._env.sim.get_state()
        if not hasattr(self, "_split_shapes"):
            self.get_state()  # Load the split
        qpos, qvel, mocap_pos, mocap_quat, rand_vec = np.split(state, self._split_shapes, axis=0)
        if not np.all(self._env._last_rand_vec == rand_vec):
            # We need to set the rand vec and then reset
            self._env._freeze_rand_vec = True
            self._env._last_rand_vec = rand_vec
            self._env.reset()
        joint_state.qpos[:] = qpos
        joint_state.qvel[:] = qvel
        self._env.set_env_state((joint_state, (np.expand_dims(mocap_pos, axis=0), np.expand_dims(mocap_quat, axis=0))))
        self._env.sim.forward()

    def reset(self, **kwargs):
        self._episode_steps = 0
        self._env.reset(**kwargs).astype(np.float32)
        if self.randomize_hand:
            # Hand init pos is usually set to self.init_hand_pos
            # We will add some uniform noise to it.
            high = np.array([0.25, 0.15, 0.2], dtype=np.float32)
            hand_init_pos = self.hand_init_pos + np.random.uniform(low=-high, high=high)
            hand_init_pos = np.clip(hand_init_pos, a_min=self._env.mocap_low, a_max=self._env.mocap_high)
            hand_init_pos = np.expand_dims(hand_init_pos, axis=0)
            for _ in range(50):
                self._env.data.set_mocap_pos("mocap", hand_init_pos)
                self._env.data.set_mocap_quat("mocap", np.array([1, 0, 1, 0]))
                self._env.do_simulation([-1, 1], self._env.frame_skip)

        # Get the obs once to reset history.
        self._get_obs()
        return self._get_obs().astype(np.float32)

    def render(self, mode="rgb_array", camera_name="corner2", width=640, height=480):
        # todo: rendering currently does work with status "ERROR: GLEW initalization error: Missing GL version"
        # todo: for now I have just removed all env.render() calls
        assert mode == "rgb_array", "Only RGB array is supported"
        # stack multiple views
        for ctx in self._env.sim.render_contexts:
            ctx.opengl_context.make_context_current()
        return self._env.render(offscreen=True, camera_name=camera_name, resolution=(width, height))

    def __getattr__(self, name):
        return getattr(self._env, name)

# Currently not used, included in the original CPL environment for metaworld but need to fix rendering issue first
# class MetaWorldSawyerImageWrapper(gym.Wrapper):
#     def __init__(self, env, width=64, height=64, camera="corner2", show_goal=False):
#         assert isinstance(
#             env.unwrapped, MetaWorldSawyerEnv
#         ), "MetaWorld Wrapper must be used with a MetaWorldSawyerEnv class"
#         super().__init__(env)
#         self._width = width
#         self._height = height
#         self._camera = camera
#         self._show_goal = show_goal
#         shape = (3, self._height, self._width)
#         self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
#
#     def _get_image(self):
#         if not self._show_goal:
#             try:
#                 self.env.unwrapped._set_pos_site("goal", np.inf * self.env.unwrapped._target_pos)
#             except ValueError:
#                 pass  # If we don't have the goal site, just continue.
#         img = self.env.render(mode="rgb_array", camera_name=self._camera, width=self._width, height=self._height)
#         return img.transpose(2, 0, 1)
#
#     def step(self, action):
#         state_obs, reward, done, info = self.env.step(action)
#         # Throw away the state-based observation.
#         info["state"] = state_obs
#         return self._get_image().copy(), reward, done, info
#
#     def reset(self):
#         # Zoom in camera corner2 to make it better for control
#         # I found this view to work well across a lot of the tasks.
#         camera_name = "corner2"
#         # Original XYZ is 1.3 -0.2 1.1
#         index = self.model.camera_name2id(camera_name)
#         self.model.cam_fovy[index] = 20.0  # FOV
#         self.model.cam_pos[index][0] = 1.5  # X
#         self.model.cam_pos[index][1] = -0.35  # Y
#         self.model.cam_pos[index][2] = 1.1  # Z
#
#         self.env.reset()
#         return self._get_image().copy()  # Return the image observation
#
#
# def get_mw_image_env(env_name, **kwargs):
#     env = MetaWorldSawyerEnv(env_name, **kwargs)
#     return MetaWorldSawyerImageWrapper(env)


class HumanInTheLoopEnv(MetaWorldSawyerEnv):
    """
    Human-in-the-loop Env Wrapper for the Safety Env in MetaDrive.
    Add code for computing takeover cost and add information to the interface.
    """
    total_steps = 0
    total_takeover_cost = 0
    total_takeover_count = 0
    total_cost = 0
    takeover = False
    takeover_recorder = deque(maxlen=2000)
    agent_action = None
    in_pause = False
    start_time = time.time()

    def reset(self, *args, **kwargs):
        self.takeover = False
        self.agent_action = None
        return super().reset(**kwargs)

    def step(self, actions):
        self.agent_action = copy.copy(actions)
        ret = super(HumanInTheLoopEnv, self).step(actions)

        self.takeover_recorder.append(self.takeover)

        # todo: add optional render depending on "use_render" variable
        # if self.config["use_render"]:  # and self.config["main_exp"]: #and not self.config["in_replay"]:
        #     super(HumanInTheLoopEnv, self).render(
        #         text={
        #             "Total Cost": round(self.total_cost, 2),
        #             "Takeover Cost": round(self.total_takeover_cost, 2),
        #             "Takeover": "TAKEOVER" if self.takeover else "NO",
        #             "Total Step": self.total_steps,
        #             "Total Time": time.strftime("%M:%S", time.gmtime(time.time() - self.start_time)),
        #             "Takeover Rate": "{:.2f}%".format(np.mean(np.array(self.takeover_recorder) * 100)),
        #             "Pause": "Press E",
        #         }
        #     )

        self.total_steps += 1

        return ret


    def get_takeover_cost(self, info):
        """Return the takeover cost when intervened."""
        takeover_action = safe_clip(np.array(info["raw_action"]), -1, 1)
        agent_action = safe_clip(np.array(self.agent_action), -1, 1)
        multiplier = np.dot(takeover_action, agent_action)
        divident = np.linalg.norm(takeover_action) * np.linalg.norm(agent_action)
        if divident < 1e-6:
            cos_dist = 1.0
        else:
            cos_dist = multiplier / divident
        return 1 - cos_dist

#
FOLDER_PATH = pathlib.Path(__file__).parent

def get_expert():

    from pvp.sb3.common.save_util import load_from_zip_file
    from pvp.sb3.ppo import PPO
    from pvp.sb3.ppo.policies import ActorCriticPolicy

    train_env = HumanInTheLoopEnv(env_name='button-press-v2')

    # Initialize agent
    algo_config = dict(
        policy=ActorCriticPolicy,
        n_steps=1024,  # n_steps * n_envs = total_batch_size
        n_epochs=20,
        learning_rate=5e-5,
        batch_size=256,
        clip_range=0.1,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=10.0,
        # tensorboard_log=trial_dir,
        create_eval_env=False,
        verbose=2,
        # seed=seed,
        device="auto",
        env=train_env
    )
    model = PPO(**algo_config)

    ckpt = FOLDER_PATH / "metaworld_ppo_10m_steps"

    print(f"Loading checkpoint from {ckpt}!")
    data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
    model.set_parameters(params, exact_match=True, device=model.device)
    print(f"Model is loaded from {ckpt}!")

    train_env.close()

    return model.policy


def load():
    global _expert_weights
    if _expert_weights is None:
        _expert_weights = np.load("/metaworld_ppo_10m_steps.zip")
    return _expert_weights

def get_expert2():
    from metaworld.policies import SawyerButtonPressV2Policy
    return SawyerButtonPressV2Policy()

_expert = get_expert2()


class FakeHumanInTheLoopMetaWorld(HumanInTheLoopEnv):
    last_takeover = None
    last_obs = None
    expert = None

    def step(self, actions):
        """Compared to the original one, we call expert_action_prob here and implement a takeover function."""
        actions = np.asarray(actions).astype(np.float32)
        self.agent_action = copy.copy(actions)
        self.last_takeover = self.takeover

        # ===== Get expert action and determine whether to take over! =====
        if self.expert is None:
            global _expert
            self.expert = _expert
            print()

        # use below if using custom ppo agent as expert
        # last_obs, _ = self.expert.obs_to_tensor(self.last_obs)
        # distribution = self.expert.get_distribution(last_obs)
        # log_prob = distribution.log_prob(torch.from_numpy(actions).to(last_obs.device))
        # action_prob = log_prob.exp().detach().cpu().numpy()
        # expert_action = distribution.sample().detach().cpu().numpy()
        # assert expert_action.shape[0] == action_prob.shape[0] == 1
        # action_prob = action_prob[0]
        # expert_action = expert_action[0]
        # expert_action = expert_action.astype(np.float32)
        # # todo: change below to dependent on config['free_level'] as in metadrive example
        # if action_prob < 0.05:
        #     actions = expert_action
        #     self.takeover = True
        # else:
        #     self.takeover = False
        # print(f"Action probability: {action_prob}, agent action: {actions}, expert action: {expert_action}, takeover: {self.takeover}")

        # use below if using metaworld policy as expert
        action = self.expert.get_action(self.last_obs)

        # takeover when cosine similarily cost exceeds a certain value
        cosine_similarity = np.dot(action, actions) / (np.linalg.norm(action) * np.linalg.norm(actions))
        if np.linalg.norm(action) * np.linalg.norm(actions) < 1e-6:
            cosine_similarity = 1
        cosine_similarity_cost = 1 - cosine_similarity

        if cosine_similarity_cost > 0.2:
            self.takeover = True
            actions = action
        else:
            self.takeover = False

        ret = super(HumanInTheLoopEnv, self).step(actions)
        self.last_obs = ret[0]
        # add relevant things to info variable in order for shared_monitor to work
        ret[3]['raw_action'] = self.agent_action
        ret[3]['takeover_start'] = True if not self.last_takeover and self.takeover else False
        ret[3]['takeover'] = self.takeover and not ret[3]['takeover_start']
        ret[3]['takeover_cost'] = self.get_takeover_cost(ret[3]) if self.takeover else 0

        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1
        return ret

    def reset(self, **kwargs):
        o = super().reset(**kwargs)
        self.last_obs = o
        return o


if __name__ == '__main__':

    env = HumanInTheLoopEnv(env_name='button-press-v2')
    o = env.reset()
    print(env.action_space)
    steps = 0
    while True:
        _, _, done, info = env.step(env.action_space.sample())
        steps += 1
        print(info)
        if done or int(info["success"]) == 1:
            print(info, steps)
            env.reset()
            break
