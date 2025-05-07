#goal: add fake (imag) pos samples to neg trajs. see performance of algos
import copy
import math
import pathlib

import gymnasium as gym
import numpy as np
import torch

FOLDER_PATH = pathlib.Path(__file__).parent


import robosuite as suite
from robosuite import load_controller_config
# from thrifty.utils.hardcoded_nut_assembly import HardcodedPolicy
from robosuite.utils.transform_utils import pose2mat

from robosuite.wrappers import VisualizationWrapper
import numpy as np
import sys
import time

from gym import spaces
from gym.core import Env
from robosuite.wrappers import Wrapper

class GymWrapper(Wrapper, Env):
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None
        self.metadata = {}

        # set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def reset(self):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict)

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ob_dict, reward, done, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, done, info

    def seed(self, seed=None):
        """
        Utility function to set numpy seed

        Args:
            seed (None or int): If specified, numpy seed to set

        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()

def get_expert():
    from pvp.sb3.common.save_util import load_from_zip_file
    from pvp.sb3.ppo import PPO
    from pvp.sb3.ppo.policies import ActorCriticPolicy

    if True:
        render = False
        controller_config = load_controller_config(default_controller='OSC_POSE')
        configr = {
            "env_name": "Wipe",
            "robots": "UR5e",
            "controller_configs": controller_config,
        }
        env = suite.make(
                **configr,
                has_renderer=render,
                has_offscreen_renderer=False,
                render_camera="agentview",
                ignore_done=True,
                use_camera_obs=False,
                reward_shaping=True,
                control_freq=20,
                hard_reset=True,
                use_object_obs=True
            )
        train_env = GymWrapper(env)
        # train_env = VisualizationWrapper(train_env, indicator_configs=None)
        
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

    ckpt = FOLDER_PATH / "best_model.zip"

    print(f"Loading checkpoint from {ckpt}!")
    data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
    model.set_parameters(params, exact_match=True, device=model.device)
    print(f"Model is loaded from {ckpt}!")

    train_env.close()

    return model.policy

_expert = get_expert()
    
class CustomWrapper(gym.Env):
    last_takeover = None
    last_obs = None
    expert = None
    takeover = None
    last_turn = None
    from collections import deque 
    takeover_recorder = deque(maxlen=2000)
    total_steps = 0
    total_takeover_cost = 0
    total_takeover_count = 0
    total_cost = 0
    agent_action = None
    total_reward = 0
    rec = []
    t = 0
    def seed(self, seed=None):
        return
    def __init__(self, env, unwrapped_env, config):
        self.env = env
        self._env = unwrapped_env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.gripper_closed = False
        self.viewer = env.viewer
        self.robots = env.robots
        self.config = dict({
                "use_discrete": False,
                "disable_expert": False,
                "manual_control": False,
                "use_render": False,
                "expert_deterministic": True,
                "future_steps_predict": 20,
                "update_future_freq": 10,
                "stop_img_samples": 3,
                "future_steps_preference": 1,
                "expert_noise": 0,
                "switch_to_expert": 10,
                "eval": False,
                "MAX_EP_LEN": 600,
                "cos_similarity": False,
                })
        self.config.update(config)
        # if self.config["eval"]:
        #     self.config["use_render"] = False
        self._render = self.config["use_render"]
        if self.expert is None:
            global _expert
            self.expert = _expert
    def obs_to_tensor(self, obs):
        obs_tensor = torch.as_tensor(obs).to("cuda")
        return obs_tensor
    def obs_to_tensor_expand(self, obs):
        obs_tensor = torch.as_tensor(obs).to("cuda")
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor
    def get_takeover_cost(self, info):
        """Return the takeover cost when intervened."""
        if not self.config["cos_similarity"]:
            return 1
        takeover_action = np.clip(np.array(info["raw_action"]), -1, 1)
        agent_action = np.clip(np.array(self.agent_action), -1, 1)
        multiplier = (agent_action[0] * takeover_action[0] + agent_action[1] * takeover_action[1])
        divident = np.linalg.norm(takeover_action) * np.linalg.norm(agent_action)
        if divident < 1e-6:
            cos_dist = 1.0
        else:
            cos_dist = multiplier / divident
        return 1 - cos_dist
    
    def expert_act(self, o):
        # last_obs, _ = self.expert.obs_to_tensor(o)
        # distribution = self.expert.get_distribution(last_obs)
        # log_prob = distribution.log_prob(torch.from_numpy(actions).to(last_obs.device))
        # action_prob = log_prob.exp().detach().cpu().numpy()
        # action_prob = action_prob[0]
        expert_action, _  = self.expert.predict(o, deterministic=True)
        return expert_action
    
    def expert_value(self, o, a):
        o, a = self.obs_to_tensor_expand(o), self.obs_to_tensor_expand(a)
        values, log_prob, entropy = self.expert.evaluate_actions(o, a)
        return values[0].item(), log_prob[0].item()
    def reset(self):
        r = self.env.reset()
        self.render()
        settle_action = np.zeros(6)
        for _ in range(0):
            r, r2, r3, r4 = self.env.step(settle_action)
            self.render()
        self.gripper_closed = False
        self.last_obs = r
        self.last_takeover = None
        self.takeover = None
        self.t = 0
        return r

    def get_state(self):
        obj = self._env.robots[0].controller
        np_array_attributes = {}
        for attr_name in dir(obj):
            # Skip private or special attributes (e.g., __init__, __class__)
            if attr_name.startswith("_"):
                continue
            
            # Get the attribute value
            attr_value = getattr(obj, attr_name)
            
            # Check if the attribute is a NumPy array
            if isinstance(attr_value, np.ndarray):
                np_array_attributes[attr_name] = attr_value
        
        return copy.deepcopy(np_array_attributes)

    def set_state(self, state):
        obj = self._env.robots[0].controller
        for attr_name, attr_value in state.items():
            # Set the NumPy array attribute
            try:
                setattr(obj, attr_name, attr_value)
            except:
                pass
                # TODO: torque_compensation is not settable

    def predict_agent_future_trajectory(self, current_obs, n_steps, action_behavior = None, expert_mode = False):
        info = dict()
        saved_state = copy.deepcopy(self._env.sim.get_state())
        
        controller_state = self.get_state()
        L = len(self._env.wiped_markers)
        # # grip = self._env.robots[0].gripper.current_action[0]
        A = copy.deepcopy(self._env.sim.model.geom_rgba)
        C = copy.deepcopy(self._env._observables)
        
        last_turn = self.last_turn
        traj = []
        obs = current_obs
        total_reward = 0
        success = False
        total_action_diff = 0
        total_q_diff = 0
        total_log_prob = 0
        
        for step in range(n_steps):
            action = action_behavior
            if action_behavior is None:
                action = self.agent_action
                if hasattr(self, "model"):
                     action, _ = self.model.policy.predict(obs, deterministic=True)
            action_expert = self.expert_act(obs)
            action_expert = np.clip(action_expert, -1, 1)
            if expert_mode:
                action = action_expert
            # action_diff = np.linalg.norm(action - action_expert)
            action_diff = np.mean((action - action_expert) ** 2)
            
            q_diff, log_prob = self.expert_value(obs, action)
            
            total_q_diff += q_diff
            total_log_prob += log_prob
            total_action_diff += action_diff
            step_reward = 0
            o, r, d, i = self.env.step(action)
            
            # B = copy.deepcopy(self._env.sim.model.geom_rgba)
            # self.render()
            step_reward += r
            settle_action = np.zeros(6)
            for _ in range(0):
                o, r, d, i = self.env.step(settle_action)
                # self.render()
                step_reward += r
            d = self._check_success()
            total_reward += step_reward
            traj.append({
                "obs": obs.copy(),
                "action": action.copy(),
                "reward": step_reward,
                "next_obs": o.copy(),
                "done": d,
                "action_expert": action_expert.copy(),
            })
            obs = o.copy()
            if d:
                success = True
        info["clean"] = self._env._observables["proportion_wiped"].obs - C["proportion_wiped"].obs
        
        mean_action_diff = total_action_diff / n_steps
        mean_q_diff = total_q_diff / n_steps
        # self.last_turn = last_turn
        self._env.sim.set_state(saved_state)
        self.set_state(controller_state)
        self._env.wiped_markers = self._env.wiped_markers[:L]
        # # self._env.robots[0].gripper.current_action[0] = grip
        # # self.render()
        self._env.sim.model.geom_rgba[:] = A
        self._env._observables = C
        info["success"] = success
        info["total_reward"] = total_reward
        info["mean_action_diff"] = mean_action_diff
        info["mean_q_diff"] = mean_q_diff
        info["mean_log_prob"] = total_log_prob / n_steps
        return traj, info
        
    def decide_takeover(self, obs, future_steps_predict):
        if self.config["eval"]:
            return False
        # predicted_traj_real2, info_real2 = self.predict_agent_future_trajectory(obs, future_steps_predict, expert_mode=True)
        
        predicted_traj_real, info_real = self.predict_agent_future_trajectory(obs, future_steps_predict)
        #TODO: return other objectives. current: mean action difference
        
        # if info_real["mean_action_diff"] > self.config["switch_to_expert"]:
        #     self.rec.append(info_real2["total_reward"] - info_real["total_reward"])
        # return info_real["mean_action_diff"] > self.config["switch_to_expert"] 
        # return (info_real2["total_reward"] - info_real["total_reward"] > self.config["switch_to_expert"]) or (info_real["total_reward"] < 2)
        # return info_real["mean_action_diff"] > 0.5
        # print("two clean:", info_real2["clean"], info_real["clean"])
        # print("two reward:", info_real2["total_reward"], info_real["total_reward"])
        # self.rec.append(info_real2["clean"])
        # return (info_real2["clean"] - info_real["clean"] > 0.2) or (info_real["mean_log_prob"] < -10)
        # return info_real["total_reward"] < 1
        # return info_real["mean_action_diff"] > 0.5
        #examine the proportion of elimination mark!!
        return info_real["total_reward"] < 1
    
    def store_preference_pairs(self, predicted_traj, future_steps_preference, expert_action):
        for step in range(min(len(predicted_traj) - 1, future_steps_preference)):
            step_info = {
                "obs": predicted_traj[step]["obs"].copy(),
                "action": expert_action.copy(),
                "next_obs": predicted_traj[step]["obs"].copy(),
                "done": False,
            }
            positive_traj = [step_info].copy()
            negative_traj = predicted_traj[step+1:]
            self.model.imagreplay_buffer.add(positive_traj, negative_traj)
            
    def step(self, action):
        # abstract 10 actions as 1 action
        # get rid of x/y rotation, which is unintuitive for remote teleop
        action_ = action.copy()
        # action_[3] = 0.
        # action_[4] = 0.
        
        self.agent_action = copy.copy(action_)
        self.last_takeover = self.takeover
        future_steps_predict = self.config["future_steps_predict"]
        update_future_freq = self.config["update_future_freq"]
        future_steps_preference = self.config["future_steps_preference"]
        expert_noise_bound = self.config["expert_noise"]
        
        if not self.config["eval"]:
            expert_action = self.expert_act(self.last_obs)
            expert_action = np.clip(expert_action, -1, 1)
            enoise = np.random.randn(6) * expert_noise_bound
            expert_action = np.clip(enoise + expert_action, -1, 1)
        else:
            expert_action = action_.copy()
        if self.takeover == None or (self.total_steps % update_future_freq == 0):
            self.takeover = self.decide_takeover(self.last_obs, future_steps_predict)
            # if len(self.rec) > 0:
            #     print("MEAN:", np.mean(self.rec))
        
        # self.takeover2 = (expert_action[-1] * action_[-1] < 0) and not self.config["eval"]
        self.takeover2 = False 
        
        if self.takeover or self.takeover2:
            #TODO: add to preference buffer
            # if hasattr(self, "model") and hasattr(self.model, "imagreplay_buffer"):
            #     predicted_traj, info2 = self.predict_agent_future_trajectory(self.last_obs, future_steps_predict, action_behavior=self.agent_action.copy())
            action_ = expert_action
            # if hasattr(self, "model") and hasattr(self.model, "imagreplay_buffer"):
            #     self.store_preference_pairs(predicted_traj, future_steps_preference, expert_action.copy())
        else:
            self.takeover = False

        step_reward = 0
        # time.sleep(1)
        o, r, d, i = self.env.step(action_)
        # time.sleep(1)
        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1
        self.total_reward += r
        step_reward += r
        self.render()
        settle_action = np.zeros(6)
        for _ in range(0):
            o, r, d, i = self.env.step(settle_action)
            self.render()
            self.total_reward += r
            step_reward += r
        
        # self.gripper_closed = self._env._check_grasp(gripper=self._env.robots[0].gripper, object_geoms=[g for g in self._env.nuts[self._env.nut_id].contact_geoms])
        self.gripper_closed = False
        self.last_obs = o
        i["raw_action"] = copy.copy(action_)
        i["step_reward"] = step_reward
        i["action_diff_new"] = np.mean((self.agent_action - expert_action) ** 2)
        i["takeover"]  = (self.takeover == True) or (self.takeover2 == True)
        i["takeover_cost"] = self.get_takeover_cost(i)
        i["grip_wrong"] = self.takeover2
        i["gripper_closed"] = self.gripper_closed
        i["takeover_start"] = True if not self.last_takeover and self.takeover else False
        # condition = i["takeover_start"] if self.config["only_takeover_start_cost"] else self.takeover
        self.total_takeover_cost += i["takeover_cost"]
        self.total_takeover_count += 1 if self.takeover else 0
        i["total_takeover_count"] = self.total_takeover_count
        i["total_takeover_cost"] = self.total_takeover_cost
        i["total_reward"] = self.total_reward
        self.t += 1
        d = (self.t >= self.config['MAX_EP_LEN']) or self._check_success()
        i["is_success"] = int(self._check_success())
        i["episode_length"] = self.t
        r += i["is_success"]
        return o, r, d, i

    def _check_success(self):
        return self.env._check_success()

    def render(self):
        if self._render:
            self.env.render()

if __name__ == "__main__":
    
    render = True
    controller_config = load_controller_config(default_controller='OSC_POSE')
    config = {
        "env_name": "Wipe",
        "robots": "UR5e",
        "controller_configs": controller_config,
    }
    env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="agentview",
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True
        )
    unwrapped_env = env
    env = GymWrapper(env)
    env = VisualizationWrapper(env, indicator_configs=None)
    env = CustomWrapper(env, unwrapped_env, config=dict({"use_render": render}))

    arm_ = 'right'
    config_ = 'single-arm-opposed'
    active_robot = env.robots[arm_ == 'left']
    # expert_pol = HardcodedPolicy(env).act
    
    num_episodes = 50
    i, failures = 0, 0
    np.random.seed(0)
    obs, act, rew = [], [], []
    act_limit = env.action_space.high[0]
    success_rate = 0
    while i < num_episodes:
        print('Episode #{}'.format(i))
        o, total_ret, d, t = env.reset(), 0, False, 0
        curr_obs, curr_act = [], []
        while not d:
            # a = expert_pol(o)
            a = np.zeros(6)
            o, r, d, _ = env.step(a)
            #print(r)
            t += 1
            success_rate += int(env._check_success())
        i += 1
        print("SR: ", success_rate / i, t)