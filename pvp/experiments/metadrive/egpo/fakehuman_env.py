import copy
import math
import pathlib

import gymnasium as gym
import numpy as np
import torch
from metadrive.engine.logger import get_logger
from metadrive.examples.ppo_expert.numpy_expert import ckpt_path
from metadrive.policy.env_input_policy import EnvInputPolicy

from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv

FOLDER_PATH = pathlib.Path(__file__).parent

logger = get_logger()


def get_expert():
    # Return IDM Policy class instead of PPO policy
    from metadrive.policy.idm_policy import IDMPolicy
    return IDMPolicy


def obs_correction(obs):
    # due to coordinate correction, this observation should be reversed
    obs[15] = 1 - obs[15]
    obs[10] = 1 - obs[10]
    return obs


def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2 * math.pi * var)**.5
    num = math.exp(-(float(x) - float(mean))**2 / (2 * var))
    return num / denom


def load():
    global _expert_weights
    if _expert_weights is None:
        _expert_weights = np.load(ckpt_path)
    return _expert_weights


_expert_class = get_expert()


class FakeHumanEnv(HumanInTheLoopEnv):
    last_takeover = None
    last_obs = None
    expert = None

    def __init__(self, config):
        super(FakeHumanEnv, self).__init__(config)
        from metadrive.obs.state_obs import LidarStateObservation
        self.lidar = LidarStateObservation(self.config)
        if self.config["use_discrete"]:
            self._num_bins = 13
            self._grid = np.linspace(-1, 1, self._num_bins)
            self._actions = np.array(np.meshgrid(self._grid, self._grid)).T.reshape(-1, 2)

    @property
    def action_space(self) -> gym.Space:
        if self.config["use_discrete"]:
            return gym.spaces.Discrete(self._num_bins**2)
        else:
            return super(FakeHumanEnv, self).action_space

    # def _preprocess_actions(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray], int]) -> Union[np.ndarray, Dict[AnyStr, np.ndarray], int]:
    #     if self.config["use_discrete"]:
    #         print(111)
    #         return int(actions)
    #     else:
    #         return actions

    def default_config(self):
        """Revert to use the RL policy (so no takeover signal will be issued from the human)"""
        config = super(FakeHumanEnv, self).default_config()
        config.update(
            {
                "use_discrete": False,
                "disable_expert": False,
                "agent_policy": EnvInputPolicy,
                "free_level": 0.95,
                "manual_control": False,
                "use_render": False,
                "expert_deterministic": False,
            }
        )
        return config

    def continuous_to_discrete(self, a):
        distances = np.linalg.norm(self._actions - a, axis=1)
        discrete_index = np.argmin(distances)
        return discrete_index

    def discrete_to_continuous(self, a):
        continuous_action = self._actions[a.astype(int)]
        return continuous_action

    def step(self, actions):
        """Compared to the original one, we call expert_action_prob here and implement a takeover function."""
        actions = np.asarray(actions).astype(np.float32)

        if self.config["use_discrete"]:
            actions = self.discrete_to_continuous(actions)

        self.agent_action = copy.copy(actions)
        self.last_takeover = self.takeover

        # ===== Get expert action and determine whether to take over! =====

        if self.config["disable_expert"]:
            pass

        else:
            if self.expert is None:
                global _expert_class
                # Initialize IDM Policy with the agent vehicle
                self.expert = _expert_class(control_object=self.agent, random_seed=self.engine.global_random_seed)
            
            # IDM Policy's act() method doesn't need observation, it uses control_object directly
            expert_action = self.expert.act()
            if self.total_steps > 0:
                self.takeover = True
                actions = np.array(expert_action, dtype=np.float32)
            else:
                self.takeover = False

        o, r, d, i = super(HumanInTheLoopEnv, self).step(actions)
        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1


        if self.config["use_render"]:  # and self.config["main_exp"]: #and not self.config["in_replay"]:
            super(HumanInTheLoopEnv, self).render(
                text={
                    "Total Cost": round(self.total_cost, 2),
                    "Takeover Cost": round(self.total_takeover_cost, 2),
                    "Takeover": "TAKEOVER" if self.takeover else "NO",
                    "Total Step": self.total_steps,
                    # "Total Time": time.strftime("%M:%S", time.gmtime(time.time() - self.start_time)),
                    "Takeover Rate": "{:.2f}%".format(np.mean(np.array(self.takeover_recorder) * 100)),
                    "Pause": "Press E",
                }
            )

        assert i["takeover"] == self.takeover

        if self.config["use_discrete"]:
            i["raw_action"] = self.continuous_to_discrete(i["raw_action"])

        return o, r, d, i

    def _get_step_return(self, actions, engine_info):
        """Compared to original one, here we don't call expert_policy, but directly get self.last_takeover."""
        o, r, tm, tc, engine_info = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info)
        self.last_obs = o
        d = tm or tc
        last_t = self.last_takeover
        engine_info["takeover_start"] = True if not last_t and self.takeover else False
        engine_info["takeover"] = self.takeover
        condition = engine_info["takeover_start"] if self.config["only_takeover_start_cost"] else self.takeover
        if not condition:
            engine_info["takeover_cost"] = 0
        else:
            cost = self.get_takeover_cost(engine_info)
            self.total_takeover_cost += cost
            engine_info["takeover_cost"] = cost
        engine_info["total_takeover_cost"] = self.total_takeover_cost
        engine_info["native_cost"] = engine_info["cost"]
        engine_info["episode_native_cost"] = self.episode_cost
        self.total_cost += engine_info["cost"]
        self.total_takeover_count += 1 if self.takeover else 0
        engine_info["total_takeover_count"] = self.total_takeover_count
        engine_info["total_cost"] = self.total_cost
        # engine_info["total_cost_so_far"] = self.total_cost
        return o, r, d, engine_info

    def _get_reset_return(self, reset_info):
        o, info = super(HumanInTheLoopEnv, self)._get_reset_return(reset_info)
        self.last_obs = o
        self.last_takeover = False
        # Reset and re-initialize IDM Policy with the new agent vehicle after reset
        if self.expert is not None:
            self.expert.reset()
            self.expert.control_object = self.agent
        else:
            # Initialize IDM Policy if not already initialized
            global _expert_class
            self.expert = _expert_class(control_object=self.agent, random_seed=self.engine.global_random_seed)
        return o, info


if __name__ == "__main__":
    env = FakeHumanEnv(dict(free_level=0.95, use_render=False))
    env.reset()
    while True:
        _, _, done, info = env.step([0, 1])
        # done = tm or tc
        # env.render(mode="topdown")
        if done:
            print(info)
            env.reset()
