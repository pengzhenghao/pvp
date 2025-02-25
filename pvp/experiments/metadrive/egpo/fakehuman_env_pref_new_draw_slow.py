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
    from pvp.sb3.common.save_util import load_from_zip_file
    from pvp.sb3.ppo import PPO
    from pvp.sb3.ppo.policies import ActorCriticPolicy

    train_env = HumanInTheLoopEnv(config={'manual_control': False, "use_render": True})

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

    ckpt = FOLDER_PATH / "metadrive_pvp_20m_steps"

    print(f"Loading checkpoint from {ckpt}!")
    data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
    model.set_parameters(params, exact_match=True, device=model.device)
    print(f"Model is loaded from {ckpt}!")

    train_env.close()

    return model.policy


def obs_correction(obs):
    # due to coordinate correction, this observation should be reversed
    obs[15] = 1 - obs[15]
    obs[10] = 1 - obs[10]
    return obs


def normpdf(x, mean, sd):
    var = float(sd) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom


def load():
    global _expert_weights
    if _expert_weights is None:
        _expert_weights = np.load(ckpt_path)
    return _expert_weights


_expert = get_expert()


class FakeHumanEnvPref(HumanInTheLoopEnv):
    last_takeover = None
    last_obs = None
    expert = None
    pending_human_traj = []
    pending_agent_traj = []
    etakeover = True
    from collections import deque 
    advantages = deque(maxlen = 200)
    drawn_points = []
    
    def __init__(self, config):
        super(FakeHumanEnvPref, self).__init__(config)
        if self.config["use_discrete"]:
            self._num_bins = 13
            self._grid = np.linspace(-1, 1, self._num_bins)
            self._actions = np.array(np.meshgrid(self._grid, self._grid)).T.reshape(-1, 2)

    @property
    def action_space(self) -> gym.Space:
        if self.config["use_discrete"]:
            return gym.spaces.Discrete(self._num_bins ** 2)
        else:
            return super(FakeHumanEnvPref, self).action_space

    # def _preprocess_actions(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray], int]) -> Union[np.ndarray, Dict[AnyStr, np.ndarray], int]:
    #     if self.config["use_discrete"]:
    #         print(111)
    #         return int(actions)
    #     else:
    #         return actions

    def default_config(self):
        """Revert to use the RL policy (so no takeover signal will be issued from the human)"""
        config = super(FakeHumanEnvPref, self).default_config()
        config.update(
            {
                "use_discrete": False,
                "disable_expert": False,

                "agent_policy": EnvInputPolicy,
                "free_level": 0.9,
                "manual_control": False,
                "use_render": False,
                "expert_deterministic": False,
                "future_steps": 15,
                "takeover_see": 15,
                "stop_freq": 5,
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
    # def get_state(self) -> dict:
    #     import copy
    #     state = copy.deepcopy(self.vehicle.get_state())
    #     return copy.deepcopy(state)

    # def set_state(self, state: dict):
    #     self.vehicle.set_state(state)
    def get_state(self) -> dict:
        state = dict()
        state["episode_rewards"] = self.episode_rewards.copy()  # defaultdict 也可以转为普通 dict
        state["episode_lengths"] = self.episode_lengths.copy()
        state["dones"] = self.dones.copy()
        state["episode_step"] = self.episode_step
        import copy
        state["vehicle"] = copy.deepcopy(self.vehicle.get_state())
        agent_states = dict()
        for agent_id, agent in self.agents.items():
            if hasattr(agent, "get_global_states") and callable(agent.get_global_states):
                agent_states[agent_id] = agent.get_global_states()
            elif hasattr(agent, "get_state") and callable(agent.get_state):
                agent_states[agent_id] = agent.get_state()
            else:
                agent_states[agent_id] = copy.deepcopy(agent)
        manager_states = dict()
        for agent_id, agent in self.engine.managers.items():
            if hasattr(agent, "get_global_states") and callable(agent.get_global_states):
                manager_states[agent_id] = agent.get_global_states()
            elif hasattr(agent, "get_state") and callable(agent.get_state):
                manager_states[agent_id] = agent.get_state()
        state["manager_states"] = manager_states
        state["agent_states"] = agent_states
        if hasattr(self, "last_obs"):
            state["last_obs"] = self.last_obs  
        return copy.deepcopy(state)

    def set_state(self, state: dict):
        self.episode_rewards = state.get("episode_rewards", self.episode_rewards)
        self.episode_lengths = state.get("episode_lengths", self.episode_lengths)
        self.dones = state.get("dones", self.dones)
        self.vehicle.set_state(state.get("vehicle", self.vehicle))
        if "episode_step" in state and hasattr(self.engine, "episode_step"):
            self.engine.episode_step = state["episode_step"]
        # 还原 agents 状态（前提是各 agent 实现了 get_state/set_state）
        agent_states = state.get("agent_states", dict())
        for agent_id, agent in self.agents.items():
            if agent_id in agent_states and hasattr(agent, "set_state") and callable(agent.set_state):
                agent.set_state(agent_states[agent_id])
                
        manager_states = state.get("manager_states", dict())
        for agent_id, agent in self.engine.managers.items():
            if agent_id in manager_states and hasattr(agent, "set_global_states"):
                agent.set_global_states(manager_states[agent_id])
            elif agent_id in manager_states and hasattr(agent, "set_state") and callable(agent.set_state):
                agent.set_state(manager_states[agent_id])
        # 还原 self.last_obs 等变量
        if "last_obs" in state:
            self.last_obs = state["last_obs"]

    def _predict_agent_future_trajectory(self, current_obs, n_steps, use_exp = False):
        saved_state = self.get_state()
        traj = []
        obs = current_obs
        lstprob = []
        total_reward = 0
        for step in range(n_steps):
            if not use_exp:
                if hasattr(self, "model"):
                    action, _ = self.model.policy.predict(obs, deterministic=True)
                else:
                    action = self.agent_action
                #action, _ = self.model._sample_action(learning_starts=self.model.learning_starts,
                #                                    obs=obs, deterministic=True)
                #assert False
            else:
                action, _  = self.expert.predict(obs, deterministic=True)
                #action = np.array([0, 1])
            if self.config["use_discrete"]:
                action_cont = self.discrete_to_continuous(action)
            else:
                action_cont = action

            #o, r, d, i = super(HumanInTheLoopEnv, self).step(action_cont)
            
            self.engine.notrender = True
            
            actions = self._preprocess_actions(action_cont)  # preprocess environment input
            engine_info = self._step_simulator(actions)  # step the simulation
            while self.in_stop:
                self.engine.taskMgr.step()  # pause simulation
            o, r, tm, tc, i = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info=engine_info)
            d = tm or tc
            
            del self.engine.notrender
            total_reward += r
            d = self.done_function('default_agent')[0]

                        
            last_obs, _ = self.expert.obs_to_tensor(obs)
            distribution = self.expert.get_distribution(last_obs)
            log_prob = distribution.log_prob(torch.from_numpy(action_cont).to(last_obs.device))
            action_prob = log_prob.exp().detach().cpu().numpy()
            action_prob = action_prob[0]
            lstprob.append(action_prob)
            expert_action, _ = self.expert.predict(obs, deterministic=True)
            expert_action_clip = np.clip(expert_action, self.action_space.low, self.action_space.high)
            actions_n, values_n, log_prob_n = self.expert(torch.Tensor(obs).to(self.expert.device).unsqueeze(0))
            traj.append({
                "obs": obs.copy(),
                "action": action_cont.copy(),
                "reward": r,
                "next_obs": o,
                "done": d,
                "next_pos": copy.deepcopy(self.vehicle.position),
                "action_exp": expert_action_clip.copy(),
                "action_nov": action_cont.copy(),
                "values_n": values_n.item(),
            })
            obs = o
            if d:
                if r < 0:
                    total_reward = -100
                break
        self.set_state(saved_state)
        from pvp.sb3.common.utils import safe_mean
        if total_reward > 0:
            total_reward += values_n.item()
        return traj, safe_mean(lstprob[:self.config["takeover_see"]]), total_reward
    def step(self, actions):
        """Compared to the original one, we call expert_action_prob here and implement a takeover function."""
        actions = np.asarray(actions).astype(np.float32)

        if self.config["use_discrete"]:
            actions = self.discrete_to_continuous(actions)

        self.agent_action = copy.copy(actions)
        self.last_takeover = self.takeover
        
        future_steps = self.config["future_steps"]
        stop_freq = self.config["stop_freq"]
        self.human_traj = []
        if self.expert is None:
                global _expert
                self.expert = _expert
                
        if self.total_steps % stop_freq == 0:
            predicted_traj_exp, acprob, total_reward_exp = self._predict_agent_future_trajectory(self.last_obs, future_steps, use_exp=True)
            
            predicted_traj, acprob, total_reward = self._predict_agent_future_trajectory(self.last_obs, future_steps)
            
            advantage = total_reward_exp - total_reward
            if len(self.advantages) < 10 or total_reward < 0:
                self.etakeover = True
                if total_reward > 0:
                    self.advantages.append(advantage)
            else:
                q = np.quantile(list(self.advantages), self.config["free_level"])
                self.etakeover = (advantage > q)
                if advantage > q:
                    self.etakeover = True
                self.advantages.append(advantage)
        else:
            predicted_traj_exp, predicted_traj = [], []
        etakeover = self.etakeover
        # ===== Get expert action and determine whether to take over! =====

        if self.config["disable_expert"]:
            pass

        else:
            last_obs, _ = self.expert.obs_to_tensor(self.last_obs)
            distribution = self.expert.get_distribution(last_obs)
            log_prob = distribution.log_prob(torch.from_numpy(actions).to(last_obs.device))
            action_prob = log_prob.exp().detach().cpu().numpy()

            if self.config["expert_deterministic"]:
                expert_action = distribution.mode().detach().cpu().numpy()
            else:
                expert_action = distribution.sample().detach().cpu().numpy()

            
            # if np.any(expert_action != expert_action_clip):
            #     print(expert_action_clip, expert_action)
            
            assert expert_action.shape[0] == action_prob.shape[0] == 1
            action_prob = action_prob[0]
            expert_action, _  = self.expert.predict(self.last_obs, deterministic=True)
            
            expert_action_clip = np.clip(expert_action, self.action_space.low, self.action_space.high)
            
            if etakeover:

                # print(f"Action probability: {action_prob}, agent action: {actions}, expert action: {expert_action},")

                if self.config["use_discrete"]:
                    expert_action = self.continuous_to_discrete(expert_action)
                    expert_action = self.discrete_to_continuous(expert_action)

                actions = expert_action

                self.takeover = True
            else:
                self.takeover = False
            # print(f"Action probability: {action_prob:.3f}, agent action: {actions}, expert action: {expert_action}, takeover: {self.takeover}")
        if self.config["use_render"]:
            if hasattr(self,"drawer"):
                drawer = self.drawer # create a point drawer
            else:
                self.drawer = self.engine.make_point_drawer(scale=3)
                drawer = self.drawer 
            # if len(predicted_traj) > 0:
            #     #drawer.reset()
            #     for npp in self.drawn_points:
            #         npp.detachNode()
            #         self.drawer._dying_points.append(npp)
            #     self.drawn_points = []
            points, colors = [], []
            for j in range(len(predicted_traj)):
                points.append((predicted_traj[j]["next_pos"][0], predicted_traj[j]["next_pos"][1], 0.5)) # define line 1 for test
                color=(1,105/255,180/255)
                colors.append(np.clip(np.array([*color,1]), 0., 1.0))
            self.drawn_points = self.drawn_points + drawer.draw_points(points, colors) # draw points
        if self.config["use_render"]:
            if hasattr(self,"drawer"):
                drawer = self.drawer # create a point drawer
            else:
                self.drawer = self.engine.make_point_drawer(scale=3)
                drawer = self.drawer 
            points, colors = [], []
            for j in range(len(predicted_traj_exp)):
                points.append((predicted_traj_exp[j]["next_pos"][0], predicted_traj_exp[j]["next_pos"][1], 0.5)) # define line 1 for test
                color=(105/255,180/255, 1)
                colors.append(np.clip(np.array([*color,1]), 0., 1.0))
            self.drawn_points = self.drawn_points + drawer.draw_points(points, colors) # draw points
        if self.takeover:
            
            self.pending_agent_traj.append(predicted_traj)
        else:
            predicted_traj = []
        

            
        self.vehicle.real = True
        last_o = self.last_obs.copy()
        o, r, d, i = super(HumanInTheLoopEnv, self).step(actions)
        
        if len(self.advantages) >= 10:
            i["int_thred"] = np.quantile(list(self.advantages), self.config["free_level"])
        
        if hasattr(self,"drawer"):
                drawer = self.drawer # create a point drawer
        else:
                self.drawer = self.engine.make_point_drawer(scale=3)
                drawer = self.drawer 
        points, colors = [], []
        for j in range(1):
            points.append((self.vehicle.position[0], self.vehicle.position[1], 0.5)) # define line 1 for test
            color=(105/255,1,180/255)
            colors.append(np.clip(np.array([*color,1]), 0., 1.0))
        self.drawn_points = self.drawn_points + drawer.draw_points(points, colors)
        
        if self.takeover:
            self.pending_human_traj.append(self.human_traj)
            for lst in self.pending_human_traj:
                if len(lst) < future_steps:
                    lst.append({
                        "obs": last_o.copy(),
                        "action": expert_action_clip.copy(),
                        "next_obs": o.copy(),
                        "reward": r,
                        "done": d,
                        "next_pos": copy.deepcopy(self.vehicle.position),
                        "action_exp": expert_action_clip.copy(),
                        "action_nov": self.agent_action.copy(),
                    })
        else:
            if hasattr(self, "model"):
                assert len(self.pending_agent_traj) == len(self.pending_human_traj)
                for step in range(len(self.pending_agent_traj)):
                    if len(self.pending_agent_traj[step]) > 0 and hasattr(self.model, "prefreplay_buffer"):
                        self.model.prefreplay_buffer.add(self.pending_human_traj[step], self.pending_agent_traj[step])
            self.pending_agent_traj = []
            self.pending_human_traj = []
        
        self.vehicle.real = False
        position, velocity, speed, heading = copy.copy(self.vehicle.position), copy.copy(self.vehicle.velocity), copy.copy(self.vehicle.speed), copy.copy(self.vehicle.heading_theta)
        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1

        if not self.config["disable_expert"]:
            i["takeover_log_prob"] = log_prob.item()

        if self.config["use_render"]:  # and self.config["main_exp"]: #and not self.config["in_replay"]:
            self.render(
                mode="top_down",
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
        self.pending_human_traj = []
        self.pending_agent_traj = []
        for npp in self.drawn_points:
            npp.detachNode()
            self.drawer._dying_points.append(npp)
        self.drawn_points = []
        return o, info


if __name__ == "__main__":
    env = FakeHumanEnvPref(dict(free_level=0.95, use_render=True, manual_control=False, future_steps=15, stop_freq = 5))
    env.reset()
    while True:
        _, _, done, info = env.step([0, 1])
        # done = tm or tc
        #env.render(mode="topdown")
        if done:
            print(info)
            env.reset()
