#goal: add fake (imag) pos samples to neg trajs. see performance of algos
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

    train_env = HumanInTheLoopEnv(config={'manual_control': False, "use_render": False})

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


class FakeHumanEnv(HumanInTheLoopEnv):
    last_takeover = None
    last_obs = None
    expert = None
    pending_human_traj = []
    pending_agent_traj = []
    etakeover = False
    from collections import deque 
    advantages = deque(maxlen = 200)
    drawn_points = []
    delaytake = 0
    
    def __init__(self, config):
        super(FakeHumanEnv, self).__init__(config)
        if self.config["use_discrete"]:
            self._num_bins = 13
            self._grid = np.linspace(-1, 1, self._num_bins)
            self._actions = np.array(np.meshgrid(self._grid, self._grid)).T.reshape(-1, 2)

    @property
    def action_space(self) -> gym.Space:
        if self.config["use_discrete"]:
            return gym.spaces.Discrete(self._num_bins ** 2)
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
                "free_level": 0.9,
                "manual_control": False,
                "use_render": False,
                "expert_deterministic": False,
                "future_steps": 20,
                "takeover_see": 20,
                "stop_freq": 10,
                "takeover_delay": 0,
                "img_future_steps": 1,
                "stop_img_samples": 3, 
                "future_steps_predict": -1,
                "update_future_freq": -1,
                "future_steps_preference": -1,
                "expert_noise": -1,
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
    def get_state(self) -> dict:
        import copy
        state = copy.deepcopy(self.vehicle.get_state())
        return copy.deepcopy(state)

    def set_state(self, state: dict):
        self.vehicle.set_state(state)

    def step(self, actions):
        """Compared to the original one, we call expert_action_prob here and implement a takeover function."""
        actions = np.asarray(actions).astype(np.float32)

        if self.config["use_discrete"]:
            actions = self.discrete_to_continuous(actions)

        self.agent_action = copy.copy(actions)
        self.b_action = copy.copy(actions)
        self.last_takeover = self.takeover
        
        future_steps = self.config["future_steps"]
        stop_freq = self.config["stop_freq"]
        
        img_future_steps = self.config["img_future_steps"]
        stop_img_samples = self.config["stop_img_samples"]
        
        self.human_traj = []
        if self.expert is None:
                global _expert
                self.expert = _expert
        
        if True:
            last_obs, _ = self.expert.obs_to_tensor(self.last_obs)
            distribution = self.expert.get_distribution(last_obs)
            log_prob = distribution.log_prob(torch.from_numpy(actions).to(last_obs.device))
            action_prob = log_prob.exp().detach().cpu().numpy()

            if self.config["expert_deterministic"]:
                expert_action = distribution.mode().detach().cpu().numpy()
            else:
                expert_action = distribution.sample().detach().cpu().numpy()

            assert expert_action.shape[0] == action_prob.shape[0] == 1
            action_prob = action_prob[0]
            expert_action, _  = self.expert.predict(self.last_obs, deterministic=True)
            enoise = np.random.randn(2) * 0
            
            expert_action_clip = np.clip(enoise + expert_action, self.action_space.low, self.action_space.high)
        
        if (self.total_steps % stop_freq == 0) or self.last_takeover:
            predicted_traj, info2 = self.predict_agent_future_trajectory(self.last_obs, future_steps, action_behavior=self.agent_action.copy())
            total_reward = info2["total_reward"]
            
            predicted_traj_real, info3 = self.predict_agent_future_trajectory(self.last_obs, future_steps)
            total_reward_real = info3["total_reward"]
            
            if total_reward_real < 0:
                self.delaytake = 0
                self.etakeover = True
            else:
                if (self.total_steps % stop_freq == 0):
                    self.etakeover = False
        else:
            predicted_traj = []
            
        etakeover = self.etakeover
        
        # ===== Get expert action and determine whether to take over! =====
        points, colors = [], []
        if self.config["disable_expert"]:
            pass

        else:

            if etakeover:
                if self.config["use_discrete"]:
                    expert_action = self.continuous_to_discrete(expert_action)
                    expert_action = self.discrete_to_continuous(expert_action)
                actions = expert_action
                self.b_action = copy.deepcopy(expert_action)
                self.takeover = True
                
                drawer = self.drawer 
                if True:
                    
                    for sti in range(min(len(predicted_traj) - 1, stop_img_samples)):
                        # predicted_traj_exp_2, acprob_exp, total_reward_exp, total_advantage_exp = self._predict_agent_future_trajectory(predicted_traj[sti]["obs"], img_future_steps, use_exp=self.b_action)
                        dic = {
                            "obs": predicted_traj[sti]["obs"].copy(),
                            "action": self.b_action.copy(),
                            "next_obs": predicted_traj[sti]["obs"].copy(),
                            "done": False,
                        }
                        predicted_traj_exp_2 = [dic].copy()
                        if hasattr(self, "model") and hasattr(self.model, "imagreplay_buffer"):
                            self.model.imagreplay_buffer.add(predicted_traj_exp_2, predicted_traj[sti+1:])
            else:
                self.takeover = False
            
        self.vehicle.real = True
        last_o = self.last_obs.copy()
        o, r, d, i = super(HumanInTheLoopEnv, self).step(actions)
        
        
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
        
        self.vehicle.real = False
        position, velocity, speed, heading = copy.copy(self.vehicle.position), copy.copy(self.vehicle.velocity), copy.copy(self.vehicle.speed), copy.copy(self.vehicle.heading_theta)
        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1

        if not self.config["disable_expert"]:
            i["takeover_log_prob"] = log_prob.item()

        if self.config["use_render"]:  # and self.config["main_exp"]: #and not self.config["in_replay"]:
            self.render(
                # mode="top_down",
                text={
                    "Total Cost": round(self.total_cost, 2),
                    "Takeover Cost": round(self.total_takeover_cost, 2),
                    "Takeover": "TAKEOVER" if self.takeover else "NO",
                    "Total Step": self.total_steps,
                    # "Total Time": time.strftime("%M:%S", time.gmtime(time.time() - self.start_time)),
                    "Takeover Rate": "{:.2f}%".format(np.mean(np.array(self.takeover_recorder) * 100)),
                    "Pause": "Press E",
                    "advantage": self.advantage,
                    "rewardtogo": self.total_r,
                }
            )

        assert i["takeover"] == self.takeover

        i["raw_action"] = copy.deepcopy(self.b_action)
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
        if hasattr(self,"drawer"):
                drawer = self.drawer # create a point drawer
        else:
                self.drawer = self.engine.make_point_drawer(scale=3)
                
        self.last_obs = o
        self.last_takeover = False
        self.etakeover = False
        if hasattr(self, "model"):
            for step in range(len(self.pending_agent_traj)):
                if len(self.pending_agent_traj[step]) > 0 and hasattr(self.model, "prefreplay_buffer"):
                    self.model.prefreplay_buffer.add(self.pending_human_traj[step], self.pending_agent_traj[step])
                
        self.pending_human_traj = []
        self.pending_agent_traj = []
        for npp in self.drawn_points:
            npp.detachNode()
            self.drawer._dying_points.append(npp)
        self.drawn_points = []
        return o, info
    def _predict_agent_future_trajectory(self, current_obs, n_steps, use_exp = None,  return_all_states = False, realmode= False):
        all_states = []
        saved_state = self.get_state()
        traj = []
        obs = current_obs
        lstprob = []
        total_reward = 0
        
        total_advantage = 0
        for step in range(n_steps):
            old_pos = copy.deepcopy(self.vehicle.position)
            if use_exp is None:
                action = self.agent_action
                if realmode and hasattr(self, "model"):
                     action, _ = self.model.policy.predict(obs, deterministic=True)
            else:
                action  = use_exp
            if self.config["use_discrete"]:
                action_cont = self.discrete_to_continuous(action)
            else:
                action_cont = action

            self.engine.notrender = True
            
            actions = self._preprocess_actions(action_cont) 
            r = 0
            for rep in range(1):
                dt = self.config["physics_world_step_size"] * self.config["decision_repeat"]

                # 从 after_step 中读取更新后的物理量
                self.vehicle.before_step(action_cont)
                
                params = self.vehicle.get_dynamics_parameters()
                mass = params["mass"]
                max_engine_force = params["max_engine_force"]
                max_brake_force = params["max_brake_force"]

                # 当前控制信号（假设 throttle_brake 范围在 [-1, 1]）
                throttle = self.vehicle.throttle_brake
                if throttle >= 0:
                    # 如果车速超过最大速度，则不再施加正向加速
                    if self.vehicle.speed >= self.vehicle.max_speed_m_s:
                        a = 0.0
                    else:
                        engine_force = max_engine_force * throttle
                        a = engine_force / mass * 4
                else:
                    brake_force = max_brake_force * abs(throttle)
                    a = -brake_force / mass * 4

                # 更新速度：采用简单的欧拉积分
                new_speed = self.vehicle.speed + a * dt
                # 保证速度不为负
                new_speed = max(new_speed, 0.0)

                step_info = self.vehicle.after_step()
                #new_speed = step_info["velocity"]        # 车速（单位：m/s）
                current_steering = self.vehicle.steering   
                max_steering_rad = math.radians(self.vehicle.config["max_steering"])  # 如果配置是度

                # 使用车辆动力学公式更新 heading（轴距 L 根据你的模型设定）
                L = self.vehicle.FRONT_WHEELBASE + self.vehicle.REAR_WHEELBASE  # 轴距，需替换为实际值
                new_heading = self.vehicle.heading_theta + (new_speed / L) * math.tan(current_steering * max_steering_rad) * dt

                # 更新位置（假设 self.vehicle.position 是一个 2D 数组或列表）
                new_x = self.vehicle.position[0] + new_speed * dt * math.cos(new_heading)
                new_y = self.vehicle.position[1] + new_speed * dt * math.sin(new_heading)
                new_position = [new_x, new_y]
                new_velocity = [new_speed * math.cos(new_heading), new_speed * math.sin(new_heading)]

                # 然后将这些状态更新回车辆
                self.vehicle.set_position(new_position)
                self.vehicle.set_heading_theta(new_heading)
                self.vehicle.set_velocity(new_velocity)
                self.vehicle.navigation.update_localization(self.vehicle)
                r += self.reward_function('default_agent')[0]
            # print("pred", self.vehicle.position)
            del self.engine.notrender
            #if step > 0:
            total_reward += r
            
            if return_all_states:
                all_states.append(self.get_state())
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
            
            new_obs = self.get_single_observation().observe(self.vehicle)
            
            traj.append({
                "obs": obs.copy(),
                "action": action_cont.copy(),
                "reward": r,
                "next_obs": new_obs.copy(),
                "done": d,
                "pos": old_pos,
                "next_pos": copy.deepcopy(self.vehicle.position),
                "action_exp": expert_action_clip.copy(),
                "action_nov": action_cont.copy(),
                "values_n": values_n.item(),
            })
            obs = new_obs.copy()
            
            actions_n, values_next, log_prob_n = self.expert(torch.Tensor(obs).to(self.expert.device).unsqueeze(0))
            traj[-1]["advantage"] = r + 0.99 * values_n.item() - values_next.item()
            
            total_advantage += r + 0.99 * values_n.item() - values_next.item()
            if d:
                if r < 0:
                    total_reward = -100
                break
        self.set_state(saved_state)
        from pvp.sb3.common.utils import safe_mean
        if total_reward > 10:
            #total_reward += values_n.item()
            pass
        else:
            total_reward = -100
        if return_all_states:
            return traj, safe_mean(lstprob[:self.config["takeover_see"]]), total_reward, total_advantage, all_states
        
        return traj, safe_mean(lstprob[:self.config["takeover_see"]]), total_reward, total_advantage
    

if __name__ == "__main__":
    env = FakeHumanEnv(dict(free_level=0.95, use_render=True, manual_control=False, future_steps=15, stop_freq = 5))
    env.reset()
    while True:
        _, _, done, info = env.step([0, 0.1])
        # done = tm or tc
        #env.render(mode="topdown")
        if done:
            print(info)
            env.reset()