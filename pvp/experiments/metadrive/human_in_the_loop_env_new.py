import copy
import time
from collections import deque
import torch
import numpy as np
from metadrive.engine.core.onscreen_message import ScreenMessage
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.policy.manual_control_policy import TakeoverPolicyDelayWithoutBrake, TakeoverPolicyWithoutBrake
from metadrive.utils.math import safe_clip
import math
ScreenMessage.SCALE = 0.05

HUMAN_IN_THE_LOOP_ENV_CONFIG = {
    # Environment setting:
    "out_of_route_done": True,  # Raise done if out of route.
    "num_scenarios": 50,  # There are totally 50 possible maps.
    "start_seed": 100,  # We will use the map 100~150 as the default training environment.
    "traffic_density": 0.06,

    # Reward and cost setting:    "cost_to_reward": True,  # Cost will be negated and added to the reward. Useless in PVP.
    "cos_similarity": False,  # If True, the takeover cost will be the cos sim between a_h and a_n. Useless in PVP.

    # Set up the control device. Default to use keyboard with the pop-up interface.
    "manual_control": True,
    "agent_policy": TakeoverPolicyDelayWithoutBrake,
    "controller": "keyboard",  # Selected from [keyboard, xbox, steering_wheel].
    "only_takeover_start_cost": False,  # If True, only return a cost when takeover starts. Useless in PVP.

    # Visualization
    "vehicle_config": {
        "show_dest_mark": True,  # Show the destination in a cube.
        "show_line_to_dest": True,  # Show the line to the destination.
        "show_line_to_navi_mark": True,  # Show the line to next navigation checkpoint.
    },
    "horizon": 1500,
    "use_discrete": False,
    "future_steps": 20,
    "takeover_see": 20,
    "stop_freq": 10,
    "takeover_delay": 10,
}

class HumanInTheLoopEnv(SafeMetaDriveEnv):
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
    pending_agent_traj = []
    pending_human_traj = []
    start_time = time.time()
    warn = False
    expert = None
    drawn_points = []
    delay_draw = 0
    last_takeover = False
    def default_config(self):
        config = super(HumanInTheLoopEnv, self).default_config()
        config.update(HUMAN_IN_THE_LOOP_ENV_CONFIG, allow_add_new_key=True)
        return config

    def reset(self, *args, **kwargs):
        # if self.expert is None:
        #         global _expert
        #         self.expert = _expert
        future_steps = self.config["future_steps"]
        takeover_delay = self.config["takeover_delay"]
        
        self.takeover = False
        self.warn = False
        self.agent_action = None
        obs, info = super(HumanInTheLoopEnv, self).reset(*args, **kwargs)
        shared_control_policy = self.engine.get_policy(self.agent.id)
        shared_control_policy.delay_upperbound = takeover_delay
        shared_control_policy.delay = takeover_delay
        
        self.last_obs = obs
        if hasattr(self, "model"):
            for step in range(len(self.pending_agent_traj)):
                if len(self.pending_agent_traj[step]) > 0 and hasattr(self.model, "prefreplay_buffer"):
                    self.model.prefreplay_buffer.add(self.pending_human_traj[step], self.pending_agent_traj[step])
        if hasattr(self,"drawer"):
                drawer = self.drawer # create a point drawer
        else:
                self.drawer = self.engine.make_point_drawer(scale=3)
                
        
        self.pending_human_traj = []
        self.pending_agent_traj = []
        for npp in self.drawn_points:
                    npp.detachNode()
                    self.drawer._dying_points.append(npp)
        self.drawn_points = []
        self.delay_draw = 0
        self.last_takeover = False
        # The training code is for older version of gym, so we discard the additional info from the reset.
        return obs

    def _get_step_return(self, actions, engine_info):
        """Compute takeover cost here."""
        # if self.takeover:
        #     print("B")
        o, r, tm, tc, engine_info = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info)
        self.last_obs = o
        d = tm or tc

        shared_control_policy = self.engine.get_policy(self.agent.id)
        
        try:
            engine_info["raw_action"] = copy.deepcopy(shared_control_policy.b_action)
        except:
            print("First step in ep")
        
        self.b_action = copy.deepcopy(engine_info["raw_action"])
        
        last_t = self.takeover
        self.takeover = shared_control_policy.takeover if hasattr(shared_control_policy, "takeover") else False
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
        engine_info["total_cost"] = self.total_cost
        # engine_info["total_cost_so_far"] = self.total_cost
        return o, r, d, engine_info

    def _is_out_of_road(self, vehicle):
        """Out of road condition"""
        ret = (not vehicle.on_lane) or vehicle.crash_sidewalk
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        return ret

    def step(self, actions):
        """Add additional information to the interface."""
        self.agent_action = copy.copy(actions)
        # if self.takeover:
        #     print("A")
        
        future_steps = self.config["future_steps"]
        stop_freq = self.config["stop_freq"]
        
        predicted_traj_exp = []
        
        if self.takeover:
            if not self.last_takeover:
                self.delay_draw = 0
            if self.delay_draw % stop_freq == 0:
                predicted_traj_exp, acprob_exp, total_reward_exp, total_advantage_exp = self._predict_agent_future_trajectory(self.last_obs, future_steps, use_exp=self.b_action)
                predicted_traj, acprob, total_reward, total_advantage, all_states = self._predict_agent_future_trajectory(self.last_obs, future_steps, return_all_states=True)
                if len(predicted_traj) <= 10:
                    shared_control_policy = self.engine.get_policy(self.agent.id)
                    shared_control_policy.delay = 0
            else:
                predicted_traj = []
            self.warn = True
        elif self.delay_draw % stop_freq == 0:
            predicted_traj, acprob, total_reward, total_advantage = self._predict_agent_future_trajectory(self.last_obs, future_steps)
            #TODO: better warn signals
            if len(predicted_traj) < future_steps and total_reward < 0:
                self.warn = True
            else:
                self.warn = False
        else:
            predicted_traj = []
        
        self.delay_draw += 1
        
        self.human_traj = []
        
        if len(predicted_traj) > 0:
            drawer = self.drawer 
            
            
            for npp in self.drawn_points:
                    npp.detachNode()
                    self.drawer._dying_points.append(npp)
            self.drawn_points = []
            #time.sleep(0.5)
            points, colors = [], []
            for j in range(0, len(predicted_traj), 1):
                points.append((predicted_traj[j]["next_pos"][0], predicted_traj[j]["next_pos"][1], 0.5)) # define line 1 for test
                color=(int(self.warn), 1 - int(self.warn), 0)
                colors.append(np.clip(np.array([*color,1]), 0., 1.0))
            
            #drawer.draw_points(points, colors)
            #self.drawn_points = self.drawn_points + drawer.draw_points(points, colors)
            if len(predicted_traj_exp) > 0:
                #points, colors = [], []
                
                for j in range(0, len(predicted_traj_exp), 1):
                    points.append((predicted_traj_exp[j]["next_pos"][0], predicted_traj_exp[j]["next_pos"][1], 0.5)) # define line 1 for test
                    color=(0, 0, 1)
                    colors.append(np.clip(np.array([*color,1]), 0., 1.0))
                    
                cur_state = self.get_state()
                for st in all_states:
                    self.set_state(st)
                    predicted_traj_exp_2, acprob_exp, total_reward_exp, total_advantage_exp = self._predict_agent_future_trajectory(self.last_obs, future_steps, use_exp=self.b_action)
                    for j in range(0, len(predicted_traj_exp_2), 1):
                        points.append((predicted_traj_exp_2[j]["next_pos"][0], predicted_traj_exp_2[j]["next_pos"][1], 0.5)) # define line 1 for test
                        color=(0, 0, 0.5)
                        colors.append(np.clip(np.array([*color,1]), 0., 1.0))
                
                self.set_state(cur_state)
                    
            self.drawn_points = self.drawn_points + drawer.draw_points(points, colors)
        if self.takeover:
            
            self.pending_agent_traj.append(predicted_traj)
        
        last_o = self.last_obs.copy()
        if not hasattr(self, "b_action"):
            self.b_action = actions
        last_b_action = self.b_action
        last_takeover = self.takeover
        self.last_takeover = last_takeover
        self.vehicle.real = True
        
        ret = super(HumanInTheLoopEnv, self).step(actions)
        
        if last_takeover:
            o, r, d = ret[0], ret[1], ret[2]
            self.pending_human_traj.append(self.human_traj)
            for lst in self.pending_human_traj:
                if len(lst) < future_steps:
                    lst.append({
                        "obs": last_o.copy(),
                        "action": last_b_action.copy(),
                        "next_obs": o.copy(),
                        "reward": r,
                        "done": d,
                        "next_pos": copy.deepcopy(self.vehicle.position),
                        "action_exp": last_b_action.copy(),
                        "action_nov": self.agent_action.copy(),
                    })
        else:
            assert len(self.pending_agent_traj) == len(self.pending_human_traj)
            for step in range(len(self.pending_agent_traj)):
                    if len(self.pending_agent_traj[step]) > 0 and hasattr(self, "model") and hasattr(self.model, "prefreplay_buffer"):
                        self.model.prefreplay_buffer.add(self.pending_human_traj[step], self.pending_agent_traj[step])
            self.pending_agent_traj = []
            self.pending_human_traj = []
        while self.in_pause:
            self.engine.taskMgr.step()
        self.vehicle.real = False
        
        self.takeover_recorder.append(self.takeover)
        if self.config["use_render"]:  # and self.config["main_exp"]: #and not self.config["in_replay"]:
            super(HumanInTheLoopEnv, self).render(
                text={
                    "Total Cost": round(self.total_cost, 2),
                    "Takeover Cost": round(self.total_takeover_cost, 2),
                    "Takeover": "TAKEOVER" if self.takeover else "NO",
                    "Total Step": self.total_steps,
                    "Total Time": time.strftime("%M:%S", time.gmtime(time.time() - self.start_time)),
                    "Takeover Rate": "{:.2f}%".format(np.mean(np.array(self.takeover_recorder) * 100)),
                    "Pause": "Press E",
                }
            )

        self.total_steps += 1

        self.total_takeover_count += 1 if self.takeover else 0
        ret[-1]["total_takeover_count"] = self.total_takeover_count

        return ret

    def stop(self):
        """Toggle pause."""
        self.in_pause = not self.in_pause

    def setup_engine(self):
        """Introduce additional key 'e' to the interface."""
        super(HumanInTheLoopEnv, self).setup_engine()
        self.engine.accept("e", self.stop)

    def get_takeover_cost(self, info):
        """Return the takeover cost when intervened."""
        if not self.config["cos_similarity"]:
            return 1
        takeover_action = safe_clip(np.array(info["raw_action"]), -1, 1)
        agent_action = safe_clip(np.array(self.agent_action), -1, 1)
        multiplier = (agent_action[0] * takeover_action[0] + agent_action[1] * takeover_action[1])
        divident = np.linalg.norm(takeover_action) * np.linalg.norm(agent_action)
        if divident < 1e-6:
            cos_dist = 1.0
        else:
            cos_dist = multiplier / divident
        return 1 - cos_dist
    
    def get_state(self) -> dict:
        import copy
        state = copy.deepcopy(self.vehicle.get_state())
        return copy.deepcopy(state)

    def set_state(self, state: dict):
        self.vehicle.set_state(state)
    def _predict_agent_future_trajectory(self, current_obs, n_steps, use_exp = None, return_all_states = False):
        all_states = []
        saved_state = self.get_state()
        traj = []
        obs = current_obs
        lstprob = []
        total_reward = 0
        
        total_advantage = 0
        for step in range(n_steps):
            if use_exp is None:
                if hasattr(self, "model"):
                    action, _ = self.model.policy.predict(obs, deterministic=True)
                else:
                    action = self.agent_action
            else:
                action  = use_exp
            if self.config["use_discrete"]:
                action_cont = self.discrete_to_continuous(action)
            else:
                action_cont = action

            #o, r, d, i = super(HumanInTheLoopEnv, self).step(action_cont)
            
            self.engine.notrender = True
            
            actions = self._preprocess_actions(action_cont)  # preprocess environment input
            # engine_info = self._step_simulator(actions)  # step the simulation
            # while self.in_stop:
            #     self.engine.taskMgr.step()  # pause simulation
            # o, r, tm, tc, i = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info=engine_info)
            # 假设 dt 已知
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
            
            if return_all_states:
                all_states.append(self.get_state())
            #if step > 0:
            total_reward += r
            d = self.done_function('default_agent')[0]

            new_obs = self.get_single_observation().observe(self.vehicle)
            traj.append({
                "obs": obs.copy(),
                "action": action_cont.copy(),
                "reward": r,
                "next_obs": new_obs.copy(),
                "done": d,
                "next_pos": copy.deepcopy(self.vehicle.position),
                "action_exp": action_cont.copy(),
                "action_nov": action_cont.copy(),
                "values_n": 0 #values_n.item(),
            })
            obs = new_obs.copy()
            traj[-1]["advantage"] = r #+ 0.99 * values_n.item() - values_next.item()
            
            total_advantage += r #+ 0.99 * values_n.item() - values_next.item()
            if d:
                if r < 0:
                    total_reward = -100
                break
        self.set_state(saved_state)
        from pvp.sb3.common.utils import safe_mean
        #if total_reward > 0:
        #    total_reward += values_n.item()
        if return_all_states:
            return traj, safe_mean(lstprob[:self.config["takeover_see"]]), total_reward, total_advantage, all_states
        return traj, safe_mean(lstprob[:self.config["takeover_see"]]), total_reward, total_advantage

if __name__ == "__main__":
    env = HumanInTheLoopEnv({
        "manual_control": True,
        "use_render": True,
        "controller": "gamepad",
        "takeover_delay": 0,
    })
    env.reset()
    while True:
        _, _, done, _ = env.step([0, 1])
        if done:
            env.reset()
