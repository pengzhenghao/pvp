from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.utils import Config
from metadrive.utils.math import norm
from panda3d.core import LVector3
import math
import torch
from metadrive.utils.coordinates_shift import panda_vector, metadrive_vector, panda_heading
from collections import deque
import numpy as np
import copy

class BasePredictionEnv(SafeMetaDriveEnv):
    def default_config(self) -> Config:
        config = super(BasePredictionEnv, self).default_config()
        config.update(
            {
                "future_steps_predict": 20,
                "update_future_freq": 10,
                "future_steps_preference": 3,
                "expert_noise": 0,
            },
            allow_add_new_key=True
        )
        return config

    def get_state(self) -> dict:
        import copy
        state = copy.deepcopy(self.vehicle.get_state())
        return copy.deepcopy(state)
        
    def set_state(self, state: dict):
        self.vehicle.set_state(state)
    
    def predict_agent_future_trajectory(self, current_obs, n_steps, action_behavior = None, return_all_states = False):
        info = dict()
        saved_state = self.get_state()
        
        all_states = []
        traj = []
        obs = current_obs
        total_reward = 0
        failure = False
        
        for step in range(n_steps):
            old_pos = copy.deepcopy(self.vehicle.position)
            action = action_behavior
            if action_behavior is None:
                action = self.agent_action
                if hasattr(self, "model"):
                     action, _ = self.model.policy.predict(obs, deterministic=True)
            
            if self.config["use_discrete"]:
                action = self.discrete_to_continuous(action)

            #actions = self._preprocess_actions(action) 
            dt = self.config["physics_world_step_size"] * self.config["decision_repeat"]
            self.vehicle.before_step(action)
                
            params = self.vehicle.get_dynamics_parameters()
            mass = params["mass"]
            max_engine_force = params["max_engine_force"]
            max_brake_force = params["max_brake_force"]

            throttle = self.vehicle.throttle_brake
            if throttle >= 0:
                if self.vehicle.speed >= self.vehicle.max_speed_m_s:
                    a = 0.0
                else:
                    engine_force = max_engine_force * throttle
                    a = engine_force / mass * 4
            else:
                brake_force = max_brake_force * abs(throttle)
                a = -brake_force / mass * 4

            new_speed = self.vehicle.speed + a * dt
            new_speed = max(new_speed, 0.0)

            step_info = self.vehicle.after_step()
            current_steering = self.vehicle.steering
            max_steering_rad = math.radians(self.vehicle.config["max_steering"])

            L = self.vehicle.FRONT_WHEELBASE + self.vehicle.REAR_WHEELBASE
            new_heading = self.vehicle.heading_theta + (new_speed / L) * math.tan(current_steering * max_steering_rad) * dt

            new_x = self.vehicle.position[0] + new_speed * dt * math.cos(new_heading)
            new_y = self.vehicle.position[1] + new_speed * dt * math.sin(new_heading)
            new_position = [new_x, new_y]
            new_velocity = [new_speed * math.cos(new_heading), new_speed * math.sin(new_heading)]

            self.vehicle.set_position(new_position)
            self.vehicle.set_heading_theta(new_heading)
            self.vehicle.set_velocity(new_velocity)
            self.vehicle.navigation.update_localization(self.vehicle)
            r = self.reward_function('default_agent')[0]
            total_reward += r
            
            if return_all_states:
                all_states.append(self.get_state())
            d = self.done_function('default_agent')[0]

            new_obs = self.get_single_observation().observe(self.vehicle)
            
            traj.append({
                "obs": obs.copy(),
                "action": action.copy(),
                "reward": r,
                "next_obs": new_obs.copy(),
                "done": d,
                "pos": old_pos,
                "next_pos": copy.deepcopy(self.vehicle.position),
            })
            obs = new_obs.copy()
            
            if d:
                failure = (r < 0)
                break
        
        self.set_state(saved_state)
        
        failure = failure or (total_reward <= 10) #CHY: Failure if too slow.
        info["all_states"] = all_states
        info["failure"] = failure
        info["total_reward"] = total_reward
        
        return traj, info
    

    def draw_points(self, points, colors=None):
        """
        Draw a set of points with colors
        Args:
            points: a set of 3D points
            colors: a list of color for each point

        Returns: None

        """
        from panda3d.core import VBase4, NodePath, Material
        from panda3d.core import LVecBase4f
        from metadrive.engine.asset_loader import AssetLoader
        drawer = self.drawer
        new_points = []
        for k, point in enumerate(points):
            if len(drawer._dying_points) > 0:
                np = drawer._dying_points.pop()
            else:
                np = NodePath("debug_point")
                model = drawer.engine.loader.loadModel(AssetLoader.file_path("models", "sphere.egg"))
                model.setScale(drawer.scale)
                model.reparentTo(np)
            material = Material()
            if colors:
                material.setBaseColor(LVecBase4f(*colors[k]))
            else:
                material.setBaseColor(LVecBase4f(1, 1, 1, 1))
            material.setShininess(64)
            # material.setEmission((1, 1, 1, 1))
            np.setMaterial(material, True)
            np.setPos(*point)
            np.reparentTo(drawer)
            drawer._existing_points.append(np)
            new_points.append(np)
        return new_points

    def _get_reset_return(self, reset_info):
        o, info = super(BasePredictionEnv, self)._get_reset_return(reset_info)
        if hasattr(self,"drawer"):
            for npp in self.drawn_points:
                npp.detachNode()
                self.drawer._dying_points.append(npp)
            self.drawn_points = []
        else:
            self.drawn_points = []
            self.drawer = self.engine.make_point_drawer(scale=3)
        return o, info