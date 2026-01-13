import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import gymnasium
sys.modules['gym'] = gymnasium

from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
from pvp.pvp_td3 import PVPTD3
from pvp.sb3.td3.policies import TD3Policy
from pvp.sb3.sac.our_features_extractor import OurFeaturesExtractorCNN as OurFeaturesExtractor
from pvp.sb3.common.monitor import Monitor
from metadrive.component.sensors.rgb_camera import RGBCamera


def make_eval_env():
    """Create evaluation environment with rgb_camera and rendering enabled"""
    from metadrive.component.sensors.rgb_camera import RGBCamera
    sensor_size = (84, 84)
    eval_env_config = dict(
        use_render=False,  # Enable rendering for video recording
        manual_control=False,  # Allow receiving control signal from external device
        start_seed=1000,
        horizon=1500,
        image_observation=True, 
        vehicle_config=dict(image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, *sensor_size)},
        stack_size=3,
        interface_panel=["rgb_camera", "dashboard"],
        daytime="08:30",
    )
    eval_env = HumanInTheLoopEnv(config=eval_env_config)
    # Note: Monitor is optional for evaluation, but we can add it if needed
    # eval_env = Monitor(env=eval_env, filename=None)
    return eval_env


def load_model(ckpt_path, env):
    """Load trained model from checkpoint"""
    from pvp.sb3.common.save_util import load_from_zip_file
    
    # Setup algorithm config (should match training config)
    algo_config = dict(
        policy=TD3Policy,
        replay_buffer_class=None,  # Not needed for evaluation
        policy_kwargs=dict(
            features_extractor_class=OurFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=275),
            share_features_extractor=False,
            net_arch=[256],
        ),
        env=env,
        learning_rate=1e-4,
        q_value_bound=1,
        buffer_size=100000,
        learning_starts=0,
        batch_size=1024,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        action_noise=None,
        tensorboard_log=None,
        create_eval_env=False,
        verbose=2,
        seed=0,
        device="auto",
    )
    
    model = PVPTD3(**algo_config)
    
    print(f"Loading checkpoint from {ckpt_path}!")
    data, params, pytorch_variables = load_from_zip_file(ckpt_path, device=model.device, print_system_info=False)
    model.set_parameters(params, exact_match=False, device=model.device)
    print(f"Model loaded successfully!")
    
    return model


def extract_frame_from_obs(obs):
    """Extract RGB frame from observation"""
    if isinstance(obs, dict) and "image" in obs:
        img = obs["image"]
        if len(img.shape) == 3:
            # Handle stacked images: shape is [H, W, C*stack_size]
            # For stack_size=3, channels are RGBRGBRGB (9 channels)
            if img.shape[2] == 9:  # 3 channels * 3 stack
                # Take the last RGB channels (most recent frame)
                frame = img[:, :, -3:].copy()
            elif img.shape[2] == 3:
                frame = img.copy()
            else:
                # Fallback: take first 3 channels
                frame = img[:, :, :3].copy()
            
            # Convert from [0, 1] to [0, 255] if needed
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
            
            return frame
    return None


def evaluate_with_video(model, env, num_episodes=5, video_save_dir="eval_videos"):
    """Evaluate model and record videos"""
    os.makedirs(video_save_dir, exist_ok=True)
    
    episode_count = 0
    
    for ep in range(num_episodes):
        print(f"\n===== Episode {ep + 1}/{num_episodes} =====")
        
        # Reset environment
        obs = env.reset()
        done = False
        step_count = 0
        
        # Prepare video writer - get dimensions from first frame
        first_frame = extract_frame_from_obs(obs)
        if first_frame is None:
            print("Warning: Could not extract frame from observation, using default size")
            h, w = 84, 84
        else:
            h, w = first_frame.shape[:2]
        
        video_path = os.path.join(video_save_dir, f"episode_{ep + 1}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
        
        # Write first frame
        if first_frame is not None:
            frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        while not done and step_count < 1500:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            step_count += 1
            
            # Extract frame from observation
            frame = extract_frame_from_obs(obs)
            if frame is not None:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            # Render (this will show the interface panel)
            env.render()
            
            if done:
                print(f"Episode finished at step {step_count}")
                if "episode" in info:
                    print(f"Episode info: {info.get('episode', {})}")
        
        # Release video writer
        out.release()
        print(f"Video saved to {video_path}")
        
        episode_count += 1
    
    env.close()
    print(f"\n===== Evaluation Complete =====")
    print(f"Recorded {episode_count} episodes")
    print(f"Videos saved in: {video_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/home/caihy/pvp/rlforrgb.zip", help="Path to model checkpoint")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to evaluate")
    parser.add_argument("--video_dir", type=str, default="eval_videos", help="Directory to save videos")
    args = parser.parse_args()
    
    # Create environment (num_env=1, single environment)
    print("Creating evaluation environment...")
    eval_env = make_eval_env()
    
    # Load model
    print("Loading model...")
    model = load_model(args.ckpt, eval_env)
    
    # Evaluate and record videos
    print("Starting evaluation...")
    evaluate_with_video(model, eval_env, num_episodes=args.num_episodes, video_save_dir=args.video_dir)
