import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
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
        use_render=True,  # Enable rendering for video recording
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
        buffer_size=1,
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
    """Extract RGB frame from observation (参考night_driving示例)"""
    if isinstance(obs, dict) and "image" in obs:
        img = obs["image"]
        
        # 根据参考代码，图像格式可能是 (H, W, 3, stack_size) 或 (H, W, C*stack_size)
        if len(img.shape) == 4:
            # 4D数组: (H, W, 3, stack_size) - 取最后一帧
            rgb_image = img[..., -1]  # 形状: (H, W, 3)
        elif len(img.shape) == 3:
            # 3D数组: 可能是 (H, W, C*stack_size) 或 (H, W, 3)
            if img.shape[2] == 9:  # 3 channels * 3 stack
                # 取最后3个通道（最新的RGB帧）
                rgb_image = img[:, :, -3:].copy()
            elif img.shape[2] == 3:
                rgb_image = img.copy()
            else:
                # Fallback: 取前3个通道
                rgb_image = img[:, :, :3].copy()
        else:
            return None
        
        # 确保是3通道RGB格式
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            return None
        
        # 如果图像是归一化的（0-1），转换为0-255
        if rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        else:
            rgb_image = rgb_image.astype(np.uint8)
        
        return rgb_image
    return None


def evaluate_with_video(model, env, num_episodes=5, save_video=False, video_save_dir="eval_videos"):
    """Evaluate model and optionally record videos, collecting all evaluation metrics"""
    if save_video:
        os.makedirs(video_save_dir, exist_ok=True)
    
    # Collect evaluation metrics (similar to EvalCallback)
    episode_rewards = []
    episode_lengths = []
    is_success_buffer = []
    evaluations_info_buffer = defaultdict(list)
    
    episode_count = 0
    
    for ep in range(num_episodes):
        print(f"\n===== Episode {ep + 1}/{num_episodes} =====")
        
        # Reset environment
        obs = env.reset()
        done = False
        step_count = 0
        episode_reward = 0.0
        
        # Prepare video writer only if save_video is True (参考night_driving示例)
        out = None
        h, w = 84, 84  # Default dimensions
        video_path = None
        should_save_video = save_video  # Use local variable
        if should_save_video:
            # 按照参考代码的方式提取图像
            if isinstance(obs, dict) and "image" in obs:
                rgb_image = obs["image"]
                # 处理4D数组: (H, W, 3, stack_size) - 取最后一帧
                if len(rgb_image.shape) == 4:
                    rgb_image = rgb_image[..., -1]  # 形状: (H, W, 3)
                elif len(rgb_image.shape) == 3:
                    if rgb_image.shape[2] == 9:  # 3 channels * 3 stack
                        rgb_image = rgb_image[:, :, -3:]  # 取最后3个通道
                    elif rgb_image.shape[2] != 3:
                        print(f"Warning: Unexpected image shape {rgb_image.shape}, skipping video")
                        should_save_video = False
                        rgb_image = None
                else:
                    print(f"Warning: Unexpected image shape {rgb_image.shape}, skipping video")
                    should_save_video = False
                    rgb_image = None
                
                if rgb_image is not None:
                    h, w = rgb_image.shape[:2]
                    # 如果图像是归一化的（0-1），转换为0-255
                    if rgb_image.max() <= 1.0:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    else:
                        rgb_image = rgb_image.astype(np.uint8)
            else:
                print("Warning: Could not extract frame from observation, skipping video")
                should_save_video = False
                rgb_image = None
            
            if should_save_video and rgb_image is not None:
                # 按照参考代码使用mp4v编码和.mp4格式
                video_path = os.path.join(video_save_dir, f"episode_{ep + 1}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))  # 使用30fps
                
                # Check if VideoWriter was initialized successfully
                if not out.isOpened():
                    print(f"Error: Failed to initialize video writer for {video_path}")
                    out = None
                    should_save_video = False
                else:
                    # 写入第一帧
                    out.write(rgb_image)
        
        while not done and step_count < 1500:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            step_count += 1
            episode_reward += reward
            
            # Collect evaluation info (similar to EvalCallback._log_success_callback)
            if done:
                # Check for success flags
                maybe_is_success = info.get("is_success")
                if maybe_is_success is not None:
                    is_success_buffer.append(maybe_is_success)
                
                maybe_is_success2 = info.get("arrive_dest", None)
                if maybe_is_success2 is not None:
                    is_success_buffer.append(maybe_is_success2)
                
                # Collect other metrics
                for k in ["episode_energy", "route_completion", "total_cost", "arrive_dest", 
                         "max_step", "out_of_road", "crash"]:
                    if k in info:
                        evaluations_info_buffer[k].append(info[k])
            
            # Collect raw_action if available
            if "raw_action" in info:
                evaluations_info_buffer["raw_action"].append(info["raw_action"])
            
            # Extract and save frame only if should_save_video is True (参考night_driving示例)
            if should_save_video and out is not None:
                if isinstance(obs, dict) and "image" in obs:
                    rgb_image = obs["image"]
                    # 处理4D数组: (H, W, 3, stack_size) - 取最后一帧
                    if len(rgb_image.shape) == 4:
                        rgb_image = rgb_image[..., -1]  # 形状: (H, W, 3)
                    elif len(rgb_image.shape) == 3:
                        if rgb_image.shape[2] == 9:  # 3 channels * 3 stack
                            rgb_image = rgb_image[:, :, -3:]  # 取最后3个通道
                        elif rgb_image.shape[2] != 3:
                            continue  # Skip this frame
                    
                    # 如果图像是归一化的（0-1），转换为0-255
                    if rgb_image.max() <= 1.0:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    else:
                        rgb_image = rgb_image.astype(np.uint8)
                    
                    # 确保尺寸正确
                    if rgb_image.shape[:2] != (h, w):
                        rgb_image = cv2.resize(rgb_image, (w, h))
                    
                    # 直接写入图像（不需要BGR转换）
                    out.write(rgb_image)
            
            if done:
                # Use Monitor wrapper info if available (similar to evaluate_policy)
                if "episode" in info:
                    episode_rewards.append(info["episode"]["r"])
                    episode_lengths.append(info["episode"]["l"])
                else:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(step_count)
                
                print(f"Episode finished at step {step_count}, reward: {episode_reward:.2f}")
        
        # Release video writer if it was created
        if should_save_video and out is not None:
            out.release()
            # Get the actual video path (might be .avi or .mp4 depending on codec)
            if video_path is None:
                if os.path.exists(os.path.join(video_save_dir, f"episode_{ep + 1}.avi")):
                    video_path = os.path.join(video_save_dir, f"episode_{ep + 1}.avi")
                else:
                    video_path = os.path.join(video_save_dir, f"episode_{ep + 1}.mp4")
            if video_path and os.path.exists(video_path):
                file_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
                print(f"Video saved to {video_path} ({file_size:.2f} MB)")
            else:
                print(f"Warning: Video file was not created successfully")
        
        episode_count += 1
    
    env.close()
    
    # Print evaluation statistics (similar to EvalCallback)
    print(f"\n{'='*60}")
    print(f"===== Evaluation Complete =====")
    print(f"{'='*60}")
    print(f"Completed {episode_count} episodes\n")
    
    # Print reward statistics
    if len(episode_rewards) > 0:
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print(f"Episode Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Print episode length statistics
    if len(episode_lengths) > 0:
        mean_ep_length = np.mean(episode_lengths)
        std_ep_length = np.std(episode_lengths)
        print(f"Episode Length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
    
    # Print success rate if available
    if len(is_success_buffer) > 0:
        success_rate = np.mean(is_success_buffer)
        print(f"Success Rate: {100 * success_rate:.2f}%")
    
    # Print other metrics
    if evaluations_info_buffer:
        print(f"\nAdditional Metrics:")
        for k, v in evaluations_info_buffer.items():
            if len(v) > 0:
                # Handle different data types
                if isinstance(v[0], (int, float, np.number)):
                    mean_val = np.mean(np.asarray(v))
                    print(f"  {k}: {mean_val:.4f}")
                else:
                    print(f"  {k}: {len(v)} entries")
    
    if save_video:
        print(f"\nVideos saved in: {video_save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/home/caihy/pvp/bestppomodeldomainA.zip", help="Path to model checkpoint")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to evaluate")
    parser.add_argument("--save_video", action="store_true", help="Save videos (default: False)")
    parser.add_argument("--video_dir", type=str, default="eval_videos", help="Directory to save videos")
    args = parser.parse_args()
    
    # Create environment (num_env=1, single environment)
    print("Creating evaluation environment...")
    eval_env = make_eval_env()
    
    # Load model
    print("Loading model...")
    model = load_model(args.ckpt, eval_env)
    
    # Evaluate and optionally record videos
    print("Starting evaluation...")
    evaluate_with_video(
        model, 
        eval_env, 
        num_episodes=args.num_episodes, 
        save_video=args.save_video,
        video_save_dir=args.video_dir
    )
