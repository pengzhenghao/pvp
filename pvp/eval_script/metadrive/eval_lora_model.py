"""
Evaluation and Visualization script for LoRA fine-tuned models.

This script handles:
1. Loading LoRA models correctly (applying LoRA structure before loading weights)
2. Evaluating the model on the test environment
3. Visualizing CNN attention using Grad-CAM
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium
sys.modules['gym'] = gymnasium

from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
from pvp.sb3.td3.td3 import TD3
from pvp.sb3.td3.policies import TD3Policy
from pvp.sb3.sac.our_features_extractor import OurFeaturesExtractorCNN as OurFeaturesExtractor
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.lora import (
    apply_lora_to_model,
    get_lora_parameters,
    freeze_non_lora_parameters,
    print_trainable_parameters
)
from metadrive.component.sensors.rgb_camera import RGBCamera


# ============================================================================
# Environment Setup
# ============================================================================

def make_eval_env(use_render=True, start_seed=1000):
    """Create evaluation environment with rgb_camera"""
    sensor_size = (84, 84)
    eval_env_config = dict(
        use_render=use_render,
        manual_control=False,
        start_seed=start_seed,
        horizon=1500,
        image_observation=True,
        vehicle_config=dict(image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, *sensor_size)},
        stack_size=3,
        interface_panel=["rgb_camera", "dashboard"],
        daytime="06:10",
    )
    eval_env = HumanInTheLoopEnv(config=eval_env_config)
    return eval_env


# ============================================================================
# LoRA Model Loading
# ============================================================================

def load_lora_model(ckpt_path, env, lora_rank=4, lora_alpha=1.0, device="auto"):
    """
    Load a LoRA fine-tuned model correctly.
    
    The key is to:
    1. Create the base model with the same architecture
    2. Apply LoRA to match the saved model's structure
    3. Load the saved parameters (including LoRA weights)
    
    Args:
        ckpt_path: Path to the saved model checkpoint
        env: Environment for the model
        lora_rank: LoRA rank used during training
        lora_alpha: LoRA alpha used during training
        device: Device to load model on
    
    Returns:
        Loaded model with LoRA weights
    """
    from pvp.sb3.common.save_util import load_from_zip_file
    
    # Policy config (must match training config)
    policy_kwargs = dict(
        features_extractor_class=OurFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=275),
        share_features_extractor=False,
        net_arch=[256],
    )
    
    # Create base TD3 model
    model = TD3(
        policy=TD3Policy,
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        buffer_size=1,  # Minimal buffer for evaluation
        learning_starts=0,
        batch_size=1024,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        action_noise=None,
        tensorboard_log=None,
        verbose=2,
        seed=0,
        device=device,
    )
    
    print(f"Applying LoRA with rank={lora_rank}, alpha={lora_alpha}")
    
    # Apply LoRA to actor (must match training structure)
    apply_lora_to_model(
        model.actor.mu,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=0.0,
        verbose=True
    )
    
    # Apply LoRA to actor_target (must match training structure)
    apply_lora_to_model(
        model.actor_target.mu,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=0.0,
        verbose=False
    )
    
    # Recreate optimizer with LoRA parameters (must match training)
    # This ensures the optimizer state dict size matches when loading
    lora_params = get_lora_parameters(model.actor)
    model.actor.optimizer = torch.optim.Adam(lora_params, lr=1e-4)
    
    # Now load the saved parameters
    print(f"Loading checkpoint from {ckpt_path}")
    data, params, pytorch_variables = load_from_zip_file(
        ckpt_path, device=model.device, print_system_info=False
    )
    model.set_parameters(params, exact_match=False, device=model.device)
    
    print("Model loaded successfully!")
    print_trainable_parameters(model.actor, "Loaded Actor")
    
    return model


# ============================================================================
# Grad-CAM Visualization
# ============================================================================

def generate_gradcam(model, obs_dict, target_layer, action_dim=0, debug=False):
    """
    Generate Grad-CAM attention map for the actor network.
    
    Args:
        model: Actor model
        obs_dict: Input observation tensor (dict or tensor)
        target_layer: Target conv layer for Grad-CAM
        action_dim: Action dimension to visualize (0=steering, 1=acceleration, None=L2 norm)
        debug: Whether to print debug info
    
    Returns:
        cam: Attention map (numpy array) or None
    """
    activations = None
    gradients = None
    
    def save_activation(module, input, output):
        nonlocal activations
        activations = output
        activations.retain_grad()
    
    def save_gradient(module, grad_input, grad_output):
        nonlocal gradients
        if grad_output is not None and len(grad_output) > 0:
            grad = grad_output[0]
            if grad is not None:
                gradients = grad.clone()
    
    # Register hooks
    handle_forward = target_layer.register_forward_hook(save_activation)
    handle_backward = target_layer.register_full_backward_hook(save_gradient)
    
    try:
        model.eval()
        
        # Enable gradients for all parameters temporarily
        original_requires_grad = {}
        for name, param in model.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = True
        
        with torch.enable_grad():
            # Forward pass
            model_output = model.forward(obs_dict)
            
            # Compute target for backward
            if action_dim is not None:
                target_output = model_output[:, action_dim].mean()
            else:
                target_output = torch.norm(model_output, dim=1).mean()
            
            # Backward pass
            model.zero_grad()
            target_output.backward(retain_graph=True)
        
        # Restore original requires_grad
        for name, param in model.named_parameters():
            param.requires_grad = original_requires_grad[name]
        
        # Get gradients from activations.grad if hook didn't capture
        if gradients is None and activations is not None and activations.grad is not None:
            gradients = activations.grad
        
        if activations is None or gradients is None:
            if debug:
                print("Warning: Could not capture activations or gradients")
            return None
        
        # Compute CAM
        gradients_abs = torch.abs(gradients)
        weights = torch.mean(gradients_abs, dim=(2, 3), keepdim=True)
        activations_abs = torch.abs(activations)
        cam = torch.sum(weights * activations_abs, dim=1, keepdim=True)
        
        # Convert to numpy
        cam = cam.cpu().detach().numpy().squeeze()
        
        # Handle edge cases
        if cam.ndim == 0:
            cam = np.array([[max(cam.item(), 0.5)]])
        elif cam.ndim == 1:
            size = int(np.sqrt(cam.shape[0]))
            if size * size == cam.shape[0]:
                cam = cam.reshape(size, size)
            else:
                cam = cam.reshape(-1, 1)
        
        # Normalize to 0-1
        cam_range = cam.max() - cam.min()
        if cam_range > 1e-8:
            cam = (cam - cam.min()) / cam_range
        else:
            cam = np.ones_like(cam) * 0.5
        
        return cam
        
    except Exception as e:
        if debug:
            print(f"Error in generate_gradcam: {e}")
            import traceback
            traceback.print_exc()
        return None
    finally:
        handle_forward.remove()
        handle_backward.remove()


def get_conv_layer_by_index(features_extractor, layer_index=-3):
    """Find the target conv layer for Grad-CAM"""
    if hasattr(features_extractor, 'cnn'):
        cnn = features_extractor.cnn
        conv_layers = []
        for module in cnn.modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(module)
        
        if len(conv_layers) > 0:
            if abs(layer_index) <= len(conv_layers):
                return conv_layers[layer_index]
            return conv_layers[0]
    return None


def extract_frame_from_obs(obs):
    """Extract RGB frame from observation"""
    if isinstance(obs, dict) and "image" in obs:
        img = obs["image"]
        
        if len(img.shape) == 4:
            rgb_image = img[..., -1]
        elif len(img.shape) == 3:
            if img.shape[2] == 9:
                rgb_image = img[:, :, -3:].copy()
            elif img.shape[2] == 3:
                rgb_image = img.copy()
            else:
                rgb_image = img[:, :, :3].copy()
        else:
            return None
        
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            return None
        
        if rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        else:
            rgb_image = rgb_image.astype(np.uint8)
        
        return rgb_image
    return None


def visualize_attention(model, obs, original_image, step=0, layer_index=-3, action_dim=0,
                        save_dir=None):
    """
    Visualize CNN attention overlaid on the original image.
    
    Args:
        model: The LoRA model
        obs: Current observation (dict format)
        original_image: RGB image (H, W, 3) numpy array
        step: Current step number
        layer_index: Conv layer index for Grad-CAM
        action_dim: Action dimension (0=steering, 1=acceleration, None=L2 norm)
        save_dir: Directory to save attention images
    
    Returns:
        vis_image: Visualization image with attention overlay
    """
    try:
        actor = model.actor
        features_extractor = actor.features_extractor
        target_conv = get_conv_layer_by_index(features_extractor, layer_index)
        
        if target_conv is None:
            return original_image
        
        # Prepare input tensor
        obs_dict, _ = model.policy.obs_to_tensor(obs)
        
        # Ensure gradients for image
        if isinstance(obs_dict, dict):
            if "image" in obs_dict:
                obs_dict["image"] = obs_dict["image"].requires_grad_(True)
            if "state" in obs_dict:
                obs_dict["state"] = obs_dict["state"].requires_grad_(True)
        
        # Generate CAM
        cam = generate_gradcam(actor, obs_dict, target_conv, action_dim=action_dim,
                               debug=(step == 0))
        
        if cam is None:
            return original_image
        
        # Resize CAM to image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to heatmap
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        cam_heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        
        # Overlay on original image
        vis_image = cv2.addWeighted(original_image, 0.6, cam_heatmap, 0.4, 0)
        
        # Scale up for visibility
        display_scale = 4
        display_h, display_w = h * display_scale, w * display_scale
        vis_image_display = cv2.resize(vis_image, (display_w, display_h), 
                                       interpolation=cv2.INTER_LINEAR)
        
        # Add legend
        legend_height = 40
        final_image = np.zeros((display_h + legend_height, display_w, 3), dtype=np.uint8)
        final_image[:display_h, :, :] = vis_image_display
        
        # Add colorbar
        colorbar_width = display_w - 100
        colorbar_height = 15
        colorbar_start_x = 50
        colorbar_start_y = display_h + 5
        for i in range(colorbar_width):
            color_val = int(255 * i / colorbar_width)
            color = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), 
                                      cv2.COLORMAP_JET)[0, 0]
            final_image[colorbar_start_y:colorbar_start_y+colorbar_height,
                       colorbar_start_x+i, :] = color
        
        # Add labels
        cv2.putText(final_image, "Low", (10, display_h + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(final_image, "High", (display_w - 45, display_h + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(final_image, f"Step: {step}", (display_w - 80, display_h + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Save if directory specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"attention_step_{step:05d}.png")
            cv2.imwrite(save_path, final_image)
        
        return final_image
        
    except Exception as e:
        if step % 100 == 0:
            print(f"Warning: Error generating attention visualization: {e}")
        return original_image


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, env, num_episodes=50, save_video=False, video_dir="eval_videos",
                   visualize_attention_flag=False, attention_dir=None, 
                   layer_index=-3, action_dim=0):
    """
    Evaluate the LoRA model and optionally visualize attention.
    
    Args:
        model: The loaded LoRA model
        env: Evaluation environment
        num_episodes: Number of episodes to evaluate
        save_video: Whether to save videos
        video_dir: Directory to save videos
        visualize_attention_flag: Whether to visualize CNN attention
        attention_dir: Directory to save attention images
        layer_index: Conv layer index for Grad-CAM
        action_dim: Action dimension for attention (0=steering, 1=accel)
    """
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
    if visualize_attention_flag and attention_dir:
        os.makedirs(attention_dir, exist_ok=True)
    
    # Metrics
    episode_rewards = []
    episode_lengths = []
    success_buffer = []
    metrics_buffer = defaultdict(list)
    
    for ep in range(num_episodes):
        print(f"\n===== Episode {ep + 1}/{num_episodes} =====")
        
        obs = env.reset()
        done = False
        step_count = 0
        episode_reward = 0.0
        
        # Video writer
        out = None
        if save_video:
            frame = extract_frame_from_obs(obs)
            if frame is not None:
                h, w = frame.shape[:2]
                video_path = os.path.join(video_dir, f"episode_{ep + 1}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
                if out.isOpened():
                    out.write(frame)
        
        while not done and step_count < 1500:
            # Get action
            action, _ = model.predict(obs, deterministic=True)
            
            # Visualize attention if enabled
            if visualize_attention_flag:
                current_frame = extract_frame_from_obs(obs)
                if current_frame is not None:
                    vis_image = visualize_attention(
                        model, obs, current_frame,
                        step=step_count,
                        layer_index=layer_index,
                        action_dim=action_dim,
                        save_dir=attention_dir
                    )
                    
                    if step_count == 0:
                        cv2.namedWindow('CNN Attention (Red=High, Blue=Low)', cv2.WINDOW_NORMAL)
                        h, w = vis_image.shape[:2]
                        cv2.resizeWindow('CNN Attention (Red=High, Blue=Low)', w, h)
                    
                    cv2.imshow('CNN Attention (Red=High, Blue=Low)', vis_image)
                    cv2.waitKey(30)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            step_count += 1
            episode_reward += reward
            
            # Collect metrics
            if done:
                if "is_success" in info:
                    success_buffer.append(info["is_success"])
                if "arrive_dest" in info:
                    success_buffer.append(info["arrive_dest"])
                
                for k in ["route_completion", "total_cost", "arrive_dest", "crash", "out_of_road"]:
                    if k in info:
                        metrics_buffer[k].append(info[k])
            
            # Save video frame
            if save_video and out is not None:
                frame = extract_frame_from_obs(obs)
                if frame is not None:
                    out.write(frame)
        
        # Record episode results
        if "episode" in info:
            episode_rewards.append(info["episode"]["r"])
            episode_lengths.append(info["episode"]["l"])
        else:
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
        
        print(f"Episode finished: steps={step_count}, reward={episode_reward:.2f}")
        
        if out is not None:
            out.release()
    
    env.close()
    if visualize_attention_flag:
        cv2.destroyAllWindows()
    
    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    
    if episode_rewards:
        print(f"Reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    if episode_lengths:
        print(f"Length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")
    if success_buffer:
        print(f"Success Rate: {100 * np.mean(success_buffer):.2f}%")
    
    if metrics_buffer:
        print("\nAdditional Metrics:")
        for k, v in metrics_buffer.items():
            if v:
                print(f"  {k}: {np.mean(v):.4f}")
    
    print(f"{'='*60}\n")
    
    return {
        "mean_reward": np.mean(episode_rewards) if episode_rewards else 0,
        "mean_length": np.mean(episode_lengths) if episode_lengths else 0,
        "success_rate": np.mean(success_buffer) if success_buffer else 0,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LoRA fine-tuned model")
    
    # Model arguments
    parser.add_argument("--ckpt", type=str, required=True, 
                        help="Path to LoRA model checkpoint")
    parser.add_argument("--lora_rank", type=int, default=4,
                        help="LoRA rank (must match training)")
    parser.add_argument("--lora_alpha", type=float, default=1.0,
                        help="LoRA alpha (must match training)")
    
    # Evaluation arguments
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes to evaluate")
    parser.add_argument("--start_seed", type=int, default=1000,
                        help="Starting seed for evaluation environment")
    
    # Video arguments
    parser.add_argument("--save_video", action="store_true",
                        help="Save evaluation videos")
    parser.add_argument("--video_dir", type=str, default="eval_videos",
                        help="Directory to save videos")
    
    # Attention visualization arguments
    parser.add_argument("--visualize_attention", action="store_true",
                        help="Visualize CNN attention using Grad-CAM")
    parser.add_argument("--attention_dir", type=str, default="attention_vis",
                        help="Directory to save attention images")
    parser.add_argument("--layer_index", type=int, default=-3,
                        help="Conv layer index for Grad-CAM (-3=third from last)")
    parser.add_argument("--action_dim", type=int, default=0,
                        help="Action dimension (0=steering, 1=acceleration, -1=L2 norm)")
    
    # Rendering
    parser.add_argument("--no_render", action="store_true",
                        help="Disable rendering (faster evaluation)")
    
    args = parser.parse_args()
    
    # Create environment
    print("Creating evaluation environment...")
    use_render = not args.no_render or args.visualize_attention
    eval_env = make_eval_env(use_render=use_render, start_seed=args.start_seed)
    
    # Load LoRA model
    print("Loading LoRA model...")
    model = load_lora_model(
        args.ckpt,
        eval_env,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )
    
    # Handle action_dim=-1 as None
    action_dim = None if args.action_dim < 0 else args.action_dim
    
    # Evaluate
    print("Starting evaluation...")
    results = evaluate_model(
        model,
        eval_env,
        num_episodes=args.num_episodes,
        save_video=args.save_video,
        video_dir=args.video_dir,
        visualize_attention_flag=args.visualize_attention,
        attention_dir=args.attention_dir if args.visualize_attention else None,
        layer_index=args.layer_index,
        action_dim=action_dim
    )
    
    print("Evaluation complete!")
