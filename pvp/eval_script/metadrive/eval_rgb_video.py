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


def generate_gradcam(model, obs_dict, target_layer, target_output=None, debug=False, action_dim=None):
    """
    生成Grad-CAM attention map
    
    Args:
        model: Actor模型
        obs_dict: 输入观察tensor（可以是dict或单个tensor）
        target_layer: 目标卷积层
        target_output: 目标输出（用于反向传播），如果为None则从model_output计算
        debug: 是否打印详细调试信息
        action_dim: 要可视化的action维度 (0=steering, 1=acceleration, None=L2 norm)
    
    Returns:
        cam: attention map (numpy array) 或 None
    """
    activations = None
    gradients = None
    gradient_hook_handle = None
    hook_called = False
    
    def save_activation(module, input, output):
        nonlocal activations, gradient_hook_handle
        # 保存激活（不detach，保持梯度图）
        activations = output
        
        # 关键修复：使用retain_grad()确保非叶子tensor的梯度被保留
        # 因为卷积层的输出不是叶子tensor，默认不会保存梯度
        activations.retain_grad()
        
        if debug:
            print(f"Debug: Activation captured - shape: {output.shape}, requires_grad: {output.requires_grad}")
            print(f"Debug: Activation grad_fn: {output.grad_fn}")
            print(f"Debug: Called retain_grad() on activations")
        
        # 在输出tensor上直接注册hook来捕获梯度
        def gradient_hook(grad):
            nonlocal gradients, hook_called
            hook_called = True
            if debug:
                print(f"Debug: Gradient hook called! grad is None: {grad is None}")
                if grad is not None:
                    print(f"Debug: Gradient shape: {grad.shape}, mean: {grad.mean().item():.6f}, max: {grad.max().item():.6f}")
            if grad is not None:
                gradients = grad.clone()
            return grad
        
        # 注册hook - 这会在反向传播时被调用
        gradient_hook_handle = output.register_hook(gradient_hook)
        if debug:
            print(f"Debug: Hook registered on output tensor")
    
    # 注册前向hook
    handle_forward = target_layer.register_forward_hook(save_activation)
    
    # 也注册反向hook作为备用
    def save_gradient_backup(module, grad_input, grad_output):
        nonlocal gradients, hook_called
        hook_called = True
        if debug:
            print(f"Debug: Backup backward hook called! grad_output: {grad_output}")
        if grad_output is not None and len(grad_output) > 0:
            grad = grad_output[0]
            if grad is not None:
                if debug:
                    print(f"Debug: Backup gradient captured - shape: {grad.shape}")
                gradients = grad.clone()
    
    handle_backward = target_layer.register_full_backward_hook(save_gradient_backup)
    
    try:
        # 确保模型允许梯度流
        model.eval()
        
        # 确保所有参数都允许梯度
        for param in model.parameters():
            if param.requires_grad is False:
                param.requires_grad = True
        
        with torch.enable_grad():
            # 前向传播 - 重新计算以确保梯度图连续
            model_output = model.forward(obs_dict)
            if debug:
                print(f"Debug: Model output - shape: {model_output.shape}, requires_grad: {model_output.requires_grad}")
                print(f"Debug: Model output grad_fn: {model_output.grad_fn}")
            
            # 关键修复：从model_output重新计算target_output，确保梯度图连续
            if target_output is None or not target_output.requires_grad or target_output.grad_fn is None:
                if action_dim is not None:
                    # 使用指定的action维度 (0=steering, 1=acceleration)
                    # 直接使用原始值（不用abs），后面在计算CAM时对梯度取绝对值
                    # 这样可以捕获正负两个方向的影响
                    target_output = model_output[:, action_dim].mean()
                    if debug:
                        print(f"Debug: Using action_dim={action_dim} as target, raw value={model_output[:, action_dim].item():.4f}")
                else:
                    # 使用L2 norm
                    target_output = torch.norm(model_output, dim=1).mean()
            if debug:
                print(f"Debug: Using target_output - requires_grad: {target_output.requires_grad}, grad_fn: {target_output.grad_fn}")
                # 检查activations是否在前向传播中被创建
                if activations is not None:
                    print(f"Debug: Activations captured during forward - shape: {activations.shape}, grad_fn: {activations.grad_fn}")
                    # 检查activations是否真的在计算图中
                    print(f"Debug: Checking if activations is in computation graph...")
                    # 尝试通过activations创建一个依赖
                    test_sum = activations.sum()
                    print(f"Debug: activations.sum() - requires_grad: {test_sum.requires_grad}, grad_fn: {test_sum.grad_fn}")
                else:
                    print(f"Debug: WARNING - Activations NOT captured during forward!")
            
            # 反向传播前，确保activations在计算图中
            # 关键：检查activations是否真的连接到model_output
            if activations is not None and debug:
                # 尝试通过activations创建一个依赖，看看它是否真的在计算图中
                # 如果activations在计算图中，那么通过它计算的值应该能连接到model_output
                test = activations.sum() + model_output.sum() * 0  # 创建一个依赖
                print(f"Debug: Test connection - test.requires_grad: {test.requires_grad}, test.grad_fn: {test.grad_fn}")
            
            # 反向传播
            model.zero_grad()
            target_output.backward(retain_graph=True)
            
            if debug:
                print(f"Debug: Backward pass completed")
                print(f"Debug: Hook called: {hook_called}")
                print(f"Debug: Gradients captured: {gradients is not None}")
                # 检查梯度流
                print(f"Debug: After backward:")
                print(f"  - target_output.grad_fn: {target_output.grad_fn}")
                print(f"  - model_output.grad_fn: {model_output.grad_fn}")
                if activations is not None:
                    print(f"  - activations.grad_fn: {activations.grad_fn}")
                    print(f"  - activations.requires_grad: {activations.requires_grad}")
                    # 检查activations的梯度（因为调用了retain_grad()，应该会有梯度）
                    if hasattr(activations, 'grad'):
                        if activations.grad is not None:
                            print(f"  - activations.grad exists (from retain_grad) - shape: {activations.grad.shape}")
                        else:
                            print(f"  - activations.grad is None (even with retain_grad)")
                            # 尝试手动检查梯度流
                            print(f"  - Checking if activations is connected to model_output...")
                            # 检查activations的grad_fn是否指向model_output
                            if hasattr(activations, 'grad_fn') and activations.grad_fn is not None:
                                print(f"    activations.grad_fn type: {type(activations.grad_fn)}")
                # 检查模型输出的梯度
                if model_output.grad is not None:
                    print(f"Debug: model_output.grad exists - shape: {model_output.grad.shape}")
                else:
                    print(f"Debug: model_output.grad is None")
        
        # 检查是否捕获了激活和梯度
        if activations is None:
            print("Warning: No activations captured")
            return None
        
        # 如果hook没有捕获到梯度，尝试从activations.grad获取（因为调用了retain_grad()）
        if gradients is None:
            if activations is not None and hasattr(activations, 'grad') and activations.grad is not None:
                # 使用retain_grad()保存的梯度
                gradients = activations.grad
                if debug:
                    print(f"Debug: Using gradients from activations.grad (retain_grad) - shape: {gradients.shape}")
            else:
                print(f"Warning: No gradients captured from hook or retain_grad")
                print(f"Debug: activations.requires_grad: {activations.requires_grad if hasattr(activations, 'requires_grad') else 'N/A'}")
                print(f"Debug: activations.grad_fn: {activations.grad_fn if hasattr(activations, 'grad_fn') else 'N/A'}")
                print(f"Debug: gradient_hook_handle: {gradient_hook_handle}")
                print(f"Debug: hook_called: {hook_called}")
                print(f"Debug: activations.grad: {activations.grad if hasattr(activations, 'grad') else 'N/A'}")
                # 检查target_output的梯度
                if hasattr(target_output, 'grad') and target_output.grad is not None:
                    print(f"Debug: target_output.grad exists - shape: {target_output.grad.shape}")
                else:
                    print(f"Debug: target_output.grad is None")
                return None
        
        # 计算权重（全局平均池化梯度）
        gradients_tensor = gradients  # [batch, channels, H, W]
        activations_tensor = activations  # [batch, channels, H, W]
        
        # 关键修复：对梯度取绝对值，这样正负影响都会被捕获
        # 这意味着：不管一个像素让steering变大还是变小，只要影响大就显示红色
        # 原始Grad-CAM只关注正贡献，但我们想看所有显著影响
        gradients_abs = torch.abs(gradients_tensor)
        
        # 对每个通道的梯度绝对值进行全局平均池化
        weights = torch.mean(gradients_abs, dim=(2, 3), keepdim=True)  # [batch, channels, 1, 1]
        
        # 使用激活的绝对值，因为我们关心的是激活的强度而非方向
        activations_abs = torch.abs(activations_tensor)
        
        # 加权组合特征图
        cam = torch.sum(weights * activations_abs, dim=1, keepdim=True)  # [batch, 1, H, W]
        
        # 不需要ReLU了，因为abs已经保证非负
        if debug:
            print(f"Debug: CAM range before normalization: [{cam.min().item():.4f}, {cam.max().item():.4f}]")
        
        # 归一化到0-1
        cam = cam.cpu().detach().numpy()
        
        # 移除batch和channel维度，保留H, W
        # cam shape: [batch, 1, H, W] -> [H, W]
        cam = cam.squeeze()  # 移除所有size=1的维度
        
        # 处理各种形状情况
        if cam.ndim == 0:
            # 标量（1x1 feature map），创建一个1x1的2D数组
            if debug:
                print(f"Debug: CAM is a scalar (1x1 feature map). Using uniform attention.")
            cam = np.array([[max(cam.item(), 0.5)]])  # 使用0.5作为默认值，避免全零
        elif cam.ndim == 1:
            # 1D数组，reshape成2D
            size = int(np.sqrt(cam.shape[0]))
            if size * size == cam.shape[0]:
                cam = cam.reshape(size, size)
            else:
                # 无法reshape，创建一个列向量
                cam = cam.reshape(-1, 1)
        elif cam.ndim > 2:
            # 多余维度，squeeze到2D
            while cam.ndim > 2:
                cam = cam.squeeze(0)
        
        if debug:
            print(f"Debug: CAM shape after processing: {cam.shape}")
        
        # 归一化到0-1
        cam_range = cam.max() - cam.min()
        if cam_range > 1e-8:
            cam = (cam - cam.min()) / cam_range
        else:
            # CAM全为相同值或接近零，使用均匀分布
            if debug:
                print(f"Debug: CAM has very small range ({cam_range:.6f}), using uniform attention")
            cam = np.ones_like(cam) * 0.5  # 使用0.5作为均匀注意力
        
        return cam
    except Exception as e:
        print(f"Error in generate_gradcam: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 移除hook
        handle_forward.remove()
        handle_backward.remove()
        if gradient_hook_handle is not None:
            gradient_hook_handle.remove()


class GradCAM:
    """Grad-CAM可视化类，用于可视化CNN的注意力"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册hook
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, target_output):
        """
        生成Grad-CAM attention map
        
        Args:
            input_tensor: 输入图像tensor
            target_output: 目标输出（通常是action的某个维度）
        
        Returns:
            cam: attention map (numpy array)
        """
        # 前向传播
        self.model.eval()
        output = self.model.forward(input_tensor)
        
        # 反向传播
        self.model.zero_grad()
        target_output.backward(retain_graph=True)
        
        # 计算权重（全局平均池化梯度）
        gradients = self.gradients[0]  # [batch, channels, H, W]
        activations = self.activations[0]  # [batch, channels, H, W]
        
        # 对每个通道的梯度进行全局平均池化
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [batch, channels, 1, 1]
        
        # 加权组合特征图
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [batch, 1, H, W]
        cam = F.relu(cam)  # 只保留正激活
        
        # 归一化到0-1
        cam = cam.squeeze().cpu().detach().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam


def get_conv_layer_by_index(features_extractor, layer_index=-3, verbose=False):
    """找到CNN的指定卷积层用于Grad-CAM
    
    Args:
        features_extractor: CNN特征提取器
        layer_index: 卷积层索引，负数表示从后往前数
                     -1: 最后一个 (1x1 输出，不推荐)
                     -2: 倒数第二个 (4x4 输出)
                     -3: 倒数第三个 (更大空间分辨率，推荐)
        verbose: 是否打印详细信息
    
    Returns:
        目标卷积层
    """
    if hasattr(features_extractor, 'cnn'):
        cnn = features_extractor.cnn
        # 遍历Sequential找到所有Conv2d
        conv_layers = []
        for module in cnn.modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(module)
        
        # 打印所有卷积层信息（用于调试）
        if len(conv_layers) > 0:
            if verbose:
                print(f"Debug: Found {len(conv_layers)} Conv2d layers")
                for i, conv in enumerate(conv_layers):
                    print(f"  Conv {i}: {conv}")
            
            # 选择指定索引的卷积层
            if abs(layer_index) <= len(conv_layers):
                target_conv = conv_layers[layer_index]
                if verbose:
                    print(f"Debug: Using conv layer at index {layer_index}: {target_conv}")
            else:
                target_conv = conv_layers[0]  # fallback to first layer
                if verbose:
                    print(f"Debug: Index {layer_index} out of range, using first conv layer: {target_conv}")
            
            return target_conv
    return None


def get_cnn_output_before_flatten(features_extractor):
    """获取CNN在Flatten之前的输出（用于Grad-CAM）"""
    if hasattr(features_extractor, 'cnn'):
        cnn = features_extractor.cnn
        # 找到Flatten层之前的所有层
        layers_before_flatten = []
        for module in cnn.children():
            if isinstance(module, nn.Flatten):
                break
            layers_before_flatten.append(module)
        
        if len(layers_before_flatten) > 0:
            return nn.Sequential(*layers_before_flatten)
    return None


def analyze_cnn_contribution(model, obs, debug=True):
    """
    分析CNN特征对决策的贡献程度
    
    通过比较：
    1. 正常输出
    2. 将CNN特征置零后的输出
    来判断CNN是否真的在使用图像信息
    
    Returns:
        contribution_ratio: CNN贡献比例 (0-1, 越高说明越依赖图像)
    """
    try:
        actor = model.policy.actor
        features_extractor = actor.features_extractor
        
        # 准备输入tensor
        obs_dict, _ = model.policy.obs_to_tensor(obs)
        
        # 正常前向传播
        with torch.no_grad():
            normal_action = actor.forward(obs_dict)
            normal_action_np = normal_action.cpu().numpy()
        
        # 将图像置零后的前向传播
        if isinstance(obs_dict, dict) and "image" in obs_dict:
            obs_dict_zeroed = {k: v.clone() for k, v in obs_dict.items()}
            obs_dict_zeroed["image"] = torch.zeros_like(obs_dict_zeroed["image"])
            
            with torch.no_grad():
                zeroed_action = actor.forward(obs_dict_zeroed)
                zeroed_action_np = zeroed_action.cpu().numpy()
            
            # 计算差异
            action_diff = np.abs(normal_action_np - zeroed_action_np)
            action_magnitude = np.abs(normal_action_np) + 1e-8
            contribution_ratio = np.mean(action_diff / action_magnitude)
            
            if debug:
                print(f"\n=== CNN Contribution Analysis ===")
                print(f"Normal action:  steering={normal_action_np[0, 0]:.4f}, accel={normal_action_np[0, 1]:.4f}")
                print(f"Zeroed action:  steering={zeroed_action_np[0, 0]:.4f}, accel={zeroed_action_np[0, 1]:.4f}")
                print(f"Action diff:    steering={action_diff[0, 0]:.4f}, accel={action_diff[0, 1]:.4f}")
                print(f"CNN contribution ratio: {contribution_ratio:.2%}")
                if contribution_ratio < 0.1:
                    print("WARNING: CNN contribution is very low! The model may be relying mostly on state info.")
                elif contribution_ratio > 0.5:
                    print("Good: CNN is significantly contributing to the decision.")
                print("=" * 35 + "\n")
            
            return contribution_ratio
        else:
            print("Warning: Cannot analyze CNN contribution - no image in obs_dict")
            return 0.0
    except Exception as e:
        print(f"Error analyzing CNN contribution: {e}")
        return 0.0


def visualize_attention(model, obs, original_image, attention_save_dir=None, step=0, 
                        layer_index=-3, action_dim=0):
    """
    可视化CNN的注意力，将attention map叠加到原始图像上
    
    Args:
        model: 训练好的模型
        obs: 当前观察（dict格式，包含image和state）
        original_image: 原始RGB图像 (H, W, 3) numpy array
        attention_save_dir: 保存attention图像的目录
        step: 当前步数
        layer_index: 使用的卷积层索引 (-3=倒数第三层，有更高分辨率)
        action_dim: 要可视化的action维度 (0=steering, 1=acceleration, None=L2 norm)
    
    Returns:
        vis_image: 叠加了attention map的可视化图像
    """
    try:
        # 获取actor网络
        actor = model.policy.actor
        
        # 找到目标卷积层
        features_extractor = actor.features_extractor
        target_conv = get_conv_layer_by_index(features_extractor, layer_index=layer_index, verbose=(step == 0))
        
        if target_conv is None:
            if step == 0:
                print("Warning: Could not find target conv layer for Grad-CAM")
            return original_image
        
        if step == 0:
            print(f"Debug: Target conv layer: {target_conv}")
            print(f"Debug: Features extractor type: {type(features_extractor)}")
            print(f"Debug: Action dimension for Grad-CAM: {action_dim} ({'steering' if action_dim == 0 else 'acceleration' if action_dim == 1 else 'L2 norm'})")
        
        # 准备输入tensor - obs_to_tensor返回(obs_dict, vectorized)
        obs_dict, _ = model.policy.obs_to_tensor(obs)
        
        # 对于字典观察，需要为每个tensor设置requires_grad
        # 特别是image tensor需要梯度来计算attention
        if isinstance(obs_dict, dict):
            # 确保image tensor需要梯度
            if "image" in obs_dict:
                obs_dict["image"] = obs_dict["image"].requires_grad_(True)
            # state tensor也可以设置，但通常不需要
            if "state" in obs_dict:
                obs_dict["state"] = obs_dict["state"].requires_grad_(True)
        else:
            # 如果不是字典，直接设置
            obs_dict = obs_dict.requires_grad_(True)
        
        # 生成CAM - generate_gradcam内部会进行前向和反向传播
        # 不需要在这里预先计算target，因为generate_gradcam会重新计算以确保梯度图连续
        # 只在第一步启用详细调试
        cam = generate_gradcam(actor, obs_dict, target_conv, None, 
                               debug=(step == 0), action_dim=action_dim)
        
        if cam is None:
            if step == 0 or step % 100 == 0:  # 首次或每100步打印一次
                print(f"Warning: CAM generation returned None at step {step}")
                print("This means Grad-CAM failed. Returning original image.")
            return original_image
        
        # 检查CAM尺寸
        if step == 0 or step % 100 == 0:
            print(f"Debug: CAM generated successfully! CAM shape: {cam.shape}, original image shape: {original_image.shape}")
            print(f"Debug: CAM value range: [{cam.min():.4f}, {cam.max():.4f}]")
        
        # 将CAM上采样到原始图像尺寸
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 转换为热力图 (JET colormap: 蓝色=低注意力, 红色=高注意力)
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        cam_heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        
        # 叠加到原始图像上 (原始图像60%, 热力图40%)
        vis_image = cv2.addWeighted(original_image, 0.6, cam_heatmap, 0.4, 0)
        
        # 放大可视化图像以提高可见性 (4x)
        display_scale = 4
        display_h, display_w = h * display_scale, w * display_scale
        vis_image_display = cv2.resize(vis_image, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
        
        # 添加颜色图例和说明
        # 创建一个更大的画布来放置图例
        legend_height = 40
        final_image = np.zeros((display_h + legend_height, display_w, 3), dtype=np.uint8)
        final_image[:display_h, :, :] = vis_image_display
        
        # 添加颜色条 (从蓝到红)
        colorbar_width = display_w - 100
        colorbar_height = 15
        colorbar_start_x = 50
        colorbar_start_y = display_h + 5
        for i in range(colorbar_width):
            color_val = int(255 * i / colorbar_width)
            color = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
            final_image[colorbar_start_y:colorbar_start_y+colorbar_height, 
                       colorbar_start_x+i, :] = color
        
        # 添加文字标签
        cv2.putText(final_image, "Low", (10, display_h + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(final_image, "High", (display_w - 45, display_h + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(final_image, f"Step: {step}", (display_w - 80, display_h + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # 保存attention图像（如果指定了目录）
        if attention_save_dir:
            os.makedirs(attention_save_dir, exist_ok=True)
            attention_path = os.path.join(attention_save_dir, f"attention_step_{step:05d}.png")
            cv2.imwrite(attention_path, final_image)
        
        return final_image
        
    except Exception as e:
        # 静默失败，返回原图（避免刷屏）
        if step % 100 == 0:  # 每100步打印一次错误
            print(f"Warning: Error generating attention visualization: {e}")
        return original_image


def evaluate_with_video(model, env, num_episodes=5, save_video=False, video_save_dir="eval_videos", 
                        enable_attention_vis=False, attention_save_dir=None,
                        layer_index=-3, action_dim=0):
    """Evaluate model and optionally record videos, collecting all evaluation metrics
    
    Args:
        model: 训练好的模型
        env: 评估环境
        num_episodes: episode数量
        save_video: 是否保存视频
        video_save_dir: 视频保存目录
        enable_attention_vis: 是否可视化CNN注意力
        attention_save_dir: 注意力图像保存目录（如果为None，则只显示不保存）
        layer_index: Grad-CAM使用的卷积层索引 (-3=倒数第三层，有更高分辨率)
        action_dim: 要可视化的action维度 (0=steering, 1=acceleration, None=L2 norm)
    """
    if save_video:
        os.makedirs(video_save_dir, exist_ok=True)
    if enable_attention_vis and attention_save_dir:
        os.makedirs(attention_save_dir, exist_ok=True)
    
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
            
            # Visualize attention if enabled
            if enable_attention_vis:
                # 在第一步分析CNN的贡献程度
                if step_count == 0 and ep == 0:
                    analyze_cnn_contribution(model, obs, debug=True)
                
                # Extract current frame for visualization
                current_frame = extract_frame_from_obs(obs)
                if current_frame is not None:
                    # Generate attention visualization
                    vis_image = visualize_attention(
                        model, obs, current_frame, 
                        attention_save_dir=attention_save_dir,
                        step=step_count,
                        layer_index=layer_index,
                        action_dim=action_dim
                    )
                    # Display attention visualization with larger window
                    # vis_image 已经在 visualize_attention 中放大了4倍
                    h, w = vis_image.shape[:2]
                    
                    # 只在第一帧创建窗口（避免闪烁）
                    if step_count == 0:
                        cv2.namedWindow('CNN Attention (Red=High, Blue=Low)', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('CNN Attention (Red=High, Blue=Low)', w, h)
                    
                    cv2.imshow('CNN Attention (Red=High, Blue=Low)', vis_image)
                    # 增加等待时间到30ms，确保稳定显示
                    cv2.waitKey(30)
            
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
    parser.add_argument("--visualize_attention", action="store_true", help="Visualize CNN attention (default: False)")
    parser.add_argument("--attention_dir", type=str, default="attention_vis", help="Directory to save attention visualizations")
    parser.add_argument("--layer_index", type=int, default=-3, 
                        help="Conv layer index for Grad-CAM (-3=third from last, higher resolution; -2=second from last)")
    parser.add_argument("--action_dim", type=int, default=0, 
                        help="Action dimension to visualize (0=steering, 1=acceleration, -1=L2 norm)")
    args = parser.parse_args()
    
    # Create environment (num_env=1, single environment)
    print("Creating evaluation environment...")
    eval_env = make_eval_env()
    
    # Load model
    print("Loading model...")
    model = load_model(args.ckpt, eval_env)
    
    # Evaluate and optionally record videos
    print("Starting evaluation...")
    if args.visualize_attention:
        print("CNN Attention visualization enabled")
        print("Red/yellow regions indicate areas the CNN focuses on")
    # Handle action_dim=-1 as None (for L2 norm)
    action_dim = None if args.action_dim < 0 else args.action_dim
    
    evaluate_with_video(
        model, 
        eval_env, 
        num_episodes=args.num_episodes, 
        save_video=args.save_video,
        video_save_dir=args.video_dir,
        enable_attention_vis=args.visualize_attention,
        attention_save_dir=args.attention_dir if args.visualize_attention else None,
        layer_index=args.layer_index,
        action_dim=action_dim
    )
    
    if args.visualize_attention:
        cv2.destroyAllWindows()