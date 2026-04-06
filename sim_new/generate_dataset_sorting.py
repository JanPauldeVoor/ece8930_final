import numpy as np
import gymnasium as gym
import torch
from dm_control.utils import inverse_kinematics as ik
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import shutil
from pathlib import Path

from gym_so101.constants import *

import gym_so101 

repo_id = "SO101Sorting" 
local_dir = Path(f"local_datasets/{repo_id}")
if local_dir.exists():
    print(f"Found existing dataset at {local_dir}. Deleting and overwriting...")
    shutil.rmtree(local_dir)

dataset = LeRobotDataset.create(
    repo_id=repo_id,
    root=local_dir,
    fps=50,
    features={
        "observation.images.workspace_cam": {
            "dtype": "video", 
            "shape": (3, 480, 640),
            "names": ["c", "h", "w"] 
        },
        "observation.images.wrist_cam": {
            "dtype": "video", 
            "shape": (3, 480, 640),
            "names": ["c", "h", "w"]
        },
        "observation.state": {
            "dtype": "float32", 
            "shape": (6,),
            "names": SO101_JOINTS   
        },
        "action": {
            "dtype": "float32", 
            "shape": (6,),
            "names": SO101_JOINTS    
        },
        "language_instruction": {
            "dtype": "string", 
            "shape": (1,),
            "names": None           
        }
    }
)

def generate_trajectory(start_pos, end_pos, steps):
    """Linearly interpolates between two 3D points."""
    return np.linspace(start_pos, end_pos, steps)

def add_noise(target_pos, noise_std=0.005, apply_to_z=False):
    """
    Injects Gaussian noise into the target waypoint.
    """
    noise = np.random.normal(0, noise_std, size=3)
    if not apply_to_z:
        noise[2] = 0.0 
    return target_pos + noise

def calculate_target_quat(target_pos):
    """
    Dynamically calculates the reachable quaternion for a 5-DOF arm.
    It calculates the angle to the target and rotates the 'pointing down' 
    quaternion to match that yaw.
    """
    # Calculate the required base rotation (yaw) to look at the target (x,y)
    yaw = np.arctan2(target_pos[1], target_pos[0])
    
    # Generate a quaternion for a Z-axis rotation by 'yaw'
    q_dynamic = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    
    return q_dynamic


def solve_ik(physics, target_pos, gripper_action):
    original_qpos = physics.data.qpos.copy()
    
    # Get  mathematically reachable orientation
    target_quat = calculate_target_quat(target_pos)

    # Solve IK with a relaxed tolerance
    result = ik.qpos_from_site_pose(
        physics,
        site_name="ee_site",
        target_pos=target_pos,
        target_quat=target_quat, 
        inplace=False,
        max_steps=100,
        tol=1e-3  
    )
    
    physics.data.qpos[:] = original_qpos
    
    raw_joints = np.zeros(6, dtype=np.float32)
    raw_joints[:5] = result.qpos[:5] 
    raw_joints[5] = gripper_action 
    
    normalized_action = normalize_so100(raw_joints)
    
    return normalized_action.astype(np.float32)

def get_corrected_target(target_pos, z_offset, pinch_offset=0.015):
    """
    Calculates the target position correcting for the asymmetric gripper.
    pinch_offset shifts the target along the gripper's local Y-axis 
    (the axis the fingers close along).
    """
    # Calculate the yaw angle pointing toward the target
    yaw = np.arctan2(target_pos[1], target_pos[0])
    
    # Calculate the local Y-axis vector in world space
    # (If local X points to the target, local Y is perpendicular to it)
    local_y_world = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    
    # Apply the offset to shift the ee_site so the fixed jaw sits flush with the cube
    corrected_pos = target_pos.copy()
    corrected_pos += local_y_world * pinch_offset
    corrected_pos[2] += z_offset
    
    return corrected_pos

def collect_demonstrations(num_episodes=50):
    env = gym.make(
        "gym_so101/SO101Sorting", 
        obs_type="so101_pixels_agent_pos",
        observation_width=640,
        observation_height=480
    )

    for episode in range(num_episodes):
        print(f"Recording episode {episode + 1}/{num_episodes}...")
        obs, info = env.reset()
        
        physics = env.unwrapped._env.physics
        
        # Get start positions       
        red_cube_pos = physics.named.data.site_xpos['red_cube_site'].copy()
        blue_cube_pos = physics.named.data.site_xpos['blue_cube_site'].copy()
        left_bin_pos = physics.named.data.site_xpos['left_bin_center'].copy()
        right_bin_pos = physics.named.data.site_xpos['right_bin_center'].copy()

        start_ee_pos = physics.named.data.site_xpos['ee_site'].copy()

        # Tune pinch_offset (e.g., +0.015 or -0.015) until the fixed jaw perfectly grazes the cube
        PINCH_OFFSET = 0.015 

        red_cube_hover_pos = get_corrected_target(red_cube_pos, z_offset=0.08, pinch_offset=PINCH_OFFSET)
        red_cube_touch_pos = get_corrected_target(red_cube_pos, z_offset=-0.01, pinch_offset=PINCH_OFFSET)

        blue_cube_hover_pos = get_corrected_target(blue_cube_pos, z_offset=0.08, pinch_offset=PINCH_OFFSET)
        blue_cube_touch_pos = get_corrected_target(blue_cube_pos, z_offset=-0.01, pinch_offset=PINCH_OFFSET)

        left_bin_hover_pos = left_bin_pos + np.array([0.0, 0.0, 0.08])
        right_bin_hover_pos = right_bin_pos + np.array([0.0, 0.0, 0.08])

        left_bin_release_pos = left_bin_pos + np.array([0.0, 0.0, 0.04])
        right_bin_release_pos = right_bin_pos + np.array([0.0, 0.0, 0.04])
        
        # Phases: (target_pos, interpolation_steps, settling_steps, gripper_action, apply noise)
        # settling_steps allows the physical simulation to catch up to the IK target
        phases = [
                # (target_site, offset, interpolation_steps, settling_steps, gripper_action, apply_noise)
                ('red_cube_site', np.array([0.0, 0.0, 0.08]), 40, 20, 1, True),   # hover over red cube
                ('red_cube_site', np.array([0.0, 0.0, -0.01]), 30, 20, 1, True),  # touch red cube
                ('red_cube_site', np.array([0.0, 0.0, -0.01]), 30, 15, 0.02, False), # close gripper
                ('red_cube_site', np.array([0.0, 0.0, 0.08]), 30, 20, 0.02, False),  # Lift cube   
                # ('right_bin_center', np.array([0.0, 0.0, 0.08]), 40, 20, 0.02, True),   # Hover right bin
                # ('right_bin_center', np.array([0.0, 0.0, 0.04]), 30, 20, 1, False),     # Drop Cube  
            ]

        for target_site, offset, interp_steps, settle_steps, gripper_action, apply_z in phases:
            
            # Interpolation Phase (Moving smoothly toward a potentially moving target)
            for step in range(interp_steps):
                fresh_target_pos = physics.named.data.site_xpos[target_site].copy() + offset
                noisy_target = add_noise(fresh_target_pos, noise_std=0.005, apply_to_z=apply_z)

                current_ee_pos = physics.named.data.site_xpos['ee_site'].copy()

                steps_left = interp_steps - step
                step_xyz = current_ee_pos + (noisy_target - current_ee_pos) / steps_left
                
                action = solve_ik(physics, step_xyz, gripper_action)
    
                dataset.add_frame({
                    "observation.images.workspace_cam": obs["observation.images.workspace_cam"],
                    "observation.images.wrist_cam": obs["observation.images.wrist_cam"],
                    "observation.state": obs["observation.state"],
                    "action": action,
                    "task": "Sort Cubes.",
                    "language_instruction": "Place the red cube in the right bin, and place the blue cube in the left bin."
                })
                
                obs, reward, terminated, truncated, info = env.step(action)

            # Settling Phase (Wait for physics to catch up, still tracking if needed)
            for _ in range(settle_steps):
                # Recalculate static target in case the object is sliding while we settle
                fresh_target_pos = physics.named.data.site_xpos[target_site].copy() + offset
                action = solve_ik(physics, fresh_target_pos, gripper_action)

                dataset.add_frame({
                    "observation.images.workspace_cam": obs["observation.images.workspace_cam"],
                    "observation.images.wrist_cam": obs["observation.images.wrist_cam"],
                    "observation.state": obs["observation.state"],
                    "action": action,
                    "task": "Sort Cubes.",
                    "language_instruction": "Place the red cube in the right bin, and place the blue cube in the left bin."
                })
                obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Phase complete. Reward: {reward:.3f}, Dist to cube: {info.get('is_success', False)}")
            
            print(f"Red: {red_cube_pos}\nHover:{red_cube_hover_pos}\nEE:{current_ee_pos}")

            
        dataset.save_episode()
        
    print("Dataset generation complete!")

if __name__ == "__main__":
    collect_demonstrations(num_episodes=1)