import numpy as np
import gymnasium as gym
import torch
from dm_control.utils import inverse_kinematics as ik
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import shutil
from pathlib import Path

from gym_so101.constants import *

import gym_so101 

repo_id = "so101_touch_cube" 
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
    
    # Get the mathematically reachable orientation
    target_quat = calculate_target_quat(target_pos)

    # Solve IK with a relaxed tolerance
    result = ik.qpos_from_site_pose(
        physics,
        site_name="ee_site",
        target_pos=target_pos,
        target_quat=target_quat, 
        inplace=False,
        max_steps=200,
        tol=1e-3  
    )
    
    physics.data.qpos[:] = original_qpos
    
    raw_joints = np.zeros(6, dtype=np.float32)
    raw_joints[:5] = result.qpos[:5] 
    raw_joints[5] = gripper_action 
    
    normalized_action = normalize_so100(raw_joints)
    
    return normalized_action.astype(np.float32)

def collect_demonstrations(num_episodes=50):
    env = gym.make(
        "gym_so101/SO101TouchCube", 
        obs_type="so101_pixels_agent_pos",
        observation_width=640,
        observation_height=480
    )

    for episode in range(num_episodes):
        print(f"Recording episode {episode + 1}/{num_episodes}...")
        obs, info = env.reset()
        
        physics = env.unwrapped._env.physics
        
        # Get start positions
        block_pos = physics.named.data.site_xpos['red_cube_site'].copy()
        start_ee_pos = physics.named.data.site_xpos['ee_site'].copy()

        # Define the exact Keyframes (No Noise)
        hover_pos = block_pos + np.array([0.0, 0.0, 0.08]) # 8cm exactly above
        touch_pos = block_pos + np.array([0.0, 0.0, -0.01]) # Exactly on the block
        
        # Phases: (target_pos, interpolation_steps, settling_steps, gripper_action, apply noise)
        # settling_steps allows the physical simulation to catch up to the IK target
        phases = [
            (hover_pos, 40, 20, 0.0, True),    
            (touch_pos, 30, 15, 0.0, False),    
            (touch_pos, 0,  20, 0.0, False), # Hold touch   
        ]
        
        current_ee_pos = start_ee_pos

        for target_pos, interp_steps, settle_steps, gripper_action, apply_z in phases:
            
            noisy_target = add_noise(target_pos, noise_std=0.005, apply_to_z=apply_z)

            # Interpolation Phase (Moving smoothly)
            if interp_steps > 0:
                trajectory = generate_trajectory(current_ee_pos, noisy_target, interp_steps)
                for step_xyz in trajectory:
                    action = solve_ik(physics, step_xyz, gripper_action)
        
                    dataset.add_frame({
                        "observation.images.workspace_cam": obs["observation.images.workspace_cam"],
                        "observation.images.wrist_cam": obs["observation.images.wrist_cam"],
                        "observation.state": obs["observation.state"],
                        "action": action,
                        "task": "Touch Cube.",
                        "language_instruction": "Touch the red cube with the gripper."
                    })
                    
                    obs, reward, terminated, truncated, info = env.step(action)
            else:
                # If no interpolation steps, just calculate the final action once
                action = solve_ik(physics, step_xyz, gripper_action)
            
            # Settling Phase (Wait for physics to catch up to the final waypoint)
            for _ in range(settle_steps):
                dataset.add_frame({
                    "observation.images.workspace_cam": obs["observation.images.workspace_cam"],
                    "observation.images.wrist_cam": obs["observation.images.wrist_cam"],
                    "observation.state": obs["observation.state"],
                    "action": action,
                    "task": "Touch Cube.",
                    "language_instruction": "Touch the red cube with the gripper."
                })
                obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Phase complete. Reward: {reward:.3f}, Dist to cube: {info.get('is_success', False)}")
            
            # Update position for the next phase start
            current_ee_pos = target_pos
            
        dataset.save_episode()
        
    print("Dataset generation complete!")

if __name__ == "__main__":
    collect_demonstrations(num_episodes=50)