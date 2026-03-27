import gymnasium as gym
from gymnasium import spaces
import pybullet as pb
import pybullet_data
import numpy as np
import cv2
import math

class SO101SimEnv(gym.Env):
    """A Gymnasium wrapper for the PyBullet SO-101 arm, built for LeRobot."""
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        
        # --- 1. LeRobot Action & Observation Spaces ---
        # 6 DOF arm = 6 joint angles
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32)
        
        # LeRobot expects a dictionary with specific keys for states and images
        self.observation_space = spaces.Dict({
            "observation.state": spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32),
            "observation.images.workspace": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        })

        # --- 2. PyBullet Bootup ---
        self.physics_client = pb.connect(pb.GUI if render_mode == "human" else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.81)

        # --- 3. Load World & Robot ---
        self.plane_id = pb.loadURDF("plane.urdf")
        self.table_id = pb.loadURDF("table/table.urdf", basePosition=[0, 0.5, 0])
        
        # Load your SO-101 (Ensure the path is correct for your container)
        self.robot_id = pb.loadURDF("assets/so101/so101.urdf", basePosition=[0, 0, 0.625], useFixedBase=True)

        # Identify the 6 movable motors (ignoring fixed joints)
        self.motor_indices = []
        for i in range(pb.getNumJoints(self.robot_id)):
            if pb.getJointInfo(self.robot_id, i)[2] in [pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC]:
                self.motor_indices.append(i)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset all motors to 0 (Home position)
        for i, motor_idx in enumerate(self.motor_indices):
            pb.resetJointState(self.robot_id, motor_idx, targetValue=0.0)
            pb.setJointMotorControl2(self.robot_id, motor_idx, pb.POSITION_CONTROL, targetPosition=0.0)

        # Let physics settle
        for _ in range(10):
            pb.stepSimulation()

        return self._get_obs(), {}

    def step(self, action):
        # 1. Apply the LeRobot action array to the joints
        for i, motor_idx in enumerate(self.motor_indices):
            pb.setJointMotorControl2(self.robot_id, motor_idx, pb.POSITION_CONTROL, targetPosition=action[i])

        # 2. Step the physics engine (240Hz physics -> 30Hz control = 8 steps)
        for _ in range(8):
            pb.stepSimulation()

        # 3. Gather observations
        obs = self._get_obs()
        
        # Simulation environments for VLA data collection rarely use RL rewards/dones
        reward = 0.0
        terminated = False
        truncated = False
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """Extracts the current state and renders the camera for LeRobot."""
        # Get actual joint angles
        joint_states = pb.getJointStates(self.robot_id, self.motor_indices)
        current_joints = np.array([state[0] for state in joint_states], dtype=np.float32)

        # Render D435 Workspace Camera
        cam_pos = [0, -0.2, 1.2]
        target_pos = [0, 0.5, 0.625]
        view_matrix = pb.computeViewMatrix(cam_pos, target_pos, [0, 0, 1])
        proj_matrix = pb.computeProjectionMatrixFOV(42, 640/480, 0.01, 3.0)
        
        img_arr = pb.getCameraImage(640, 480, view_matrix, proj_matrix, renderer=pb.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.reshape(img_arr[2], (480, 640, 4))[..., :3]
        
        return {
            "observation.state": current_joints,
            "observation.images.workspace": rgb # LeRobot expects RGB, not BGR!
        }

    def close(self):
        pb.disconnect(self.physics_client)