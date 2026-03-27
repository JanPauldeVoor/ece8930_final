import numpy as np
from dm_control.suite import base
from dm_control.rl import control

from gym_so101.constants import SO100_START_ARM_POSE, unnormalize_so100

# Global variable used by env.py to pass the randomized pose down to the task
BOX_POSE = [None, None] 


class SO101SortingTask(base.Task):
    """
    dm_control Task for SO101. 
    Handles physics initialization, action application, and raw observations.
    """
    ARM_DOF = 5
    GRIPPER_DOF = 2

    def __init__(self, random=None, observation_width=640, observation_height=480):
        self.observation_width = observation_width
        self.observation_height = observation_height
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)
        
        # Apply the randomized block poses to the freejoints in MuJoCo
        # BOX_POSE[0] should be red block, BOX_POSE[1] should be blue block
        if BOX_POSE[0] is not None:
            physics.named.data.qpos[-7:] = BOX_POSE[0]
        # if BOX_POSE[1] is not None:
        #     physics.named.data.qpos['blue_block'] = BOX_POSE[1]

    def before_step(self, action, physics):
        action = action.copy()
        left_arm_action = action[: self.ARM_DOF + 1]
        env_action = unnormalize_so100(left_arm_action)
        super().before_step(env_action, physics)
        return

    def get_observation(self, physics):
        """Extracts raw data from the physics engine."""
        obs = {}
        # Read the 6 hinge joints
        obs['qpos'] = physics.data.qpos[:6].copy()
        
        # Get coordinates of the sites you added to the XML
        obs['red_block_position'] = physics.named.data.site_xpos['red_block_center'].copy()
        # obs['blue_block_position'] = physics.named.data.site_xpos['blue_block_center'].copy()
        obs['ee_position'] = physics.named.data.site_xpos['gripper_center'].copy()
        
        # Render cameras using dm_control's physics engine
        obs['images'] = {}
        obs['images']['workspace_cam'] = physics.render(height=480, width=640, camera_id='workspace_cam')
        obs['images']['wrist_cam'] = physics.render(height=480, width=640, camera_id='wrist_cam')
        
        return obs

    def get_reward(self, physics):
        """Calculates the scalar reward based on site distances."""
        red_pos = physics.named.data.site_xpos['red_block_center']
        left_bin = physics.named.data.site_xpos['left_bin_dropzone']
        
        # Simple distance check for the red block
        distance = np.linalg.norm(red_pos - left_bin)
        return 1.0 if distance < 0.05 else 0.0

    # Required custom methods for your GoalEnv wrapper to call
    def get_red_cube_position(self, physics):
        return physics.named.data.site_xpos['red_block_center'].copy()