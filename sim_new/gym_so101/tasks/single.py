import numpy as np
import collections

from dm_control.suite import base
from dm_control.rl import control

from gym_so101.constants import SO101_START_ARM_POSE, unnormalize_so101, RED_CUBE_GEOM, BLUE_CUBE_GEOM, TABLE_GEOM

# Global variable used by env.py to pass the randomized pose down to the task
BOX_POSE = [None, None]


"""
Heavily taken from: https://github.com/ilonajulczuk/gym-so100-c/blob/main/gym_so100/tasks/single_arm.py

Environment for simulated robot one arm manipulation, with joint position control
Action space:      [left_arm_qpos (5),             # absolute joint position
                    left_gripper_positions (1),    # absolute gripper position
                    

Observation space: {"qpos": Concat[ left_arm_qpos (5),         # absolute joint position
                                    left_gripper_position (1),  # absolute gripper position
                    "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                    left_gripper_velocity (1),  # absolute gripper velocity (pos: opening, neg: closing)
                    "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'

physics.named.data.qpos mapping: 
 0    shoulder_pan 
 1   shoulder_lift 
 2      elbow_flex 
 3      wrist_flex 
 4      wrist_roll 
 5         gripper 
 6  red_cube_joint 
 7  red_cube_joint 
 8  red_cube_joint 
 9  red_cube_joint 
10  red_cube_joint 
11  red_cube_joint 
12  red_cube_joint 
13 blue_body_joint 
14 blue_body_joint 
15 blue_body_joint 
16 blue_body_joint 
17 blue_body_joint 
18 blue_body_joint 
19 blue_body_joint 
"""

class SO101Task(base.Task):
    """ Base Class for all S0101 Tasks
    """
    ARM_DOF = 5
    GRIPPER_DOF = 2
    
    def __init__(self, random=None, observation_width=640, observation_height=480):
        self.observation_width = observation_width
        self.observation_height = observation_height
        super().__init__(random=random)

    def before_step(self, action, physics) -> None:
        action = action.copy()
        left_arm_action = action[:self.ARM_DOF+1]
        env_action = unnormalize_so101(left_arm_action)
        
        physics.data.ctrl[:6] = env_action
        
        super().before_step(env_action, physics)
    
    def initialize_episode(self, physics):
        return super().initialize_episode(physics)
    
    @staticmethod
    def get_qpos(physics) -> np.array:
        """ Get positions for arm and gripper
        """
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:SO101Task.ARM_DOF + SO101Task.GRIPPER_DOF]
        left_arm_qpos = left_qpos_raw[:SO101Task.ARM_DOF]
        left_gripper_qpos = [left_qpos_raw[SO101Task.ARM_DOF]]
        return np.concatenate([left_arm_qpos, left_gripper_qpos])

    @staticmethod
    def get_qvel(physics) -> np.array:
        """ Get velocities for arm and gripper
        """
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:SO101Task.ARM_DOF + SO101Task.GRIPPER_DOF]
        left_arm_qvel = left_qvel_raw[:SO101Task.ARM_DOF]
        left_gripper_qvel = [left_qvel_raw[SO101Task.ARM_DOF]]
        return np.concatenate([left_arm_qvel, left_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        """ Get environment state
        """
        raise NotImplementedError

    def _precompute_bin_aabb(self, physics) -> None:
        """ Precompute bin data
        Assumes there is a left and/or right bin.
        """
        # Compute left bin data
        try:
            site_id = physics.model.site("left_bin_center").id
            if site_id is None:
                raise ValueError
            
            center = physics.data.site_xpos[site_id].copy()
            self.left_bin_center = center
            hw = 0.06  # half-width in xy  (edit to match XML)
            h = 0.03  # inner height      (edit to match XML)
            self.left_bin_min = center + np.array([-hw, -hw, 0.0])
            self.left_bin_max = center + np.array([hw, hw, h])
            self.left_bin_center = center
            self.left_bin_radius = hw  # for bonus
        except:
            print("could not precompute left bin data")

        # Compute right bin data
        try:
            site_id = physics.model.site("right_bin_center").id
            if site_id is None:
                raise ValueError
            
            center = physics.data.site_xpos[site_id].copy()
            self.right_bin_center = center
            hw = 0.06  # half-width in xy  (edit to match XML)
            h = 0.03  # inner height      (edit to match XML)
            self.right_bin_min = center + np.array([-hw, -hw, 0.0])
            self.right_bin_max = center + np.array([hw, hw, h])
            self.right_bin_center = center
            self.right_bin_radius = hw  # for bonus
        except:
            print("could not precompute right bin data")

        self.red_cube_half = 0.01  # edge/2  (match cube size)
        self.blue_cube_half = 0.01  # edge/2  (match cube size)

    def _cube_over_bin(self, target_cube_pos, target_bin:str) -> bool:
        """ Detects if the target cube is over the target bin

        Args:
            target_cube_pos: Current position of the target cube
            target_bin: Name of the target bin ("lef_bin"/"right_bin")

        Returns:
            True if target cube is over the target bin, False otherwise
        """

        if target_bin == "left_bin":
            bin_min = self.left_bin_min
            bin_max = self.left_bin_max
        elif target_bin == "right_bin":
            bin_min = self.right_bin_min
            bin_max = self.right_bin_max
        else:
            print(f"Invalid target_bin: {target_bin}")
            raise ValueError

        cube_over_bin =  (bin_min[0] < target_cube_pos[0] < bin_max[0]) and \
                            (bin_min[1] < target_cube_pos[1] < bin_max[1])

        return cube_over_bin

    def _cube_inside_bin(self, target_cube, target_cube_pos, target_bin:str) -> bool:
        """ Detects if the target cube is inside the target bin

        Args:
            target_cube: Name of the target cube ("red_cube"/"blue_cube")
            target_cube_pos: Current position of the target cube
            target_bin: Name of the target bin ("lef_bin"/"right_bin")

        Returns:
            True if target cube is in the target bin, False otherwise
        """
        if target_cube == "red_cube":
            cube_half = self.red_cube_half
        elif target_cube == "blue_cube":
            cube_half = self.blue_cube_half
        else:
            print(f"Invalid target_cube: {target_cube}")
            raise ValueError

        if target_bin == "left_bin":
            bin_min = self.left_bin_min
            bin_max = self.left_bin_max
        elif target_bin == "right_bin":
            bin_min = self.right_bin_min
            bin_max = self.right_bin_max
        else:
            print(f"Invalid target_bin: {target_bin}")
            raise ValueError

        lower = target_cube_pos - cube_half
        upper = target_cube_pos + cube_half
        return np.all(lower > bin_min) and np.all(upper < bin_max)

    def get_observation(self, physics) -> collections.OrderedDict:
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)

        obs["images"] = {}
        obs["images"]["workspace_cam"] = physics.render(
            height=self.observation_height,
            width=self.observation_width,
            camera_id="workspace_cam",
        )
        obs["images"]["wrist_cam"] = physics.render(
            height=self.observation_height,
            width=self.observation_width,
            camera_id="wrist_cam",
        )

        self._precompute_bin_aabb(physics)
        self._get_contact_pairs(physics)
        id_red_cube_site = physics.model.site("red_cube_site").id
        red_cube_pos = physics.data.site_xpos[id_red_cube_site]
        id_blue_cube_site = physics.model.site("blue_cube_site").id
        blue_cube_pos = physics.data.site_xpos[id_blue_cube_site]

        id_ee_site = physics.model.site("ee_site").id
        ee_pos = physics.data.site_xpos[id_ee_site]

        obs["blue_cube_position"] = blue_cube_pos.astype(np.float32)
        obs["red_cube_position"] = red_cube_pos.astype(np.float32)
        obs["left_bin_position"] = self.left_bin_center.astype(np.float32)  
        obs["right_bin_position"] = self.right_bin_center.astype(np.float32)  
        
        obs["ee_position"] = ee_pos.astype(np.float32)  
        return obs

    def get_reward(self, physics):
        raise NotImplementedError
    
    def get_cube_positions(self, physics) -> list[np.float32, np.float32]:
        """ Returns the position for the red and blue cube

        Returns:
            [red_cube_pos, blue_cube_pos]
        """
        id_red_cube_site = physics.model.site("red_cube_site").id
        red_cube_pos = physics.data.site_xpos[id_red_cube_site]
        red_cube_pos = red_cube_pos.astype(np.float32)

        id_blue_cube_site = physics.model.site("blue_cube_site").id
        blue_cube_pos = physics.data.site_xpos[id_blue_cube_site]
        blue_cube_pos = blue_cube_pos.astype(np.float32)

        return [red_cube_pos, blue_cube_pos]
    
    def get_ee_position(self, physics):
        id_ee_site = physics.model.site("ee_site").id
        ee_pos = physics.data.site_xpos[id_ee_site]
        return ee_pos
    
    def get_fingertip_geoms(self):
        """ Concatenates the names for all fingertip geoms in the SO101

        Returns:
            Set of strings of geom names as a constant
        """
        FIXED_FINGER_GEOMS = {f"fixed_jaw_{i}" for i in range(1, 3)}
        MOVING_FINGER_GEOMS = {f"moving_jaw_1"}
        FINGERTIP_GEOMS =  MOVING_FINGER_GEOMS | FIXED_FINGER_GEOMS
        return FINGERTIP_GEOMS
    
    def _get_contact_pairs(self, physics) -> list[str]:
        """ Returns all contact pairs in environment as a list of strings

        Returns:
            list[(geom_1, geom_2), (geom_1, geom3), ...]
        """
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        self.contact_pairs = all_contact_pairs
    
    def search_contact_pairs(self, target_geom1, target_geom2):

        contact = any(
            (g1 in target_geom1 and g2 == target_geom2) or
            (g2 in target_geom1 and g1 == target_geom2)
            for (g1, g2) in self.contact_pairs
        )

        return contact

class SO101TouchCubeTask(SO101Task):
    """ Manipulator will learn to touch cube

    "Actions are normalized to [-1, 1] range. Observations are not, to be used with VecNormalize
    """
    def __init__(self, random=None, observation_width=640, observation_height=480):
        super().__init__(
            random=random,
            observation_width=observation_width,
            observation_height=observation_height,
        )
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:6] = SO101_START_ARM_POSE
            np.copyto(physics.data.ctrl, SO101_START_ARM_POSE)
            assert BOX_POSE[0] is not None
            assert BOX_POSE[1] is not None
            physics.named.data.qpos[6:13] = BOX_POSE[0]
            physics.named.data.qpos[13:] = BOX_POSE[1]

            
            # print(f"{BOX_POSE=}")
        return super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[6:]
        return env_state

    def get_reward(self, physics):
        self._precompute_bin_aabb(physics)
        id_cube_site = physics.model.site("red_cube_site").id
        cube_pos = physics.data.site_xpos[id_cube_site]

        id_ee_site = physics.model.site("ee_site").id
        ee_pos = physics.data.site_xpos[id_ee_site]
        ee_cube_dist = np.linalg.norm(ee_pos - cube_pos)

        CUBE_GEOM = "red_cube"
        FIXED_FINGER_GEOMS = {f"fixed_jaw_{i}" for i in range(1, 3)}
        MOVING_FINGER_GEOMS = {f"moving_jaw_1"}
        FINGERTIP_GEOMS =  MOVING_FINGER_GEOMS | FIXED_FINGER_GEOMS

        # return whether gripper is touching the red cube
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_gripper = any(
            (g1 in FINGERTIP_GEOMS and g2 == CUBE_GEOM)
            or (g2 in FINGERTIP_GEOMS and g1 == CUBE_GEOM)
            for (g1, g2) in all_contact_pairs
        )
        
        reward = 0.0
        # Multi-stage distance rewards (smoother progression)
        if ee_cube_dist < 0.7:
            reward = max(reward, 0.1 * (1 - ee_cube_dist / 0.7))
        if ee_cube_dist < 0.5:
            reward = max(reward, 0.2 * (1 - ee_cube_dist / 0.5))
        if ee_cube_dist < 0.3:
            reward = max(reward, 0.5 * (1 - ee_cube_dist / 0.3))
        if ee_cube_dist < 0.1:  # NEW: bridge the gap
            reward = max(reward, 1.0 * (1 - ee_cube_dist / 0.1))
        if ee_cube_dist < 0.05:
            reward = max(reward, 2.0 * (1 - ee_cube_dist / 0.05))

        # Add contact bonus (already have the code!)
        if touch_gripper:
            reward += 1.0  # Big bonus for actually touching
            
        success = touch_gripper and ee_cube_dist < 0.07
        if success:
            print("SUCCESS!")
            return self.max_reward

        reward -= 0.2
        return reward

class SO101SortingTask(SO101Task):
    """
    dm_control Task for SO101. 
    Handles physics initialization, action application, and raw observations.
    """
    ARM_DOF = 5
    GRIPPER_DOF = 2

    def __init__(self, random=None, observation_width=640, observation_height=480):
        super().__init__(
            observation_width = observation_width,
            observation_height = observation_height
        )
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # Apply the randomized cube poses to the freejoints in MuJoCo
        # BOX_POSE[0] should be red cube, BOX_POSE[1] should be blue cube
        with physics.reset_context():
            physics.named.data.qpos[:6] = SO101_START_ARM_POSE
            np.copyto(physics.data.ctrl, SO101_START_ARM_POSE)

            assert BOX_POSE[0] is not None
            assert BOX_POSE[1] is not None
            physics.named.data.qpos[6:13] = BOX_POSE[0]
            physics.named.data.qpos[13:] = BOX_POSE[1]

        return super().initialize_episode(physics)
    
    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[6:]
        return env_state

    def get_reward(self, physics):
        """Calculates the scalar reward based on site distances."""
        self._precompute_bin_aabb(physics)
        self._get_contact_pairs(physics)

        cube_positions = self.get_cube_positions(physics)
        red_cube_pos = cube_positions[0]
        blue_cube_pos = cube_positions[1]
        ee_pos = self.get_ee_position(physics)

        ee_dist_red_cube = np.linalg.norm(ee_pos - red_cube_pos)
        ee_dist_blue_cube = np.linalg.norm(ee_pos - blue_cube_pos)

        FINGERTIP_GEOMS = self.get_fingertip_geoms()

        red_cube_gripper_contact = self.search_contact_pairs(FINGERTIP_GEOMS, RED_CUBE_GEOM)
        blue_cube_gripper_contact = self.search_contact_pairs(FINGERTIP_GEOMS, BLUE_CUBE_GEOM)

        red_cube_over_bin = self._cube_over_bin(red_cube_pos, "left_bin")
        blue_cube_over_bin = self._cube_over_bin(blue_cube_pos, "right_bin")

        red_cube_inside_bin = self._cube_inside_bin("red_cube", red_cube_pos, "left_bin")
        blue_cube_inside_bin = self._cube_inside_bin("blue_cube", blue_cube_pos, "right_bin")

        red_cube_released = red_cube_inside_bin and (not red_cube_gripper_contact)
        blue_cube_released = blue_cube_inside_bin and (not blue_cube_gripper_contact)
        
        red_cube_table_contact = self.search_contact_pairs(RED_CUBE_GEOM, TABLE_GEOM)
        blue_cube_table_contact = self.search_contact_pairs(BLUE_CUBE_GEOM, TABLE_GEOM)

        reward = 0.0

        if not red_cube_released and not blue_cube_released:

            if red_cube_gripper_contact:
                reward = 1.0
            if blue_cube_gripper_contact:
                reward = 1.0
        
            # lifted cubes
            if red_cube_gripper_contact and not red_cube_table_contact:
                reward = 2.0
            if blue_cube_gripper_contact and not blue_cube_table_contact:
                reward = 2.0

            # cubes over bin
            if red_cube_over_bin:
                reward = 2.5
            if blue_cube_over_bin:
                reward = 2.5
        
            if red_cube_inside_bin:
                reward += 1
            if blue_cube_inside_bin:
                reward += 1

            if red_cube_released:
                reward += 1.25
            if blue_cube_released:
                reward += 1.25
        else: 
            reward = self.max_reward

        return reward

    # Required custom methods for your GoalEnv wrapper to call
    def get_red_cube_position(self, physics):
        return physics.named.data.site_xpos['red_cube_center'].copy()