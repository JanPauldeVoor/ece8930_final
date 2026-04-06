import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces

from gym_so101.constants import (
    SO101_ACTIONS,
    ASSETS_DIR,
    DT,
    SO101_JOINTS,
    bin_min,
    bin_max,
)
from gym_so101.tasks.single import (
    BOX_POSE,
    SO101SortingTask,
    SO101TouchCubeTask
)

def sample_so101_box_pose(num_boxes=2, min_dist=0.04,seed=None):
    x_range = [0.15, 0.2]
    y_range = [-.1, 0.1]
    z_range = [0.05, 0.05]

    rng = np.random.RandomState(seed)
    ranges = np.vstack([x_range, y_range, z_range])
    
    poses = []
    while len(poses) < num_boxes:
        candidate_pos = rng.uniform(ranges[:, 0], ranges[:, 1])
        
        # Check distance against already accepted poses to avoid overlap
        collision = False
        for p in poses:
            # We only care about XY distance since they share the same Z height
            if np.linalg.norm(candidate_pos[:2] - p[:2]) < min_dist:
                collision = True
                break
        
        if not collision:
            poses.append(candidate_pos)

    cube_quat = np.array([1, 0, 0, 0])
    
    # Return a list of complete pose arrays [x, y, z, qw, qx, qy, qz]
    return [np.concatenate([pos, cube_quat]) for pos in poses]

class SO101Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task,
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        self._env = self._make_env_task(self.task)

        if self.obs_type == "so101_pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "observation.images.workspace_cam": spaces.Box(
                        low=0,
                        high=255,
                        shape=(3, self.observation_height, self.observation_width),
                        dtype=np.uint8,
                    ),
                    "observation.images.wrist_cam": spaces.Box(
                        low=0,
                        high=255,
                        shape=(3, self.observation_height, self.observation_width),
                        dtype=np.uint8,
                    ),
                    "observation.state": spaces.Box(
                        low=-10.0,
                        high=10.0,
                        shape=(len(SO101_JOINTS),),
                        dtype=np.float32,
                    ),
                }
            )
        elif self.obs_type == "so101_state":
            self.observation_space = spaces.Box(
                low=-100.0,
                high=100.0,
                shape=(len(SO101_JOINTS) + 3 * 3,),  # joints + box + bin + ee
                dtype=np.float32,
            )

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(len(SO101_ACTIONS),), dtype=np.float32
        )

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        assert self.render_mode == "rgb_array"
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        image = self._env.physics.render(height=height, width=width, camera_id="workspace_cam")
        return image

    def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        if task_name == "SO101Sorting":
            xml_path = ASSETS_DIR / "scene.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = SO101SortingTask(
                observation_width=self.observation_width,
                observation_height=self.observation_height,
            )
        elif task_name == "so101_touch_cube":
            xml_path = ASSETS_DIR / "scene.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = SO101TouchCubeTask(
                observation_width=self.observation_width,
                observation_height=self.observation_height,
            )
        else:
            raise NotImplementedError(task_name)

        env = control.Environment(
            physics,
            task,
            time_limit,
            control_timestep=DT,
            n_sub_steps=None,
            flat_observation=False,
        )
        return env

    def _format_raw_obs(self, raw_obs):
        if self.obs_type == "so101_pixels_agent_pos":
            workspace_img = raw_obs["images"]["workspace_cam"].copy()
            workspace_img = np.transpose(workspace_img, (2, 0, 1))

            # Process Wrist Cam
            wrist_img = raw_obs["images"]["wrist_cam"].copy()
            wrist_img = np.transpose(wrist_img, (2, 0, 1))

            obs = {
                "observation.images.workspace_cam": workspace_img,
                "observation.images.wrist_cam": wrist_img,
                "observation.state": raw_obs["qpos"].astype(np.float32), 
            }
            return obs
        elif self.obs_type == "so101_state":
            obs = np.concatenate(
            [
                raw_obs["red_block_position"].astype(np.float32), 
                raw_obs["ee_position"].astype(np.float32),        
                raw_obs["qpos"].astype(np.float32), 
            ]
        )
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        if self.task == "SO101Sorting":
            poses = sample_so101_box_pose(num_boxes=2, seed=seed)
            BOX_POSE[0] = poses[0]
            BOX_POSE[1] = poses[1]
        elif self.task == "so101_touch_cube":
            poses = sample_so101_box_pose(num_boxes=2, seed=seed)
            BOX_POSE[0] = poses[0]
            BOX_POSE[1] = poses[1]
        else:
            raise ValueError(self.task)

        raw_obs = self._env.reset()

        observation = self._format_raw_obs(raw_obs.observation)

        info = {"is_success": False}
        return observation, info

    def step(self, action):
        assert action.ndim == 1
        _, reward, _, raw_obs = self._env.step(action)
        terminated = is_success = reward == 4

        info = {"is_success": is_success}

        observation = self._format_raw_obs(raw_obs)

        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        pass


class SO100GoalEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode="rgb_array",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
    ):
        super().__init__()
        self.max_episode_steps = 300
        self.current_step = 0
        self.total_steps = 0

        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        task = SO101SortingTask(
            observation_width=self.observation_width,
            observation_height=self.observation_height,
        )
        self.task = task
        self._env = self._make_env_task(self.task)

        goal_dim = 3  # x, y, z coordinates of the goal

        pixels_flat_size = observation_height * observation_width * 3
        agent_pos_size = len(SO101_JOINTS)
        obs_size = pixels_flat_size + agent_pos_size

        obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # GoalEnv observation space structure
        self.observation_space = spaces.Dict(
            {
                "observation": obs_space,
                "achieved_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
                ),
                "desired_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
                ),
            }
        )

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(len(SO101_ACTIONS),), dtype=np.float32
        )

        # Goal sampling parameters
        self.bin_goal_space = spaces.Box(
            low=np.array([bin_min[0] + 0.005, bin_min[1] + 0.005, 0.01]),
            high=np.array([bin_max[0] - 0.005, bin_max[1] - 0.005, 0.05]),
            dtype=np.float32,
        )

        # Success threshold
        self.distance_threshold = 0.01

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        assert self.render_mode == "rgb_array"
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        image = self._env.physics.render(height=height, width=width, camera_id="workspace_cam")
        return image

    def _flatten_observation(self, base_obs):
        pixels_flat = base_obs["observation.images.workspace_cam"].flatten().astype(np.float32) / 255.0
        agent_pos = base_obs["observation.state"].astype(np.float32)
        return np.concatenate([pixels_flat, agent_pos])

    def _make_env_task(self, task):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        xml_path = ASSETS_DIR / "scene.xml"
        physics = mujoco.Physics.from_xml_path(str(xml_path))
        task = SO101TouchCubeTask(
            observation_width=self.observation_width,
            observation_height=self.observation_height,
        )

        env = control.Environment(
            physics,
            task,
            time_limit,
            control_timestep=DT,
            n_sub_steps=None,
            flat_observation=False,
        )
        return env

    def _format_raw_obs(self, raw_obs):
        img_workspace = raw_obs["images"]["workspace_cam"].copy()
            
        # 2. Transpose from (H, W, C) to (C, H, W)
        img_workspace = np.transpose(img_workspace, (2, 0, 1))
        
        obs = {
            "observation.images.workspace_cam": img_workspace, # Must be uint8
            "observation.state": raw_obs["qpos"].astype(np.float32), 
        }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        # TODO(rcadene): how to seed the env?
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)
        self.box_pose = sample_so101_box_pose(seed)
        BOX_POSE[0] = self.box_pose

        raw_obs = self._env.reset()
        self.goal = self._sample_goal()

        base_obs = self._format_raw_obs(raw_obs.observation)
        observation = self._get_goal_obs(base_obs)

        info = {"is_success": False}
        return observation, info

    def _sample_goal(self):
        """Sample a goal position within the bin"""
        if self.total_steps < 5000:
            self.lifted_goal_space = spaces.Box(
                low=np.array([self.box_pose[0] - 0.03, self.box_pose[1] - 0.03, 0.01]),
                high=np.array([self.box_pose[0] + 0.03, self.box_pose[1] + 0.03, 0.05]),
                dtype=np.float32,
            )
            return self.lifted_goal_space.sample()
        if self.total_steps == 5000:
            print("Switching to bin goal sampling")

        return self.bin_goal_space.sample()

    def _extract_achieved_goal(self):
        """Extract current cube position from physics"""
        cube_pos = self._env.task.get_red_cube_position(self._env.physics)
        return cube_pos.copy().astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute sparse reward for HER"""
        distance = np.linalg.norm(achieved_goal - desired_goal)

        # Handle batch inputs (when HER calls this)
        if achieved_goal.ndim > 1:
            distances = np.linalg.norm(achieved_goal - desired_goal, axis=1)
            rewards = np.where(distances < self.distance_threshold, 0.0, -1.0)
            return rewards.astype(np.float32)
        else:
            # Handle single inputs (when environment calls this)
            distance = np.linalg.norm(achieved_goal - desired_goal)
            return 0.0 if distance < self.distance_threshold else -1.0

    def _is_success(self, achieved_goal, desired_goal):
        """Check if goal is achieved"""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance < self.distance_threshold

    def _get_goal_obs(self, base_obs):
        """Convert base observation to goal-structured observation"""
        # Extract achieved goal (current cube position)
        achieved_goal = self._extract_achieved_goal()

        # Structure observation for GoalEnv
        return {
            "observation": self._flatten_observation(base_obs),
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal.copy(),
        }

    def step(self, action):
        if self.current_step % 10 == 0:
            print(f"Step {self.current_step + 1}/{self.max_episode_steps}")
        assert action.ndim == 1
        _, reward, _, raw_obs = self._env.step(action)
        is_success = False

        info = {"is_success": is_success}

        base_obs = self._format_raw_obs(raw_obs)
        observation = self._get_goal_obs(base_obs)

        reward = self.compute_reward(
            observation["achieved_goal"], observation["desired_goal"], info
        )

        # Check if goal is achieved
        success = self._is_success(
            observation["achieved_goal"], observation["desired_goal"]
        )
        info["is_success"] = success

        # Increment step counter
        self.current_step += 1
        self.total_steps += 1

        truncated = False
        # Check if max steps reached
        if self.current_step >= self.max_episode_steps:
            truncated = True
            print("Reached max steps, truncating episode.")
            info["TimeLimit.truncated"] = True

        terminated = success
        return observation, reward, terminated, truncated, info

    def close(self):
        pass