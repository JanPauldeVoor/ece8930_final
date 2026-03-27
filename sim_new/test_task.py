# The goal of this file is to:
# - instantiate the environment
# - step with random actions
# - render the video of the actual "pixels" but could also be from rendered images.
import imageio
import gymnasium as gym
import numpy as np
import gym_so101

env = gym.make(
    "gym_so101/SO101TouchCube",
    obs_type="so101_pixels_agent_pos", 
    observation_width=640,
    observation_height=480,
)
observation, info = env.reset()
frames = []

for _ in range(5):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    image_array = observation["observation.images.workspace_cam"]
    render_frame = np.transpose(image_array, (1, 2, 0))
    frames.append(render_frame)

    if terminated or truncated:
        observation, info = env.reset()


env.close()
imageio.mimsave("outputs/example.mp4", np.stack(frames), fps=25)