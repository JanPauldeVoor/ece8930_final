import numpy as np
import time
from so101_env import SO101SimEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset

print("Initializing Simulation Environment...")
env = SO101SimEnv(render_mode="human")
timestamp = int(time.time())
repo_id = f"local/so101_sim_{timestamp}"

print(f"Creating new dataset: {repo_id}")

# --- Initialize the LeRobot Dataset ---
dataset = LeRobotDataset.create(
    repo_id=repo_id, 
    fps=30,
    features={
        "observation.state": {"dtype": "float32", "shape": (6,)},
        "action": {"dtype": "float32", "shape": (6,)},
        "observation.images.workspace": {"dtype": "image", "shape": (480, 640, 3)},
    }
)

print("Starting data collection...")
obs, _ = env.reset()

# Sweeps the first joint (base) back and forth
num_frames = 150 # 5 seconds of data at 30fps
target_action = np.zeros(6, dtype=np.float32)

for frame in range(num_frames):
    # Create a sine wave motion for the base joint (index 0)
    target_action[0] = np.sin(frame / 20.0) * 1.5 
    
    # Send the action to the simulation
    next_obs, reward, terminated, truncated, info = env.step(target_action)
    
    # Record the frame into the LeRobot Dataset
    dataset.add_frame({
        "observation.state": obs["observation.state"],
        "action": target_action,
        "observation.images.workspace": obs["observation.images.workspace"]
    }, task="Wave the robot arm")
    
    obs = next_obs
    print(f"Recorded frame {frame+1}/{num_frames}")

# Save the episode and consolidate the dataset
dataset.save_episode(task="Wave the robot arm")
dataset.consolidate()
env.close()

print(f"\nDataset saved to ~/.cache/huggingface/lerobot/{repo_id}")