import pybullet as pb
import pybullet_data
import numpy as np
import cv2
import os
import time

from utils.sim.camera_rendering import render_d435

os.makedirs("tasks/hw3_task1/calibration_images", exist_ok=True)
os.makedirs("assets", exist_ok=True)

pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.loadURDF("plane.urdf")
pb.loadURDF("table/table.urdf", basePosition=[0, 0.5, 0])

# Load checkerboard
board_id = pb.loadURDF("assets/checkerboard.urdf", basePosition=[0, 0.5, 0.7])


print("Starting procedural calibration capture...")

# --- 3. Data Collection Loop ---
for i in range(20):
    rand_x = np.random.uniform(-0.1, 0.1)
    rand_y = np.random.uniform(0.45, 0.50)
    rand_z = np.random.uniform(0.65, 0.70)
    
    rand_roll = np.random.uniform(-0.1, 0.1)
    rand_pitch = np.random.uniform(-0.1, 0.1)
    rand_yaw = np.random.uniform(-0.1, 0.1)
    
    quat = pb.getQuaternionFromEuler([rand_roll, rand_pitch, rand_yaw])
    pb.resetBasePositionAndOrientation(board_id, [rand_x, rand_y, rand_z], quat)
    
    pb.stepSimulation()
    time.sleep(0.1)
    
    img = render_d435(pb)
    cv2.imwrite(f"tasks/hw3_task1/calibration_images/calib_{i:02d}.png", img)

pb.disconnect()
print("Data collection complete.")