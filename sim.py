import gymnasium as gym
import pybullet as pb
import pybullet_data
import numpy as np
import pkgutil
import cv2
import time
import os

from utils.sim.create_objects import *
from utils.sim.camera_rendering import render_camera
from utils.sim.environment_setup import environment_setup

# Setup environment
pb, plane_id, table_id, so101_id, _ = environment_setup()

# Create Left (Red Target) and Right (Blue Target) Bins
left_bin = create_bin(pb, [-0.3, 0.6, 0.63], [1, 0.5, 0.5, 1])  # Light Red
right_bin = create_bin(pb, [0.3, 0.6, 0.63], [0.5, 0.5, 1, 1])  # Light Blue

# Create Target Objects
red_block = create_block(pb, [-0.1, 0.45, 0.65], [1, 0, 0, 1]) # Red
blue_block = create_block(pb, [0.1, 0.45, 0.65], [0, 0, 1, 1]) # Blue

# Update this index based on the actual end-effector link in the SO-101 URDF
end_effector_link_index = 6 

print("Starting simulation loop. Press Ctrl+C in the terminal to exit.")

# --- 6. Main Simulation Loop ---
step_counter = 0

try:
    while True:
        pb.stepSimulation()

        # Only render cameras every 8 steps (240Hz / 8 = 30Hz)
        if step_counter % 8 == 0:
            # Workspace Camera (Intel D435)
            d435_pos = [0, -0.2, 1.2]
            d435_target = [0, 0.5, 0.625] 
            d435_img = render_camera(pb, d435_pos, d435_target, [0, 0, 1], v_fov=42, width=640, height=480)
            cv2.imshow("D435 Workspace View", d435_img)
            
            # Wrist Camera (Intel D405)
            ee_state = pb.getLinkState(so101_id, end_effector_link_index)
            ee_pos = np.array(ee_state[0])
            ee_ori = ee_state[1]
            
            rot_matrix = np.array(pb.getMatrixFromQuaternion(ee_ori)).reshape(3, 3)
            cam_offset = rot_matrix.dot(np.array([0, -0.05, 0.05]))
            d405_pos = ee_pos + cam_offset
            d405_target = d405_pos + rot_matrix.dot(np.array([0, 0, 0.5]))
            
            try:
                d405_img = render_camera(pb, d405_pos, d405_target, [0, 0, 1], v_fov=58, width=640, height=480)
                cv2.imshow("D405 Wrist View", d405_img)
            except Exception:
                pass 
                
            cv2.waitKey(1)
        
        step_counter += 1
        time.sleep(1./240.)

except KeyboardInterrupt:
    print("Simulation stopped.")
    pb.disconnect()
    cv2.destroyAllWindows()