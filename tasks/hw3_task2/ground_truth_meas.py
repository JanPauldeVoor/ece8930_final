import pybullet as pb
import pybullet_data
from pupil_apriltags import Detector
import numpy as np
import pkgutil
import cv2
import math
import time
import os

from utils.camera_rendering import *
from utils.camera_transformations import *
from utils.create_objects import *
from utils.environment_setup import environment_setup

pb, plane_id, table_id, so101_id, _ = environment_setup()

# Create Target Objects
red_block = create_block(pb, [-0.1, 0.45, 0.65], [1, 0, 0, 1]) # Red
blue_block = create_block(pb, [0.1, 0.45, 0.65], [0, 0, 1, 1]) # Blue
tags, tag_positions = spawn_apriltags(pb)

at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)


# Update this index based on the actual end-effector link in the SO-101 URDF
end_effector_link_index = 6 


print("Starting simulation loop. Press Ctrl+C in the terminal to exit.")

# --- 6. Main Simulation Loop ---
step_counter = 0

init_x, init_y, init_z = 0.0, 0.4, 0.75

# Create GUI Sliders
sl_x = pb.addUserDebugParameter("Target X", -0.5, 0.5, init_x)
sl_y = pb.addUserDebugParameter("Target Y", 0.2, 0.8, init_y)
sl_z = pb.addUserDebugParameter("Target Z", 0.6, 1.0, init_z)
sl_roll  = pb.addUserDebugParameter("Target Roll", -3.14, 3.14, 0)
sl_pitch = pb.addUserDebugParameter("Target Pitch", 0, 3.14, 1.57) # Pointing down
sl_yaw   = pb.addUserDebugParameter("Target Yaw", -3.14, 3.14, 0)

# Create a 'Record' button
btn_record = pb.addUserDebugParameter("Record Corner Point", 1, 0, 0)
record_count = 0
object_points_base = []
motor_indices = []
for i in range(pb.getNumJoints(so101_id)):
    info = pb.getJointInfo(so101_id, i)
    joint_type = info[2]
    # Type 0 is REVOLUTE, Type 1 is PRISMATIC
    if joint_type in [pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC]:
        motor_indices.append(i)

print("\n--- MANUAL CALIBRATION MODE ---")
print("1. Use sliders to touch the 4 corners of the AprilTag black boundary.")
print("2. Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.")
print("3. Click 'Record Corner Point' in the GUI for each.")

try:
    while True:
        pb.stepSimulation()
        target_pos = [pb.readUserDebugParameter(sl_x), 
              pb.readUserDebugParameter(sl_y), 
              pb.readUserDebugParameter(sl_z)]

        target_orn = pb.getQuaternionFromEuler([pb.readUserDebugParameter(sl_roll), 
                                                pb.readUserDebugParameter(sl_pitch), 
                                                pb.readUserDebugParameter(sl_yaw)])

        # Inverse Kinematics
        # end_effector_link_index should be the index of your gripper tip
        joint_poses = pb.calculateInverseKinematics(so101_id, end_effector_link_index, 
                                            target_pos, target_orn)

        # Apply to Joints
        for i, motor_idx in enumerate(motor_indices):
            pb.setJointMotorControl2(so101_id, motor_idx, pb.POSITION_CONTROL, 
                             targetPosition=joint_poses[i])

        # Recording Button
        # readUserDebugParameter for a button returns the total number of times it was clicked
        if pb.readUserDebugParameter(btn_record) > record_count:
            record_count += 1
            
            # Get the current actual position of the gripper tip
            ee_state = pb.getLinkState(so101_id, end_effector_link_index)
            actual_xyz = ee_state[0]
            
            object_points_base.append(actual_xyz)
            print(f"Recorded Corner {record_count}/4: {np.round(actual_xyz, 4)}")
            
            if record_count == 4:
                obj_points_arr = np.array(object_points_base)
                np.save("object_points_base.npy", obj_points_arr)
                print("\nSUCCESS: All 4 points recorded and saved to 'object_points_base.npy'!")
       
        # Only render cameras every 8 steps (240Hz / 8 = 30Hz)
        if step_counter % 8 == 0:
            #  Workspace Camera (Intel D435)
            v_fov = 42
            width, height = 640, 480
            d435_pos = [0, -0.2, 1.2]
            d435_target = [0, 0.5, 0.625]
            bgr_img, _ = render_rgbd_camera(pb, d435_pos, d435_target, [0, 0, 1], 
                                            v_fov=42, width=640, height=480)
            
            # Convert to grayscale (The detector requires a 1-channel image)
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

            # Detect Tags
            detections = at_detector.detect(gray_img, estimate_tag_pose=False)
            
            for det in detections:
                # 'det.corners' is a 4x2 array of [u, v] coordinates
                # Order: Bottom-Left, Bottom-Right, Top-Right, Top-Left 
                # (Note: Pupil uses a different order than OpenCV's clockwise)
                corners = det.corners.astype(int)
                
                # Explicitly mark the Top-Left corner (index 3 in Pupil)
                cv2.circle(bgr_img, tuple(corners[0]), 5, (0, 0, 255), -1)
                cv2.circle(bgr_img, tuple(corners[1]), 5, (0, 0, 255), -1)
                cv2.circle(bgr_img, tuple(corners[2]), 5, (0, 0, 255), -1)
                cv2.circle(bgr_img, tuple(corners[3]), 5, (0, 0, 255), -1)
                
            # Display the camera feed
            cv2.imshow("D435 AprilTag View", bgr_img)

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