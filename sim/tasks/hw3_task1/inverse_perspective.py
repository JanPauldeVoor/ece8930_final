import pybullet as pb
import pybullet_data
import numpy as np
import pkgutil
import cv2
import math
import time
import os

from utils.sim.camera_rendering import render_camera,render_rgbd_camera
from utils.sim.create_objects import create_block

def find_blue_block(bgr_image):
    """Finds the (u, v) pixel coordinate of the blue block using HSV masking."""
    # Convert to HSV color space for robust lighting invariance
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    
    # Define bounds for the blue block (PyBullet's pure blue [0, 0, 1, 1])
    lower_blue = np.array([110, 100, 100])
    upper_blue = np.array([130, 255, 255])
    
    # Create a binary mask (white where blue is, black everywhere else)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find the contours of the masked object
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Grab the largest contour (in case of noise)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the centroid (center pixel) of the contour using image moments
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            u = int(M["m10"] / M["m00"]) # X pixel coordinate
            v = int(M["m01"] / M["m00"]) # Y pixel coordinate
            return u, v
            
    return None, None

def get_intrinsics_from_fov(v_fov, width, height):
    """Calculates focal lengths and optical center from FOV."""
    fov_rad = v_fov * math.pi / 180.0
    f_y = (height / 2.0) / math.tan(fov_rad / 2.0)
    f_x = f_y # Assuming square pixels
    c_x, c_y = width / 2.0, height / 2.0
    return f_x, f_y, c_x, c_y

def pixel_to_camera_frame(u, v, Z_c, f_x, f_y, c_x, c_y):
    """Unprojects a pixel and depth into OpenCV Camera 3D space."""
    X_c = (u - c_x) * Z_c / f_x
    Y_c = (v - c_y) * Z_c / f_y
    return np.array([X_c, Y_c, Z_c, 1.0]) # Homogenous coordinate

def camera_to_world_frame(cam_coords, view_matrix):
    """Transforms camera-relative coordinates to PyBullet world coordinates."""
    # PyBullet uses the OpenGL coordinate system (+Z is backward, +Y is up)
    # Must convert OpenCV coordinates to OpenGL before transforming
    gl_coords = np.array([cam_coords[0], -cam_coords[1], -cam_coords[2], 1.0])
    
    # PyBullet returns view_matrix as a 1D column-major array. 
    # We reshape it, transpose it to standard row-major, and invert it.
    view_mat_4x4 = np.array(view_matrix).reshape(4, 4).T
    inv_view_mat = np.linalg.inv(view_mat_4x4)
    
    # Multiply the inverted matrix by our coordinate to get World Space
    world_coords = inv_view_mat.dot(gl_coords)
    return world_coords[:3] # Return just [X, Y, Z]

# Initialize Simulation 
physicsClient = pb.connect(pb.GUI)

# Make sure there's hardware acceleration
egl = pkgutil.get_loader('eglRenderer')
if egl:
    plugin_id = pb.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
else:
    plugin_id = pb.loadPlugin("eglRendererPlugin")

if plugin_id >= 0:
    print("Hardware accelerated rendering (EGL) enabled successfully.")
else:
    print("Warning: EGL plugin failed to load. Defaulting to standard rendering.")

pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setGravity(0, 0, -9.81)

# Build Environment 
plane_id = pb.loadURDF("plane.urdf")
# The default PyBullet table is ~0.625m high
table_id = pb.loadURDF("table/table.urdf", basePosition=[0, 0.5, 0])

# Load SO-101 Arm
urdf_path = "assets/so101/so101.urdf"

so101_pos = [0, 0.1, 0.625] # Mounted on the edge of the table
so101_ori = pb.getQuaternionFromEuler([0, 0, 0])

if os.path.exists(urdf_path):
    so101_id = pb.loadURDF(urdf_path, so101_pos, so101_ori, useFixedBase=True)
    print("SO-101 Arm loaded successfully.")
else:
    print(f"WARNING: Could not find URDF at {urdf_path}.")
    print("Using a placeholder robotic arm (KUKA) for testing the environment...")
    so101_id = pb.loadURDF("kuka_iiwa/model.urdf", so101_pos, so101_ori, useFixedBase=True)


# Create Target Objects
red_block = create_block(pb, [-0.1, 0.45, 0.65], [1, 0, 0, 1]) # Red
blue_block = create_block(pb, [0.1, 0.45, 0.65], [0, 0, 1, 1]) # Blue


# Update this index based on the actual end-effector link in the SO-101 URDF
end_effector_link_index = 6 

print("Starting simulation loop. Press Ctrl+C in the terminal to exit.")

step_counter = 0

try:
    while True:
        pb.stepSimulation()
        
        # Only render cameras every 8 steps (240Hz / 8 = 30Hz)
        if step_counter % 8 == 0:
            # Workspace Camera (Intel D435)
            v_fov = 42
            width, height = 640, 480
            d435_pos = [0, -0.2, 1.2]
            d435_target = [0, 0.5, 0.625]
            
            bgr_img, depth_map = render_rgbd_camera(pb, d435_pos, d435_target, [0, 0, 1], v_fov, width, height)
            view_matrix = pb.computeViewMatrix(d435_pos, d435_target, [0, 0, 1])
            
            # Find the block
            u, v = find_blue_block(bgr_img)
            
            if u is not None and v is not None:
                # Read the Depth
                Z_c = depth_map[v, u]
                
                # Unproject to Camera Space
                f_x, f_y, c_x, c_y = get_intrinsics_from_fov(v_fov, width, height)
                cam_coords = pixel_to_camera_frame(u, v, Z_c, f_x, f_y, c_x, c_y)
                
                # Transform to World Space
                world_x, world_y, world_z = camera_to_world_frame(cam_coords, view_matrix)
                
                # Print the final target coordinate
                print(f"World Target Coordinates: X={cam_coords[0]:.3f}, Y={cam_coords[1]:.3f}, Z={cam_coords[2]:.3f}")
                
                # Draw  red marker to the calculated coordinate to verify
                cv2.drawMarker(bgr_img, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)  
            cv2.imshow("D435 Workspace View", bgr_img)

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