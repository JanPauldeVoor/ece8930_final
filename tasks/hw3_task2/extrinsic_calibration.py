import pybullet as pb
import pybullet_data
from pupil_apriltags import Detector
import numpy as np
import pkgutil
import cv2
import math
import time
import os

from utils.environment_setup import environment_setup
from utils.create_objects import create_block
from utils.camera_rendering import *

def create_tag_mesh(filepath="assets/tag_mesh.obj"):
    """Generates a mathematically perfect 100mm x 100mm 3D quad with explicit UV mapping."""
    os.makedirs("assets", exist_ok=True)
    with open(filepath, "w") as f:
        # Define the 4 corners of a 100mm square (±0.05m from center)
        f.write("v -0.05 -0.05 0.0\n")  # Vertex 1: Bottom-Left
        f.write("v  0.05 -0.05 0.0\n")  # Vertex 2: Bottom-Right
        f.write("v  0.05  0.05 0.0\n")  # Vertex 3: Top-Right
        f.write("v -0.05  0.05 0.0\n")  # Vertex 4: Top-Left
        
        # Define the UV texture coordinates (0 to 1) mapping directly to the image corners
        f.write("vt 0.0 0.0\n") # UV 1
        f.write("vt 1.0 0.0\n") # UV 2
        f.write("vt 1.0 1.0\n") # UV 3
        f.write("vt 0.0 1.0\n") # UV 4
        
        # Create the face linking the vertices and UVs together
        f.write("f 1/1 2/2 3/3 4/4\n")

def spawn_apriltags(pb):
    """Spawns four perfectly textured 100mm AprilTags at the corners."""
    # 1. Generate the custom mesh file
    create_tag_mesh()
    
    # 2. Tell PyBullet to use the explicit mesh instead of a procedural box
    tag_shape = pb.createVisualShape(
        shapeType=pb.GEOM_MESH, 
        fileName="assets/tag_mesh.obj",
        meshScale=[1, 1, 1]
    )
    
    # Load the 1024x1024 texture you generated earlier
    texture_id = pb.loadTexture("assets/apriltag_0.png")
    
    corners = {
        "Top-Left":     [-0.3, 0.7, 0.626],
        "Top-Right":    [ 0.3, 0.7, 0.626],
        "Bottom-Right": [ 0.3, 0.3, 0.626],
        "Bottom-Left":  [-0.3, 0.3, 0.626]
    }
    
    tag_ids = {}
    
    for name, pos in corners.items():
        ori = pb.getQuaternionFromEuler([0, 0, 0])
        
        body_id = pb.createMultiBody(
            baseMass=0, 
            baseVisualShapeIndex=tag_shape, 
            basePosition=pos, 
            baseOrientation=ori
        )
        # Apply the texture to the mesh
        pb.changeVisualShape(body_id, -1, textureUniqueId=texture_id)
        tag_ids[name] = body_id
        
    return tag_ids, corners

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
    # We must convert our OpenCV coordinates to OpenGL before transforming
    gl_coords = np.array([cam_coords[0], -cam_coords[1], -cam_coords[2], 1.0])
    
    # PyBullet returns view_matrix as a 1D column-major array. 
    # We reshape it, transpose it to standard row-major, and invert it.
    view_mat_4x4 = np.array(view_matrix).reshape(4, 4).T
    inv_view_mat = np.linalg.inv(view_mat_4x4)
    
    # Multiply the inverted matrix by our coordinate to get World Space
    world_coords = inv_view_mat.dot(gl_coords)
    return world_coords[:3] # Return just [X, Y, Z]


pb, plane_id, table_id, so101_id, _ = environment_setup()

tags, tag_positions = spawn_apriltags(pb)
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)



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
            
            # --- Camera 1: Workspace Camera (Intel D435) ---
            v_fov = 42
            width, height = 640, 480
            d435_pos = [0, -0.2, 1.2]
            d435_target = [0, 0.5, 0.625]
            bgr_img, _ = render_rgbd_camera(pb, d435_pos, d435_target, [0, 0, 1], 
                                            v_fov=42, width=640, height=480)
            
            # 2. Convert to grayscale (The detector requires a 1-channel image)
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

            # 2. Detect Tags
            # pupil_apriltags expects a grayscale 8-bit image
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
                

                print(f"Tag ID {det.tag_id} center at {det.center}")
            # Display the camera feed
            cv2.imshow("D435 AprilTag View", bgr_img)
            # --- Camera 2: Wrist Camera (Intel D405) ---
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