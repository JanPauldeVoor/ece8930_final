import pyrealsense2 as rs
import numpy as np
import cv2
import os

from utils.realsense_camera import init_realsense

# Intrinsic and Distortion arrays
K_FILE = "real_world/hw3_task1/k.npy"
DIST_FILE = "real_world/hw3_task1/dist_coeffs.npy"

# Optional directory to save images to
IMAGE_DIR = "real_world/hw3_task1/assets"
os.makedirs(IMAGE_DIR, exist_ok=True)

def find_orange_block(bgr_image):
    """Finds the (u, v) pixel coordinate of the blue block using HSV masking."""
    # Convert to HSV color space for robust lighting invariance
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    
    # Define bounds for the blue block (PyBullet's pure blue [0, 0, 1, 1])
    lower_orange = np.array([10, 100, 20])
    upper_orange = np.array([25, 255, 255])
    
    # Create a binary mask (white where blue is, black everywhere else)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
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

def pixel_to_camera_frame(u, v, Z_c, f_x, f_y, c_x, c_y):
    """Unprojects a pixel and depth into OpenCV Camera 3D space."""
    X_c = (u - c_x) * Z_c / f_x
    Y_c = (v - c_y) * Z_c / f_y
    return np.array([X_c, Y_c, Z_c, 1.0]) # Homogenous coordinate

def camera_to_world_frame(Z_c, u, c_x,  f_x, v, c_y, f_y):
    """Transforms camera-relative coordinates to world coordinates."""
    X_c = ((u-c_x)*Z_c)/f_x
    Y_c = ((v-c_y)*Z_c)/f_y

    return [X_c, Y_c]


pipeline, config = init_realsense(rgb_stream=True, depth_stream=True)
if pipeline == None or config == None:
    print("ERROR: Could not initalize camera")
    exit(0)

# Start streaming
pipeline.start(config)

k = np.load(K_FILE)
f_x = k[0][0]
f_y = k[1][1]
c_x = k[0][2]
c_y = k[1][2]

pictures = 0 
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        u,v = find_orange_block(color_image)
        if u is not None and v is not None:
            # Read the Depth
            Z_c = depth_image[v, u]
            
            # Unproject to Camera Space
            cam_coords = pixel_to_camera_frame(u, v, Z_c, f_x, f_y, c_x, c_y)
            
            # # Transform to World Space
            world_x, world_y = camera_to_world_frame(Z_c, u, c_x, f_x, v, c_y, f_y)
            
            # # Print the final target coordinate
            print(f"World Target Coordinates: X={world_x:.3f}, Y={world_y:.3f}, Z={Z_c:.3f}")
            
            # Draw  red marker to the calculated coordinate to verify
            cv2.drawMarker(color_image, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)  

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)

        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(f"{IMAGE_DIR}/block_img_{pictures:02d}.png", color_image)
            pictures += 1

        elif key == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()