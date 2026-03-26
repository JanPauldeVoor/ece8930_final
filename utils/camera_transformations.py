import math
import numpy as np
import cv2

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