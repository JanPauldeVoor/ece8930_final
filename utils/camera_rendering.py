import pybullet as pb
import numpy as np
import cv2


def render_camera(pb, cam_pos, target_pos, up_vector, v_fov, width, height):
    """Renders synthetic camera images mimicking RealSense parameters."""
    view_matrix = pb.computeViewMatrix(cam_pos, target_pos, up_vector)
    proj_matrix = pb.computeProjectionMatrixFOV(v_fov, width/height, 0.01, 3.0)
    
    img_arr = pb.getCameraImage(width, height, view_matrix, proj_matrix, 
                                renderer=pb.ER_BULLET_HARDWARE_OPENGL)
    
    # Extract RGB, drop Alpha channel, and convert RGB to BGR for OpenCV
    rgb = np.reshape(img_arr[2], (height, width, 4))[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def render_d435(pb):
    cam_pos = [0, -0.2, 1.2]
    target_pos = [0, 0.5, 0.625]
    view_matrix = pb.computeViewMatrix(cam_pos, target_pos, [0, 0, 1])
    proj_matrix = pb.computeProjectionMatrixFOV(42, 640/480, 0.01, 3.0)
    
    img_arr = pb.getCameraImage(640, 480, view_matrix, proj_matrix, renderer=pb.ER_BULLET_HARDWARE_OPENGL)
    rgb = np.reshape(img_arr[2], (480, 640, 4))[..., :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def render_rgbd_camera(pb, cam_pos, target_pos, up_vector, v_fov, width, height, near_val=0.01, far_val=3.0):
    """Renders RGB and true metric Depth from PyBullet."""
    view_matrix = pb.computeViewMatrix(cam_pos, target_pos, up_vector)
    proj_matrix = pb.computeProjectionMatrixFOV(v_fov, width/height, near_val, far_val)
    
    img_arr = pb.getCameraImage(width, height, view_matrix, proj_matrix, 
                                renderer=pb.ER_BULLET_HARDWARE_OPENGL)
    
    # Extract and format RGB
    rgb = np.reshape(img_arr[2], (height, width, 4))[..., :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Extract Depth Buffer (0.0 to 1.0)
    depth_buffer = np.reshape(img_arr[3], (height, width))
    
    # Convert OpenGL non-linear depth to True Metric Depth (meters)
    true_depth = (far_val * near_val) / (far_val - (far_val - near_val) * depth_buffer)
    
    return bgr, true_depth