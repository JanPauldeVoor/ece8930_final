import pyrealsense2 as rs
import numpy as np
import cv2
import os

from utils.realsense_camera import init_realsense

# Load Calibration Files
K_FILE = "real_world/hw3_task1/k.npy"
T_MATRIX_FILE = "real_world/hw3_task2/t_base_d435.npy" # Adjust path as needed

def find_orange_block(bgr_image):
    """Finds the (u, v) pixel coordinate of the orange block using HSV masking."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([10, 100, 20])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])
            return u, v
    return None, None

def pixel_to_camera_frame(u, v, Z_c, K_matrix):
    """Task 1.3: Unprojects pixel to 3D Camera Space (P_D435)."""
    f_x, f_y = K_matrix[0, 0], K_matrix[1, 1]
    c_x, c_y = K_matrix[0, 2], K_matrix[1, 2]

    X_c = (u - c_x) * Z_c / f_x
    Y_c = (v - c_y) * Z_c / f_y
    return np.array([X_c, Y_c, Z_c, 1.0])

def camera_to_base_frame(P_D435, T_base_D435):
    """Task 3: Transforms Camera Space to Robot Base Space."""
    return np.dot(T_base_D435, P_D435)

def main():
    # Load matrices
    if not os.path.exists(K_FILE) or not os.path.exists(T_MATRIX_FILE):
        print("Error: Calibration files not found. Run Task 1 and Task 2 first.")
        return
        
    K_matrix = np.load(K_FILE)
    T_base_D435 = np.load(T_MATRIX_FILE)

    # Initialize Camera
    pipeline, config = init_realsense(rgb_stream=True, depth_stream=True)
    if pipeline is None:
        print("ERROR: Could not initialize camera")
        return

    pipeline.start(config)
    print("Streaming started. Looking for block...")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Find the block
            u, v = find_orange_block(color_image)
            
            if u is not None and v is not None:
                # Read Depth 
                Z_c = depth_frame.get_distance(u, v) 
                
                if Z_c > 0:
                    # Apply Intrinsic Matrix (Task 1.3)
                    P_D435 = pixel_to_camera_frame(u, v, Z_c, K_matrix)
                    
                    # Apply Extrinsic Matrix (Task 3)
                    P_base = camera_to_base_frame(P_D435, T_base_D435)
                    
                    print(f"Target found at Pixel ({u}, {v}) with depth {Z_c:.3f}m")
                    print(f" -> Robot Base Coord: X={P_base[0]:.3f}, Y={P_base[1]:.3f}, Z={P_base[2]:.3f}")
                    
                    cv2.drawMarker(color_image, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)  

            cv2.imshow('Perception Pipeline', color_image)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
