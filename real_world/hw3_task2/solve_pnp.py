import cv2
import numpy as np

def calculate_extrinsics(points_3d, points_2d, K_matrix, dist_coeffs):
    """
    Solves for the 4x4 transformation matrix mapping the camera to the robot base.
    """
    # 1. Solve PnP to get rotation and translation vectors
    # This gives us the pose of the Base Frame relative to the Camera Frame
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=points_3d,
        imagePoints=points_2d,
        cameraMatrix=K_matrix,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise RuntimeError("solvePnP failed to find a solution.")

    # 2. Convert the rotation vector into a 3x3 rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # 3. Construct the 4x4 homogeneous transformation matrix (T_D435_base)
    # This matrix converts points from the Base frame TO the Camera frame
    T_D435_base = np.eye(4)
    T_D435_base[:3, :3] = R
    T_D435_base[:3, 3] = tvec.flatten()

    # 4. Invert the matrix to get T_base_D435
    # This matrix converts points from the Camera frame TO the Base frame
    T_base_D435 = np.linalg.inv(T_D435_base)

    return T_base_D435

# --- Mock Test ---
# mock_3d_points = np.array([[0.5, 0.2, 0.1], [0.5, 0.12, 0.1], [0.42, 0.12, 0.1], [0.42, 0.2, 0.1]], dtype=np.float32)
# mock_2d_points = np.array([[300, 200], [400, 200], [400, 300], [300, 300]], dtype=np.float32)
# K = np.load("k.npy") # From your Task 1 output
# dist = np.load("dist_coeffs.npy")
# 
# T_matrix = calculate_extrinsics(mock_3d_points, mock_2d_points, K, dist)
# print("Final Transformation Matrix (T_base_D435):\n", T_matrix)

if __name__ == "__main__":
    # 1. Create some dummy 3D physical points (meters)
    mock_3d_points = np.array([
        [0.50, 0.20, 0.10], 
        [0.50, 0.12, 0.10], 
        [0.42, 0.12, 0.10], 
        [0.42, 0.20, 0.10]
    ], dtype=np.float32)

    # 2. Create some dummy 2D pixel coordinates
    mock_2d_points = np.array([
        [300, 200], 
        [400, 200], 
        [400, 300], 
        [300, 300]
    ], dtype=np.float32)

    # 3. Load your real intrinsic matrix and distortion coefficients 
    # (Make sure the paths point to where your k.npy and dist_coeffs.npy are saved)
    K = np.load("../hw3_task1/k.npy") 
    dist = np.load("../hw3_task1/dist_coeffs.npy")

    # 4. Run the function and print the result
    print("Calculating Extrinsics...")
    T_matrix = calculate_extrinsics(mock_3d_points, mock_2d_points, K, dist)
    
    print("\nSuccess! Final Transformation Matrix (T_base_D435):")
    print(T_matrix)
    np.save("t_base_d435.npy", T_matrix)
    print("Saved matrix to t_base_d435.npy")
