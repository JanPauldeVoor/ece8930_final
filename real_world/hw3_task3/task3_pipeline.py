import numpy as np

def pixel_to_camera_frame(u, v, Z_c, K_matrix):
    """
    Task 1.3: Unprojects a 2D pixel and depth into 3D Camera Space (P_D435).
    Returns a 4x1 homogeneous vector: [X_c, Y_c, Z_c, 1.0]^T
    """
    # Extract intrinsic parameters from K
    f_x = K_matrix[0, 0]
    f_y = K_matrix[1, 1]
    c_x = K_matrix[0, 2]
    c_y = K_matrix[1, 2]

    # Inverse pinhole camera model
    X_c = (u - c_x) * Z_c / f_x
    Y_c = (v - c_y) * Z_c / f_y

    # Return as a 4D homogeneous coordinate array
    return np.array([X_c, Y_c, Z_c, 1.0])

def camera_to_base_frame(P_D435, T_base_D435):
    """
    Task 3: Transforms the 3D point from the Camera Frame to the Robot Base Frame.
    Equation: P_base = T_base_D435 * P_D435
    """
    # Perform matrix multiplication (dot product)
    P_base = np.dot(T_base_D435, P_D435)
    
    return P_base

# ==========================================
# --- Mock Offline Test & Report Example ---
# ==========================================
if __name__ == "__main__":
    # 1. Mock inputs from Task 1 (Target observation)
    target_u = 320         # Pixel X
    target_v = 240         # Pixel Y
    target_Z_c = 0.550     # Depth in meters
    
    # 2. Mock Intrinsic Matrix (K) from Task 1.1
    # Replace with np.load("k.npy") when running your real pipeline
    mock_K = np.array([
        [615.0,   0.0, 325.0],
        [  0.0, 615.0, 245.0],
        [  0.0,   0.0,   1.0]
    ])

    # 3. Mock Extrinsic Matrix (T_base_D435) from Task 2.3
    # This is the inverted 4x4 matrix output from your solvePnP script
    mock_T_base_D435 = np.array([
        [ 0.0, -1.0,  0.0,  0.45],  # 90 deg rotation, 0.45m translation in X
        [ 1.0,  0.0,  0.0,  0.10],  # 0.10m translation in Y
        [ 0.0,  0.0,  1.0, -0.05],  # -0.05m translation in Z (height offset)
        [ 0.0,  0.0,  0.0,  1.00]
    ])

    # --- Execute Pipeline ---
    
    # Step A: Find P_D435
    P_D435 = pixel_to_camera_frame(target_u, target_v, target_Z_c, mock_K)
    print(f"1. Target in Camera Frame (P_D435): \n   [X: {P_D435[0]:.4f}, Y: {P_D435[1]:.4f}, Z: {P_D435[2]:.4f}]")

    # Step B: Find P_base
    P_base = camera_to_base_frame(P_D435, mock_T_base_D435)
    print(f"\n2. Target in Robot Base Frame (P_base): \n   [X: {P_base[0]:.4f}, Y: {P_base[1]:.4f}, Z: {P_base[2]:.4f}]")
