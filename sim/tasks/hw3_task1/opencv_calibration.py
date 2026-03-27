import cv2
import numpy as np
import glob

# Checkerboard properties
CHECKERBOARD = (8, 6) # Inner corners (Width-1, Height-1)
SQUARE_SIZE = 0.025   # 25mm in meters

# Termination criteria for corner sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points based on physical square size
# e.g., (0,0,0), (0.025,0,0), (0.05,0,0) ...
# Z = 0 for flat board
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store object points and image points from all images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

images = glob.glob('tasks/hw3_task1/calibration_images/*.png')
print(f"Found {len(images)} images. Processing...")

success_count = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

    if ret == True:
        objpoints.append(objp)
        # Refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        success_count += 1

cv2.destroyAllWindows()
print(f"Found corners in {success_count}/{len(images)} images.")

if success_count > 0:
    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\n--- Calibration Results ---")
    print("Camera Intrinsic Matrix (K):")
    print(np.round(mtx, 2))
    
    print("\nDistortion Coefficients (k1, k2, p1, p2, k3):")
    print(np.round(dist, 5))
    
    # Calculate re-projection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"\nTotal Re-projection Error: {mean_error/len(objpoints):.4f} pixels")
else:
    print("Failed to find corners in any images. Check board size and lighting.")