import cv2
import numpy as np
import glob

# Calibration Images
IMAGE_DIR = "real_world/hw3_task1/calibration_images"

# Intrinsic and Distortion arrays
K_FILE = "real_world/hw3_task1/k.npy"
DIST_FILE = "real_world/hw3_task1/dist_coeffs.npy"

# CharucoBoard properties
X_SQUARES = 14
Y_SQUARES = 9
SQUARE_SIZE = 0.020 # 20mm in meters
MARKER_SIZE = 0.015 # 15mm in meters

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
board = cv2.aruco.CharucoBoard((X_SQUARES, Y_SQUARES), SQUARE_SIZE, MARKER_SIZE, dictionary)

# Termination criteria for corner sub-pixel accuracy

# Prepare 3D object points based on physical square size
# e.g., (0,0,0), (0.025,0,0), (0.05,0,0) ...
# Z = 0 for flat board
objp = np.zeros((X_SQUARES * Y_SQUARES, 3), np.float32)
objp[:, :2] = np.mgrid[0:X_SQUARES, 0:Y_SQUARES].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store object points and image points from all images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

images = glob.glob(f'{IMAGE_DIR}/*.png')
print(f"Found {len(images)} images. Processing...")

all_charuco_corners = []
all_charuco_ids = []
image_size = None

success_count = 0
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Could not find img: {fname}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1]
     
    # Find the corners
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)

    if ids is not None and len(ids) > 0:
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)

        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            success_count += 1
        else:
            print(f"Error extracting data from: {fname}")

    objpoints.append(objp)
    # Refine corner locations
    # corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    # imgpoints.append(corners2)
    # success_count += 1

# cv2.destroyAllWindows()
print(f"Found corners in {success_count}/{len(images)} images.")

if len(all_charuco_corners) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(charucoCorners=all_charuco_corners, 
                                                                    charucoIds=all_charuco_ids,
                                                                    board=board,
                                                                    imageSize=image_size, 
                                                                    cameraMatrix=None, 
                                                                    distCoeffs=None)
    print(f"Intrinsic Matrix K:\n {mtx}\n")
    print(f"Distortion Coeffs:\n {dist}\n")

    print(f"Saving K and Dist. Coeffs to: {K_FILE} and {DIST_FILE}")
    np.save(K_FILE, mtx)
    np.save(DIST_FILE, dist)

else:
    print("ERROR: Failed to get any charuco corners")
    exit(0)


    
# Calculate re-projection error
# mean_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     # error = cv2.norm(all_charuco_corners[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
#     mean_error += error
# print(f"\nTotal Re-projection Error: {mean_error/len(objpoints):.4f} pixels")
