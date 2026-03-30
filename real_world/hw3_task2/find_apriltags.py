import cv2
import numpy as np
from pupil_apriltags import Detector

def order_corners(corners):
    """
    Orders corners to match the assignment requirement:
    Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    """
    # Sort by y-coordinate to separate top and bottom
    corners = sorted(corners, key=lambda c: c[1])
    top_corners = sorted(corners[:2], key=lambda c: c[0])
    bottom_corners = sorted(corners[2:], key=lambda c: c[0], reverse=True)
    
    # Return as: TL, TR, BR, BL
    return np.array([top_corners[0], top_corners[1], bottom_corners[0], bottom_corners[1]], dtype=np.float32)

def detect_apriltag_corners(image_path):
    # Load the image and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize the detector for the specific tag family
    at_detector = Detector(families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25)

    # Detect tags in the image
    tags = at_detector.detect(gray)

    for tag in tags:
        # We only care about Tag ID 0 for this assignment
        if tag.tag_id == 0:
            print(f"Found Tag ID 0 with high confidence: {tag.decision_margin}")
            
            # Extract and order the corners
            ordered_corners = order_corners(tag.corners)
            
            # Draw the corners for verification (Task 2.2 visualization)
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # Red, Green, Blue, Yellow
            for i, corner in enumerate(ordered_corners):
                pt = (int(corner[0]), int(corner[1]))
                cv2.circle(img, pt, 5, colors[i], -1)
                cv2.putText(img, str(i+1), (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
            
            cv2.imwrite("apriltag_detection_output.png", img)
            print("Saved visualization to apriltag_detection_output.png")
            
            return ordered_corners

    print("Tag ID 0 not found.")
    return None

# --- Mock Test ---
corners_2d = detect_apriltag_corners("real_world/hw3_task1/calibration_images/april_tag/calib_00.png")
print(corners_2d)
