import cv2
import numpy as np
import os

os.makedirs("assets", exist_ok=True)

# Load AprilTag 36h11 Dictionary
try:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
except AttributeError:
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)

# Generate Tag ID 0
# black tag to be 80mm, and the total to be 100mm.
# Map 1mm to 10 pixels 
# 80mm tag = 800 pixels
tag_size_px = 800
tag_img = cv2.aruco.generateImageMarker(dictionary, 0, tag_size_px)

# Add the 10mm white margin on all sides
# 10mm margin = 100 pixels per side
border_size = 100
tag_with_margin = cv2.copyMakeBorder(
    tag_img, border_size, border_size, border_size, border_size, 
    cv2.BORDER_CONSTANT, value=[255, 255, 255]
)

# Resize for PyBullet OpenGL compatibility
final_img = cv2.resize(tag_with_margin, (1024, 1024), interpolation=cv2.INTER_NEAREST)

# Save the texture
cv2.imwrite("assets/apriltag_0.png", final_img)
print("100mm AprilTag texture saved to assets/apriltag_0.png")