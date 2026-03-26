import pyrealsense2 as rs
import numpy as np
import cv2

from utils.realsense_camera import init_realsense

IMAGE_FOLDER = "real_world/hw3_task1/calibration_images/"

pipeline, config = init_realsense(rgb_stream=True, depth_stream=False)
if pipeline == None or config == None:
    print("ERROR: could not initalize realsense camera")
    exit(0)

# Start streaming
num_img = 0
print("Capturing stream, use 'S' to save a frame and 'Q' to quit")
pipeline.start(config)
try:
    while True:
        # Wait for a coherent set of frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
    
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)

        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(f"{IMAGE_FOLDER}/calib_{num_img:02d}.png", color_image)
            num_img += 1

        elif key == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()