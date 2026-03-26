import pyrealsense2 as rs
from pyrealsense2 import pipeline, config

def init_realsense(rgb_stream=True, depth_stream=True) -> tuple[pipeline, config]:
    """ 
    Initializes an Intel Realsense camera pipeline

    Args:
        rgb_stream (bool): Initalize rgb stream
        depth_stream (bool): Initialize depth stream

    Returns:
        pipeline: pipeline for camera
        config: config for camera
    """
    CAMERA_X = 640
    CAMERA_Y = 480
    FRAME_RATE = 30

    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    found_depth = False

    for sensor in device.sensors:
        camera_name = sensor.get_info(rs.camera_info.name)
        if rgb_stream and not found_rgb and camera_name == "RGB Camera":
            found_rgb = True
            break
    if rgb_stream and not found_rgb:
        print("ERROR: Could not find RGB camera in device sensors")
        return [None, None]
    
    if rgb_stream:
        config.enable_stream(rs.stream.depth, CAMERA_X, CAMERA_Y, rs.format.z16, FRAME_RATE)
    
    if depth_stream:
        config.enable_stream(rs.stream.color, CAMERA_X, CAMERA_Y, rs.format.bgr8, FRAME_RATE)

    return [pipeline, config]