#!/usr/bin/env python3
"""
Demo of teleoperation using MuJoCo viewer.
This script allows you to control a robot arm in the SO100 environment using keyboard inputs.

It's using MOCAP to control the end effector position and orientation.

Does not record demonstrations, it's mostly to explore the mujoco scene.
"""

import mujoco
import mujoco.viewer
import numpy as np
import pyquaternion as pyq
import glfw

from gym_so101.constants import ASSETS_DIR

MOCAP_INDEX = 0

def rotate_quaternion(quat, axis, angle):
    """
    Rotate a quaternion by an angle around an axis
    """
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)
    return q.elements


def rotate_quaternion(quat, axis, angle):
    """
    Rotate a quaternion by an angle around an axis
    """
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)
    return q.elements


def key_callback_data(key, data):
    """
    Callback for key presses but with data passed in

    Args:
        key: Key pressed
        data: MjData object
    
    Returns:
        None
    """
    global MOCAP_INDEX
    print((key))
    if key == 325:  # Up arrow - Y axis (+)
        data.mocap_pos[MOCAP_INDEX, 2] += 0.01
    elif key == 322:  # Down arrow - Y axis (-)
        data.mocap_pos[MOCAP_INDEX, 2] -= 0.01
    elif key == 321:  # Left arrow - Z axis (-)
        data.mocap_pos[MOCAP_INDEX, 0] -= 0.01
    elif key == 323:  # Right arrow - Z axis (+)
        data.mocap_pos[MOCAP_INDEX, 0] += 0.01
    elif key == 326:  # + - Y axis (+)
        data.mocap_pos[MOCAP_INDEX, 1] += 0.01
    elif key == 324:  # - - Y axis (-)
        data.mocap_pos[MOCAP_INDEX, 1] -= 0.01

    # Rotation around X-axis (Pitch)
    elif key == 320:  # Q key (rotate +10 around X)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_INDEX], [1, 0, 0], 10
        )
    elif key == 330:  # A key (rotate -10 around X)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_INDEX], [1, 0, 0], -10
        )

    # Rotation around Y-axis (Yaw)
    elif key == 327:  # W key (rotate +10 around Y)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_INDEX], [0, 1, 0], 10
        )
    elif key == 329:  # S key (rotate -10 around Y)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_INDEX], [0, 1, 0], -10
        )

    # Rotation around Z-axis (Roll)
    elif key == 331:  # E key (rotate +10 around Z)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_INDEX], [0, 0, 1], 10
        )
    elif key == 328:  # D key (rotate -10 around Z)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_INDEX], [0, 0, 1], -10
        )

    elif key == 334:  # gripper open up
        data.ctrl[5] += 0.05
    elif key == 335:  # gripper close down
        data.ctrl[5] -= 0.05
    else:
        print(f"Unmapped Key: {key}")

def print_keybind() -> None:
    print("===========================")
    print(f"TELEOPERATION KEYBINDING")
    print(f"NUMPAD IS REQUIRED")
    print("===========================")
    print(
        "ALL KEYS ASSIGNED ON NUMPAD\n"\
        "5: Y Axis +\n" \
        "2: Y Axis -\n" \
        "1: Z Axis -\n" \
        "3: Z Axis +\n" \
        "")

def main():
    xml_path = ASSETS_DIR / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    data.mocap_pos[MOCAP_INDEX] = data.xpos[gripper_id]
    data.mocap_quat[MOCAP_INDEX] = data.xquat[gripper_id]

    def key_callback(key):
        key_callback_data(key, data)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()