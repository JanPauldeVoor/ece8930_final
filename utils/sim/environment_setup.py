import pybullet as pb
import pkgutil
import pybullet_data
import os

def environment_setup():
    physicsClient = pb.connect(pb.GUI)

    # Make sure there's hardware acceleration
    egl = pkgutil.get_loader('eglRenderer')
    if egl:
        plugin_id = pb.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
    else:
        plugin_id = pb.loadPlugin("eglRendererPlugin")

    if plugin_id >= 0:
        print("Hardware accelerated rendering (EGL) enabled successfully.")
    else:
        print("Warning: EGL plugin failed to load. Defaulting to standard rendering.")

    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.81)

    # Build Environment 
    plane_id = pb.loadURDF("plane.urdf")
    # The default PyBullet table is ~0.625m high
    table_id = pb.loadURDF("table/table.urdf", basePosition=[0, 0.5, 0])

    # Load SO-101 Arm
    urdf_path = "assets/so101/so101.urdf"

    so101_pos = [0, 0.1, 0.625] # Mounted on the edge of the table
    so101_ori = pb.getQuaternionFromEuler([0, 0, 0])

    if os.path.exists(urdf_path):
        so101_id = pb.loadURDF(urdf_path, so101_pos, so101_ori, useFixedBase=True)
        print("SO-101 Arm loaded successfully.")
    else:
        print(f"WARNING: Could not find URDF at {urdf_path}.")
        print("Using a placeholder robotic arm (KUKA) for testing the environment...")
        so101_id = pb.loadURDF("kuka_iiwa/model.urdf", so101_pos, so101_ori, useFixedBase=True)

    return pb, plane_id, table_id, so101_id, physicsClient