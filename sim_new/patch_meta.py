import json
from pathlib import Path

# Path to your local dataset's info.json
info_path = Path("local_datasets/so101_touch_cube/meta/info.json")

if not info_path.exists():
    print(f"Could not find {info_path}. Make sure you are in the correct directory.")
    exit(1)

with open(info_path, 'r') as f:
    data = json.load(f)

# The joint names you defined in constants.py
joints = [
    "left_arm_waist", "left_arm_shoulder", "left_arm_elbow", 
    "left_arm_forearm_roll", "left_arm_wrist_rotate", "left_arm_gripper"
]

print("Patching features...")
for key, feature_dict in data.get('features', {}).items():
    if key in ['action', 'observation.state']:
        feature_dict['names'] = joints
        print(f" -> Added joint names to '{key}'")
    elif 'images' in key:
        feature_dict['names'] = ['c', 'h', 'w']  # Channels, Height, Width
        print(f" -> Added image names to '{key}'")
    else:
        # For language_instruction, timestamp, episode_index, etc.
        if 'names' not in feature_dict:
            feature_dict['names'] = None 
            print(f" -> Added null names to '{key}'")

# Save the patched JSON back to the file
with open(info_path, 'w') as f:
    json.dump(data, f, indent=2)

print("\nSuccessfully patched info.json!")