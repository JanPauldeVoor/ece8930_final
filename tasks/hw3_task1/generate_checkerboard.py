import cv2
import numpy as np
import os

filepath = "assets/checkerboard.png"

os.makedirs("assets", exist_ok=True)

cols, rows = 9, 7
square_size_px = 100
tex_size = 1024 # Strict power of 2 for PyBullet

# Create a solid white background (1024x1024)
board_img = np.ones((tex_size, tex_size), dtype=np.uint8) * 255

# Calculate offsets to perfectly center the 900x700 checkerboard
start_x = (tex_size - (cols * square_size_px)) // 2
start_y = (tex_size - (rows * square_size_px)) // 2

# Draw the checkerboard
for r in range(rows):
    for c in range(cols):
        if (r + c) % 2 == 1: # Black square
            x1 = start_x + c * square_size_px
            y1 = start_y + r * square_size_px
            x2 = x1 + square_size_px
            y2 = y1 + square_size_px
            board_img[y1:y2, x1:x2] = 0

cv2.imwrite("assets/checkerboard.png", board_img)
print("Checkerboard texture saved.")

sq_size = 0.025
with open("assets/calibration_board.urdf", "w") as f:
    f.write('<?xml version="1.0" ?>\n<robot name="board">\n')
    
    # Add a 4cm padding around the board
    padding = 0.04 
    w = (cols * sq_size) + (padding * 2)
    h = (rows * sq_size) + (padding * 2)
    
    f.write('  <link name="base_link">\n')
    f.write('    <visual>\n')
    f.write(f'      <geometry><box size="{w} {h} 0.001"/></geometry>\n')
    f.write('      <material name="white"><color rgba="1 1 1 1"/></material>\n')
    f.write('    </visual>\n')
    f.write('    <inertial><mass value="0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>\n')
    f.write('  </link>\n')
    
    # Center the squares on the new, larger board
    start_x = -((cols - 1) * sq_size) / 2.0
    start_y = -((rows - 1) * sq_size) / 2.0
    
    for r in range(rows):
        for c in range(cols):
            color = "0.1 0.1 0.1 1" if (r + c) % 2 == 1 else "0.9 0.9 0.9 1"
            link_name = f"sq_{r}_{c}"
            x = start_x + c * sq_size
            y = start_y + r * sq_size
            
            f.write(f'  <link name="{link_name}">\n')
            f.write('    <visual>\n')
            f.write(f'      <geometry><box size="{sq_size} {sq_size} 0.0015"/></geometry>\n')
            f.write(f'      <material name="mat_{r}_{c}"><color rgba="{color}"/></material>\n')
            f.write('    </visual>\n')
            f.write('    <inertial>\n')
            f.write('      <mass value="0.001"/>\n')
            f.write('      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>\n')
            f.write('    </inertial>\n')
            f.write('  </link>\n')
            
            f.write(f'  <joint name="j_{r}_{c}" type="fixed">\n')
            f.write('    <parent link="base_link"/>\n')
            f.write(f'    <child link="{link_name}"/>\n')
            f.write(f'    <origin xyz="{x:.4f} {y:.4f} 0.001" rpy="0 0 0"/>\n')
            f.write('  </joint>\n')
            
    f.write('</robot>\n')

print("Checkerboard urdf generated")