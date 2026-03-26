import pybullet as pb
import os

def create_block(pb, pos, color, mass=0.05):
    """Creates a 4x4cm graspable block."""
    v_shape = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=color)
    c_shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
    return pb.createMultiBody(baseMass=mass, baseCollisionShapeIndex=c_shape, 
                              baseVisualShapeIndex=v_shape, basePosition=pos)

def create_bin(pb, pos, color):
    """Creates a simple flat tray to represent a bin."""
    v_shape = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.1, 0.1, 0.01], rgbaColor=color)
    c_shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.1, 0.1, 0.01])
    return pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_shape, 
                              baseVisualShapeIndex=v_shape, basePosition=pos)


def create_tag_mesh(filepath="assets/tag_mesh.obj"):
    """Generates a mathematically perfect 100mm x 100mm 3D quad with explicit UV mapping."""
    os.makedirs("assets", exist_ok=True)
    with open(filepath, "w") as f:
        # Define the 4 corners of a 100mm square (±0.05m from center)
        f.write("v -0.05 -0.05 0.0\n")  # Vertex 1: Bottom-Left
        f.write("v  0.05 -0.05 0.0\n")  # Vertex 2: Bottom-Right
        f.write("v  0.05  0.05 0.0\n")  # Vertex 3: Top-Right
        f.write("v -0.05  0.05 0.0\n")  # Vertex 4: Top-Left
        
        # Define the UV texture coordinates (0 to 1) mapping directly to the image corners
        f.write("vt 0.0 0.0\n") # UV 1
        f.write("vt 1.0 0.0\n") # UV 2
        f.write("vt 1.0 1.0\n") # UV 3
        f.write("vt 0.0 1.0\n") # UV 4
        
        # Create the face linking the vertices and UVs together
        f.write("f 1/1 2/2 3/3 4/4\n")

def spawn_apriltags(pb):
    """Spawns four perfectly textured 100mm AprilTags at the corners."""
    # Generate mesh file
    create_tag_mesh()
    
    # Use the explicit mesh instead of a procedural box
    tag_shape = pb.createVisualShape(
        shapeType=pb.GEOM_MESH, 
        fileName="assets/tag_mesh.obj",
        meshScale=[1, 1, 1]
    )
    
    # Load the 1024x1024 texture you generated earlier
    texture_id = pb.loadTexture("assets/apriltag_0.png")
    
    corners = {
        "Top-Left":     [-0.3, 0.7, 0.626],
        "Top-Right":    [ 0.3, 0.7, 0.626],
        "Bottom-Right": [ 0.3, 0.3, 0.626],
        "Bottom-Left":  [-0.3, 0.3, 0.626]
    }
    
    tag_ids = {}
    
    for name, pos in corners.items():
        ori = pb.getQuaternionFromEuler([0, 0, 0])
        
        body_id = pb.createMultiBody(
            baseMass=0, 
            baseVisualShapeIndex=tag_shape, 
            basePosition=pos, 
            baseOrientation=ori
        )
        # Apply the texture to the mesh
        pb.changeVisualShape(body_id, -1, textureUniqueId=texture_id)
        tag_ids[name] = body_id
        
    return tag_ids, corners