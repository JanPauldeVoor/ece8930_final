import torch
import numpy as np
import imageio
import gymnasium as gym

# Import your custom environment so Gymnasium registers it
import gym_so101 

# Safely import the PI0.5 policy class based on LeRobot's structure
try:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy as TrainedPolicy
except ImportError:
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy as TrainedPolicy

def evaluate():
    print("Loading the trained PI0.5 model...")
    checkpoint_dir = "outputs/pi05_training/checkpoints/003000/pretrained_model"

    policy = TrainedPolicy.from_pretrained(checkpoint_dir)
    policy.to("cuda")
    policy.eval()

    print("Starting custom SO-101 environment...")
    env = gym.make("gym_so101/SO101TouchCube", 
        obs_type="so101_pixels_agent_pos",
        observation_width=640,
        observation_height=480
    ) 
    obs, info = env.reset()

    # --- UPA-RFAS SETUP ---
    # Load your trained patch here. For now, we will create a dummy 40x40 red patch
    # to test the pipeline. Replace this with: torch.load("path/to/upa_rfas_patch.pt")
    patch_size = 40
    adv_patch = torch.ones((1, 3, patch_size, patch_size), device="cuda")
    adv_patch[:, 1:, :, :] = 0  # Zero out Green and Blue channels to make it Red
    patch_x, patch_y = 100, 100 # Top-left coordinates for the patch placement
    # ----------------------

    frames = []
    max_steps = 300

    print("Letting the AI drive...")
    with torch.inference_mode():
        for step in range(max_steps):
            lerobot_obs = {}
            
            # --- SMART IMAGE EXTRACTION ---
            if "pixels" in obs:
                img = obs["pixels"]
                if isinstance(img, dict):
                    img = img.get("workspace_cam", list(img.values())[0])
            elif "images" in obs:
                img = obs["images"]["workspace_cam"]
            else:
                img = next((v for k, v in obs.items() if isinstance(v, np.ndarray) and len(v.shape) == 3), None)
                if img is None:
                    raise KeyError(f"Could not find image in obs! Available keys: {list(obs.keys())}")

            # --- SMART STATE EXTRACTION ---
            state = obs.get("agent_pos", obs.get("state"))
            if state is None:
                state = next((v for k, v in obs.items() if isinstance(v, np.ndarray) and len(v.shape) == 1), None)

            # 1. Format the Image Tensor
            img_tensor = torch.from_numpy(img).float() / 255.0
            
            # SMART PERMUTE: If channels are at the end (H, W, C), move them to the front (C, H, W)
            if img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
                
            img_tensor = img_tensor.unsqueeze(0).to("cuda")
            img_tensor = torch.nn.functional.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            
            # --- UPA-RFAS ADVERSARIAL PATCH INJECTION ---
            # Overlay the patch on the 224x224 image tensor before it goes to the model
            end_y = min(224, patch_y + adv_patch.shape[2])
            end_x = min(224, patch_x + adv_patch.shape[3])
            p_h = end_y - patch_y
            p_w = end_x - patch_x
            
            # Apply the patch (overwriting the original pixels)
            img_tensor[:, :, patch_y:end_y, patch_x:end_x] = adv_patch[:, :, :p_h, :p_w]
            # --------------------------------------------

            lerobot_obs["observation.images.workspace_cam"] = img_tensor
            
            # 2. Format the State Tensor
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to("cuda")
            lerobot_obs["observation.state"] = state_tensor

            # 3. Dummy Language Tensors (FIXED TO BOOL)
            lerobot_obs["observation.language.tokens"] = torch.zeros((1, 200), dtype=torch.long, device="cuda")
            lerobot_obs["observation.language.attention_mask"] = torch.zeros((1, 200), dtype=torch.bool, device="cuda")

            # 4. Ask the neural network for the next action
            action_tensor = policy.select_action(lerobot_obs)
            
            if isinstance(action_tensor, dict):
                action_numpy = action_tensor["action"].cpu().numpy().squeeze()
            else:
                action_numpy = action_tensor.cpu().numpy().squeeze()
            
            # 5. Step the environment forward
            obs, reward, terminated, truncated, info = env.step(action_numpy)
            
            # 6. Save the video frame 
            # (UPDATED: Use the patched tensor so the attack is visible in the MP4!)
            patched_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
            vid_frame = patched_np.astype(np.uint8)
            frames.append(vid_frame)
            
            if terminated or truncated:
                print(f"Episode finished early at step {step}! Reward: {reward}")
                break

    print(f"Saving video with {len(frames)} frames to evaluation_result.mp4...")
    imageio.mimsave("evaluation_result.mp4", frames, fps=50)
    print("Evaluation complete! Open the MP4 to watch your trained policy.")

if __name__ == "__main__":
    evaluate()
