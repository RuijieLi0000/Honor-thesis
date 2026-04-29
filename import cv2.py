import cv2
import torch
import sys
import os
import numpy as np

# Automatic path location
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.join(current_dir, "Depth-Anything-V2")
sys.path.append(repo_path)

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    print("Module Depth-Anything-V2 successfully located!！")
except ImportError:
    print("Error: Module not found. Please check if the Depth-Anything-V2 folder exists.")
    sys.exit()

# Device settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# load model
model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
model = DepthAnythingV2(**model_configs['vits'])
checkpoint_path = os.path.join(repo_path, "checkpoints", "depth_anything_v2_vits.pth")

if not os.path.exists(checkpoint_path):
    print(f"Weight file not found: {checkpoint_path}")
    sys.exit()

model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.to(device).eval()
print("Model loaded successfully!")

# Open camera and set 360-degree panoramic resolution
cap = cv2.VideoCapture(0)
# Force the camera to output a wide aspect ratio
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Adjust to a 2:1 360-degree panoramic ratio
    # Set the width to twice the height to ensure that the left and right views are not cropped.
    frame = cv2.resize(frame, (1200, 600))

    # Inference
    with torch.no_grad():
        depth = model.infer_image(frame, input_size=518)

    # Process depth map (define depth_viz)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_norm = depth_norm.astype(np.uint8)
    
    # Generate the variable depth_viz we want to display
    depth_viz = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

    # outputs
    # Stack the original image and depth map vertically for a more immersive view
    combined = np.vstack((frame, depth_viz))
    cv2.imshow('OWL 360 Full Vision (Original vs Depth)', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()