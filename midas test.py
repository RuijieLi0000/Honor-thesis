import cv2
import torch
import numpy as np

# Setting device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"正在使用设备: {device}")

# loadMiDaS
model_type = "MiDaS_small" 
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device).eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Turn on camera (OWL 360)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 360-degree panoramic view, 2:1 aspect ratio
    frame = cv2.resize(frame, (1200, 600))
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_viz = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)

    combined = np.vstack((frame, depth_viz))
    cv2.imshow('OWL 360 MiDaS Vision', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()