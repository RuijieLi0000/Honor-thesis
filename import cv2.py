import cv2
import torch
import sys
import os
import numpy as np

# 1. 自动定位路径
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.join(current_dir, "Depth-Anything-V2")
sys.path.append(repo_path)

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    print("成功找到 Depth-Anything-V2 模块！")
except ImportError:
    print("错误：找不到模块，请检查 Depth-Anything-V2 文件夹是否存在。")
    sys.exit()

# 2. 设置设备 (CPU可能较慢，有GPU会自动切换)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"正在使用设备: {device}")

# 3. 加载模型
model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
model = DepthAnythingV2(**model_configs['vits'])
checkpoint_path = os.path.join(repo_path, "checkpoints", "depth_anything_v2_vits.pth")

if not os.path.exists(checkpoint_path):
    print(f"找不到权重文件: {checkpoint_path}")
    sys.exit()

model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.to(device).eval()
print("模型加载成功！")

# 4. 打开摄像头并设置 360 度全景分辨率
cap = cv2.VideoCapture(0)
# 强制要求摄像头输出宽画幅
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # --- 关键：调整为 2:1 的 360度全景比例 ---
    # 宽度设为高度的两倍，确保左右视野不被裁剪
    frame = cv2.resize(frame, (1200, 600))

    # 5. 推理
    with torch.no_grad():
        # input_size 越大细节越多，但也越卡，建议先用 518 
        depth = model.infer_image(frame, input_size=518)

    # 6. 处理深度图 (定义 depth_viz)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_norm = depth_norm.astype(np.uint8)
    
    # 生成我们要显示的变量 depth_viz
    depth_viz = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

    # 7. 显示结果
    # 我们把原图和深度图上下拼在一起看，视野更震撼
    combined = np.vstack((frame, depth_viz))
    cv2.imshow('OWL 360 Full Vision (Original vs Depth)', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()