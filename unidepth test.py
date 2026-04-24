import cv2
import torch
import numpy as np
import sys

# ======================================================
# 🚀 第一部分：强制修复 Torch Hub 的 Authorization Bug
# ======================================================
import torch.hub
def patched_validate_not_a_forked_repo(repo_owner, repo_name, ref):
    # 这个补丁直接跳过报错的验证逻辑
    return True
# 将官方的有 Bug 的函数替换成我们这个不报错的函数
torch.hub._validate_not_a_forked_repo = patched_validate_not_a_forked_repo
print("✅ 已应用 Torch Hub 兼容性补丁")
# ======================================================

# 1. 加载模型
print("正在通过 Hub 加载 Depth-Anything-V2-Small (CPU 优化版)...")
try:
    # 注意这里加上了 trust_repo=True
    model = torch.hub.load("baidu-research/Depth-Anything-V2", "depth_anything_v2_vits", pretrained=True, trust_repo=True)
    device = torch.device("cpu")
    model = model.to(device).eval()
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 加载仍然失败: {e}")
    print("\n💡 最后的建议：如果补丁无效，请在终端运行: pip install --upgrade torch")
    sys.exit()

# 2. 开启 Owl 摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("错误：无法打开摄像头")
    sys.exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 预处理 ---
    # CPU 环境下，我们将图像缩小到 320x240 以保证基本的帧率
    raw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- 深度推理 ---
    with torch.no_grad():
        # infer_image 是该模型的便捷接口，内部会自动处理缩放
        depth = model.infer_image(raw_img, 320) # 320 为输入分辨率

    # --- 后处理与可视化 ---
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_viz = (depth_norm * 255).astype(np.uint8)
    
    # 使用伪彩色映射
    depth_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_MAGMA)
    depth_res = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))

    # --- 拼接显示 ---
    combined = np.hstack((frame, depth_res))
    cv2.putText(combined, "Depth-Anything-V2 (CPU Mode)", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Owl Camera + Depth Estimation', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()