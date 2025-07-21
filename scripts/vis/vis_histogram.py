import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns  # 导入 seaborn 库

def quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = np.cross(xyz, vec) * 2
    return (vec - quat[:, 0:1] * t + np.cross(xyz, t)).reshape(shape)

folder = "default_controller_slow"
path = f"{folder}/motion.npz"
meta_path = f"{folder}/meta.json"

with open(meta_path, 'r') as f:
    meta = json.load(f)
print(meta)
data = np.load(path)
for key in data.files:
    print(f"{key}: {data[key].shape}")
body_lin_vel_w = data['body_lin_vel_w']
body_ang_vel_w = data['body_ang_vel_w']
body_quat_w = data['body_quat_w']
body_lin_vel_b = quat_apply_inverse(body_quat_w, body_lin_vel_w)
body_ang_vel_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)

pelvis_idx = meta['body_names'].index('pelvis')
root_lin_vel_b = body_lin_vel_b[:, pelvis_idx, :]
root_ang_vel_b = body_ang_vel_b[:, pelvis_idx, :]

# --- 绘图部分（已修改为 KDE Plot） ---
fig, axs = plt.subplots(2, 1, figsize=(6, 5))  # 稍微调整画布大小以便更好地显示
axs = axs.flatten()
colors = ['r', 'g', 'b']
labels = ['x', 'y', 'z']

# 子图[0]: 根部线速度的KDE图
for i in range(3):
    sns.kdeplot(root_lin_vel_b[:, i], ax=axs[0], color=colors[i], label=labels[i], fill=True, alpha=0.4)
axs[0].set_title('Root Linear Velocity KDE Plot')
axs[0].set_xlabel('Velocity')
axs[0].set_ylabel('Density') # KDE 图的y轴是密度
axs[0].legend()

# 子图[1]: 根部角速度的KDE图
for i in range(3):
    sns.kdeplot(root_ang_vel_b[:, i], ax=axs[1], color=colors[i], label=labels[i], fill=True, alpha=0.4)
axs[1].set_title('Root Angular Velocity KDE Plot')
axs[1].set_xlabel('Velocity')
axs[1].set_ylabel('Density') # KDE 图的y轴是密度
axs[1].legend()

plt.tight_layout() # 使用 tight_layout() 避免标题和标签重叠
plt.show()