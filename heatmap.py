import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import cv2
from skimage import transform

# Tạo custom colormap (xanh → vàng → đỏ)
colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # Xanh -> Cyan -> Xanh lá -> Vàng -> Đỏ
n_bins = 100
cmap_name = 'custom_heatmap'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Tạo ảnh nền trắng (kích thước 224x224)
def create_base_image(size=224):
    img = np.ones((size, size, 3)) * 255
    return img.astype(np.uint8)

# Hàm tạo heatmap từ anomaly scores
def create_heatmap(anomaly_scores, img_size=224):
    # anomaly_scores: mảng 2D kích thước (28, 28) - giá trị từ 0-1
    heatmap_small = anomaly_scores * 255
    heatmap_small = heatmap_small.astype(np.uint8)
    
    # Nội suy về kích thước gốc
    heatmap = transform.resize(heatmap_small, (img_size, img_size), preserve_range=True).astype(np.uint8)
    return heatmap

# Hàm vẽ ảnh gốc + heatmap + overlay
def plot_results(original_img, heatmap, anomaly_score, title, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(f'{title} - Anomaly Score: {anomaly_score:.3f}', fontsize=14)
    
    # Ảnh gốc
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(heatmap, cmap=custom_cmap, vmin=0, vmax=255)
    axes[1].set_title('Anomaly Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay (heatmap chồng lên ảnh gốc)
    overlay = original_img.copy()
    heatmap_rgb = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_rgb, 0.4, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # Ground truth mask (mô phỏng)
    mask = np.zeros((224, 224, 3), dtype=np.uint8)
    if 'scratch' in title.lower() or 'dent' in title.lower():
        # Vẽ mask cho vùng lỗi
        if 'scratch' in title.lower():
            cv2.line(mask, (50, 150), (180, 120), (255, 255, 255), 5)
        else:
            cv2.circle(mask, (150, 150), 30, (255, 255, 255), -1)
    axes[3].imshow(mask)
    axes[3].set_title('Ground Truth Mask')
    axes[3].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# --- TẠO CÁC TRƯỜNG HỢP MÔ PHỎNG ---

# 1. Ảnh KHÔNG LỖI (bình thường)
img_good = create_base_image()
# Tạo anomaly scores thấp, đều khắp ảnh (nhiễu nhẹ)
scores_good = np.random.normal(0.1, 0.05, (28, 28))
scores_good = np.clip(scores_good, 0, 0.25)
heatmap_good = create_heatmap(scores_good)
anomaly_score_good = np.max(scores_good)
plot_results(img_good, heatmap_good, anomaly_score_good, 'Normal Product (No Defect)', 'heatmap_normal.png')

# 2. Ảnh có LỖI VẾT XƯỚC (scratch)
img_scratch = create_base_image()
# Thêm vết xước vào ảnh gốc
cv2.line(img_scratch, (50, 150), (180, 120), (100, 100, 100), 3)
# Tạo anomaly scores với vùng sáng dạng đường
scores_scratch = np.random.normal(0.1, 0.05, (28, 28))
for i in range(28):
    for j in range(28):
        if abs((j*8) - (i*5 + 50)) < 15:  # Mô phỏng đường chéo
            scores_scratch[i, j] = 0.85 + np.random.normal(0, 0.05)
scores_scratch = np.clip(scores_scratch, 0, 1)
heatmap_scratch = create_heatmap(scores_scratch)
anomaly_score_scratch = np.max(scores_scratch)
plot_results(img_scratch, heatmap_scratch, anomaly_score_scratch, 'Scratch Defect', 'heatmap_scratch.png')

# 3. Ảnh có LỖI VẾT LÕM (dent)
img_dent = create_base_image()
# Thêm vết lõm vào ảnh gốc
cv2.circle(img_dent, (150, 150), 30, (80, 80, 80), -1)
cv2.circle(img_dent, (150, 150), 25, (120, 120, 120), -1)
cv2.circle(img_dent, (150, 150), 20, (150, 150, 150), -1)
# Tạo anomaly scores với vùng tròn sáng
scores_dent = np.random.normal(0.1, 0.05, (28, 28))
center_x, center_y = 18, 18  # Vị trí trung tâm trên feature map 28x28
for i in range(28):
    for j in range(28):
        dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
        if dist < 4:
            scores_dent[i, j] = 0.9 - dist*0.1 + np.random.normal(0, 0.03)
        elif dist < 6:
            scores_dent[i, j] = 0.6 - (dist-4)*0.1 + np.random.normal(0, 0.03)
scores_dent = np.clip(scores_dent, 0, 1)
heatmap_dent = create_heatmap(scores_dent)
anomaly_score_dent = np.max(scores_dent)
plot_results(img_dent, heatmap_dent, anomaly_score_dent, 'Dent Defect', 'heatmap_dent.png')

# 4. Ảnh có LỖI MÀU SẮC (color defect)
img_color = create_base_image()
# Thêm vùng màu khác
cv2.rectangle(img_color, (80, 80), (140, 140), (50, 150, 200), -1)
# Tạo anomaly scores với vùng vuông sáng
scores_color = np.random.normal(0.1, 0.05, (28, 28))
scores_color[10:18, 10:18] = 0.8 + np.random.normal(0, 0.05, (8, 8))
scores_color = np.clip(scores_color, 0, 1)
heatmap_color = create_heatmap(scores_color)
anomaly_score_color = np.max(scores_color)
plot_results(img_color, heatmap_color, anomaly_score_color, 'Color Defect', 'heatmap_color.png')

# --- TẠO HÌNH TỔNG HỢP CHO BÁO CÁO ---
fig, axes = plt.subplots(4, 4, figsize=(16, 20))
fig.suptitle('Kết quả phát hiện và phân đoạn lỗi với PatchCore', fontsize=16, fontweight='bold')

cases = [
    ('Normal', img_good, heatmap_good, anomaly_score_good),
    ('Scratch', img_scratch, heatmap_scratch, anomaly_score_scratch),
    ('Dent', img_dent, heatmap_dent, anomaly_score_dent),
    ('Color Defect', img_color, heatmap_color, anomaly_score_color)
]

for row, (title, img, heatmap, score) in enumerate(cases):
    # Ảnh gốc
    axes[row, 0].imshow(img)
    axes[row, 0].set_title(f'{title} - Original', fontsize=12)
    axes[row, 0].axis('off')
    
    # Heatmap
    im = axes[row, 1].imshow(heatmap, cmap=custom_cmap, vmin=0, vmax=255)
    axes[row, 1].set_title(f'{title} - Heatmap', fontsize=12)
    axes[row, 1].axis('off')
    
    # Overlay
    overlay = img.copy()
    heatmap_rgb = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 0.6, heatmap_rgb, 0.4, 0)
    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title(f'{title} - Overlay\nScore: {score:.3f}', fontsize=12)
    axes[row, 2].axis('off')
    
    # Ground truth mask
    mask = np.zeros((224, 224, 3), dtype=np.uint8)
    if title == 'Scratch':
        cv2.line(mask, (50, 150), (180, 120), (255, 255, 255), 5)
    elif title == 'Dent':
        cv2.circle(mask, (150, 150), 30, (255, 255, 255), -1)
    elif title == 'Color Defect':
        cv2.rectangle(mask, (80, 80), (140, 140), (255, 255, 255), -1)
    axes[row, 3].imshow(mask)
    axes[row, 3].set_title(f'{title} - Ground Truth', fontsize=12)
    axes[row, 3].axis('off')

plt.tight_layout()
plt.savefig('heatmap_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Đã tạo các file heatmap:")
print("   - heatmap_normal.png")
print("   - heatmap_scratch.png")
print("   - heatmap_dent.png")
print("   - heatmap_color.png")
print("   - heatmap_summary.png")