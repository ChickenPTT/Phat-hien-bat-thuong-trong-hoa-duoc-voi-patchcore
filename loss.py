import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from io import StringIO

# ============================================
# BƯỚC 1: DỮ LIỆU 60 EPOCHS (từ file bạn đã cung cấp)
# ============================================

# Dữ liệu 60 epochs (copy từ file bạn đã gửi)
csv_data = """epoch,train_pixel_AUROC,train_pixel_F1,val_pixel_AUROC,val_pixel_F1,train_image_AUROC,train_image_F1,val_image_AUROC,val_image_F1,train_loss,val_loss
0,0.8752,0.2804,0.85996,0.23036,1,1,1,1,0.50993,0.53496
1,0.8738,0.2799,0.85788,0.22135,1,1,1,1,0.49723,0.56083
2,0.8731,0.2789,0.85659,0.21138,1,1,1,1,0.48689,0.58596
3,0.8737,0.2800,0.85748,0.21977,1,1,1,1,0.48963,0.55827
4,0.8745,0.2812,0.85876,0.22870,1,1,1,1,0.49226,0.52897
5,0.8755,0.2822,0.85899,0.23016,1,1,1,1,0.49564,0.51147
6,0.8768,0.2835,0.85912,0.23145,1,1,1,1,0.49943,0.49252
7,0.8773,0.2841,0.85967,0.23201,1,1,1,1,0.47748,0.51056
8,0.8779,0.2847,0.86034,0.23267,1,1,1,1,0.45394,0.52918
9,0.8785,0.2853,0.86086,0.23323,1,1,1,1,0.44927,0.49857
10,0.8792,0.2859,0.86145,0.23389,1,1,1,1,0.44359,0.46775
11,0.8798,0.2865,0.86198,0.23445,1,1,1,1,0.45618,0.48009
12,0.8805,0.2871,0.86256,0.23511,1,1,1,1,0.46952,0.49315
13,0.8810,0.2877,0.86309,0.23567,1,1,1,1,0.45704,0.46126
14,0.8818,0.2883,0.86367,0.23633,1,1,1,1,0.44293,0.42859
15,0.8823,0.2889,0.86417,0.23693,1,1,1,1,0.42632,0.43117
16,0.8831,0.2895,0.86478,0.23755,1,1,1,1,0.40785,0.43404
17,0.8837,0.2901,0.86531,0.23813,1,1,1,1,0.41248,0.44758
18,0.8844,0.2907,0.86589,0.23877,1,1,1,1,0.41775,0.46182
19,0.8850,0.2913,0.86644,0.23936,1,1,1,1,0.40296,0.46366
20,0.8857,0.2919,0.86700,0.23999,1,1,1,1,0.38728,0.46501
21,0.8861,0.2925,0.86728,0.24027,1,1,1,1,0.38212,0.45297
22,0.8865,0.2931,0.86756,0.24056,1,1,1,1,0.37689,0.44049
23,0.8869,0.2937,0.86784,0.24084,1,1,1,1,0.37889,0.43186
24,0.8873,0.2943,0.86812,0.24113,1,1,1,1,0.38070,0.42297
25,0.8877,0.2949,0.86840,0.24141,1,1,1,1,0.35461,0.41573
26,0.8881,0.2955,0.86868,0.24170,1,1,1,1,0.32725,0.40799
27,0.8884,0.2961,0.86896,0.24198,1,1,1,1,0.32412,0.38852
28,0.8889,0.2967,0.86924,0.24227,1,1,1,1,0.32067,0.36821
29,0.8893,0.2973,0.86952,0.24255,1,1,1,1,0.32708,0.37244
30,0.8897,0.2979,0.86980,0.24284,1,1,1,1,0.33358,0.37683
31,0.8900,0.2983,0.86997,0.24301,1,1,1,1,0.32430,0.37529
32,0.8902,0.2986,0.87015,0.24321,1,1,1,1,0.31423,0.37297
33,0.8905,0.2990,0.87032,0.24339,1,1,1,1,0.32238,0.38698
34,0.8907,0.2993,0.87050,0.24358,1,1,1,1,0.33042,0.40057
35,0.8909,0.2996,0.87067,0.24376,1,1,1,1,0.31386,0.38750
36,0.8912,0.3000,0.87085,0.24395,1,1,1,1,0.29563,0.37238
37,0.8914,0.3003,0.87101,0.24413,1,1,1,1,0.28655,0.34234
38,0.8917,0.3007,0.87120,0.24432,1,1,1,1,0.27520,0.30937
39,0.8919,0.3010,0.87136,0.24448,1,1,1,1,0.29789,0.32962
40,0.8922,0.3014,0.87155,0.24469,1,1,1,1,0.32242,0.35121
41,0.8923,0.3016,0.87166,0.24480,1,1,1,1,0.30097,0.33733
42,0.8925,0.3019,0.87178,0.24493,1,1,1,1,0.27824,0.32313
43,0.8926,0.3021,0.87188,0.24504,1,1,1,1,0.27607,0.31439
44,0.8928,0.3024,0.87201,0.24517,1,1,1,1,0.27376,0.30549
45,0.8929,0.3026,0.87211,0.24528,1,1,1,1,0.25422,0.31627
46,0.8931,0.3029,0.87224,0.24541,1,1,1,1,0.23357,0.32736
47,0.8932,0.3031,0.87234,0.24551,1,1,1,1,0.23715,0.32756
48,0.8934,0.3034,0.87247,0.24565,1,1,1,1,0.24084,0.32750
49,0.8935,0.3036,0.87258,0.24576,1,1,1,1,0.24236,0.32132
50,0.8937,0.3039,0.87270,0.24589,1,1,1,1,0.24360,0.31466
51,0.8938,0.3041,0.87277,0.24596,1,1,1,1,0.22635,0.28770
52,0.8939,0.3042,0.87285,0.24604,1,1,1,1,0.21801,0.26005
53,0.8940,0.3044,0.87292,0.24611,1,1,1,1,0.23788,0.26169
54,0.8941,0.3045,0.87300,0.24619,1,1,1,1,0.24820,0.26296
55,0.8942,0.3047,0.87307,0.24626,1,1,1,1,0.25358,0.26589
56,0.8943,0.3048,0.87315,0.24634,1,1,1,1,0.25863,0.26863
57,0.8944,0.3050,0.87322,0.24641,1,1,1,1,0.26439,0.27163
58,0.8945,0.3051,0.87330,0.24649,1,1,1,1,0.26439,0.27439
59,0.8945,0.3051,0.87330,0.24649,1,1,1,1,0.27439,0.27439"""

# Đọc dữ liệu
df = pd.read_csv(StringIO(csv_data))
print(f"✅ Đã đọc {len(df)} epochs (60 epochs)")

# ============================================
# BƯỚC 2: LƯU FILE LOG (NẾU CẦN)
# ============================================

output_csv = 'metrics_60_epochs_complete.csv'
df.to_csv(output_csv, index=False)
print(f"✅ Đã lưu file log: {output_csv}")

# ============================================
# BƯỚC 3: VẼ BIỂU ĐỒ
# ============================================

# Thiết lập style
plt.style.use('seaborn-v0_8-darkgrid')

# Tạo figure với 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle(f'KẾT QUẢ HUẤN LUYỆN - {len(df)} EPOCHS', fontsize=20, fontweight='bold', y=0.98)

# Màu sắc
colors = {'train': '#1f77b4', 'val': '#ff7f0e'}

# ===== BIỂU ĐỒ 1: LOSS (2 ĐƯỜNG ĐI XUỐNG) =====
ax1 = axes[0, 0]
ax1.plot(df['epoch'], df['train_loss'], color=colors['train'], linewidth=2.5, 
         marker='o', markersize=4, label='Train Loss', markevery=3)
ax1.plot(df['epoch'], df['val_loss'], color=colors['val'], linewidth=2.5, 
         marker='s', markersize=4, label='Validation Loss', markevery=3)
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('Loss', fontsize=14)
ax1.set_title('BIỂU ĐỒ 1: TRAINING AND VALIDATION LOSS', fontsize=16, fontweight='bold')
ax1.legend(loc='upper right', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-2, 62)
ax1.set_ylim(0, 0.8)

# Thêm giá trị đầu và cuối
ax1.annotate(f'{df["train_loss"].iloc[0]:.3f}', xy=(0, df["train_loss"].iloc[0]), 
             xytext=(5, df["train_loss"].iloc[0]+0.05),
             arrowprops=dict(arrowstyle='->', color=colors['train']), fontsize=11, color=colors['train'])
ax1.annotate(f'{df["train_loss"].iloc[-1]:.3f}', xy=(59, df["train_loss"].iloc[-1]), 
             xytext=(45, df["train_loss"].iloc[-1]-0.08),
             arrowprops=dict(arrowstyle='->', color=colors['train']), fontsize=11, color=colors['train'])

# ===== BIỂU ĐỒ 2: ACCURACY (2 ĐƯỜNG ĐI LÊN) =====
ax2 = axes[0, 1]
ax2.plot(df['epoch'], df['train_pixel_AUROC'], color=colors['train'], linewidth=2.5, 
         marker='o', markersize=4, label='Train Accuracy (AUROC)', markevery=3)
ax2.plot(df['epoch'], df['val_pixel_AUROC'], color=colors['val'], linewidth=2.5, 
         marker='s', markersize=4, label='Validation Accuracy (AUROC)', markevery=3)
ax2.set_xlabel('Epoch', fontsize=14)
ax2.set_ylabel('Accuracy (AUROC)', fontsize=14)
ax2.set_title('BIỂU ĐỒ 2: PIXEL-LEVEL ACCURACY (AUROC)', fontsize=16, fontweight='bold')
ax2.legend(loc='lower right', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-2, 62)
ax2.set_ylim(0.8, 1.0)

# Thêm giá trị đầu và cuối
ax2.annotate(f'{df["train_pixel_AUROC"].iloc[0]:.3f}', xy=(0, df["train_pixel_AUROC"].iloc[0]), 
             xytext=(5, df["train_pixel_AUROC"].iloc[0]-0.02),
             arrowprops=dict(arrowstyle='->', color=colors['train']), fontsize=11, color=colors['train'])
ax2.annotate(f'{df["train_pixel_AUROC"].iloc[-1]:.3f}', xy=(59, df["train_pixel_AUROC"].iloc[-1]), 
             xytext=(45, df["train_pixel_AUROC"].iloc[-1]+0.01),
             arrowprops=dict(arrowstyle='->', color=colors['train']), fontsize=11, color=colors['train'])

# ===== BIỂU ĐỒ 3: PIXEL F1 =====
ax3 = axes[1, 0]
ax3.plot(df['epoch'], df['train_pixel_F1'], color=colors['train'], linewidth=2, 
         marker='o', markersize=4, label='Train Pixel F1', markevery=3)
ax3.plot(df['epoch'], df['val_pixel_F1'], color=colors['val'], linewidth=2, 
         marker='s', markersize=4, label='Validation Pixel F1', markevery=3)
ax3.set_xlabel('Epoch', fontsize=14)
ax3.set_ylabel('F1 Score', fontsize=14)
ax3.set_title('BIỂU ĐỒ 3: PIXEL-LEVEL F1 SCORE', fontsize=16, fontweight='bold')
ax3.legend(loc='lower right', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-2, 62)
ax3.set_ylim(0.15, 0.35)

# ===== BIỂU ĐỒ 4: IMAGE AUROC =====
ax4 = axes[1, 1]
ax4.plot(df['epoch'], df['train_image_AUROC'], color=colors['train'], linewidth=2, 
         marker='o', markersize=4, label='Train Image AUROC', markevery=3)
ax4.plot(df['epoch'], df['val_image_AUROC'], color=colors['val'], linewidth=2, 
         marker='s', markersize=4, label='Validation Image AUROC', markevery=3)
ax4.set_xlabel('Epoch', fontsize=14)
ax4.set_ylabel('AUROC', fontsize=14)
ax4.set_title('BIỂU ĐỒ 4: IMAGE-LEVEL AUROC', fontsize=16, fontweight='bold')
ax4.legend(loc='lower right', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-2, 62)
ax4.set_ylim(0.5, 1.05)

plt.tight_layout()
plt.savefig('training_60_epochs_complete.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Đã lưu biểu đồ: training_60_epochs_complete.png")

# ============================================
# BƯỚC 4: VẼ BIỂU ĐỒ RIÊNG CHO LOSS VÀ ACCURACY
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('LOSS AND ACCURACY OVER 60 EPOCHS', fontsize=18, fontweight='bold')

# Biểu đồ loss
axes[0].plot(df['epoch'], df['train_loss'], color=colors['train'], linewidth=2.5, 
             marker='o', markersize=4, label='Train Loss', markevery=4)
axes[0].plot(df['epoch'], df['val_loss'], color=colors['val'], linewidth=2.5, 
             marker='s', markersize=4, label='Validation Loss', markevery=4)
axes[0].set_xlabel('Epoch', fontsize=14)
axes[0].set_ylabel('Loss', fontsize=14)
axes[0].set_title('LOSS CURVES', fontsize=16, fontweight='bold')
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(-2, 62)

# Biểu đồ accuracy
axes[1].plot(df['epoch'], df['train_pixel_AUROC'], color=colors['train'], linewidth=2.5, 
             marker='o', markersize=4, label='Train Accuracy (AUROC)', markevery=4)
axes[1].plot(df['epoch'], df['val_pixel_AUROC'], color=colors['val'], linewidth=2.5, 
             marker='s', markersize=4, label='Validation Accuracy (AUROC)', markevery=4)
axes[1].set_xlabel('Epoch', fontsize=14)
axes[1].set_ylabel('Accuracy (AUROC)', fontsize=14)
axes[1].set_title('ACCURACY CURVES', fontsize=16, fontweight='bold')
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(-2, 62)
axes[1].set_ylim(0.8, 1.0)

plt.tight_layout()
plt.savefig('loss_accuracy_60_epochs.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Đã lưu biểu đồ: loss_accuracy_60_epochs.png")

# ============================================
# BƯỚC 5: XUẤT THỐNG KÊ
# ============================================

print("\n" + "="*70)
print("THỐNG KÊ LOSS VÀ ACCURACY - 60 EPOCHS")
print("="*70)
print(f"Train Loss: {df['train_loss'].iloc[0]:.4f} → {df['train_loss'].iloc[-1]:.4f} (giảm {df['train_loss'].iloc[0]-df['train_loss'].iloc[-1]:.4f})")
print(f"Val Loss:   {df['val_loss'].iloc[0]:.4f} → {df['val_loss'].iloc[-1]:.4f} (giảm {df['val_loss'].iloc[0]-df['val_loss'].iloc[-1]:.4f})")
print(f"Train Accuracy: {df['train_pixel_AUROC'].iloc[0]:.4f} → {df['train_pixel_AUROC'].iloc[-1]:.4f} (tăng {df['train_pixel_AUROC'].iloc[-1]-df['train_pixel_AUROC'].iloc[0]:.4f})")
print(f"Val Accuracy:   {df['val_pixel_AUROC'].iloc[0]:.4f} → {df['val_pixel_AUROC'].iloc[-1]:.4f} (tăng {df['val_pixel_AUROC'].iloc[-1]-df['val_pixel_AUROC'].iloc[0]:.4f})")
print("="*70)

print("\n✅ CÁC FILE ĐÃ TẠO:")
print("   - metrics_60_epochs_complete.csv (file log 60 epochs)")
print("   - training_60_epochs_complete.png (4 biểu đồ tổng hợp)")
print("   - loss_accuracy_60_epochs.png (2 biểu đồ loss và accuracy riêng)")