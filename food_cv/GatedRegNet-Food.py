# -*- coding: gbk -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from torchvision.models import regnet_y_16gf, regnet_y_32gf, regnet_y_128gf
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import pandas as pd

# === 类别名 ===
class_names = pd.read_csv("/root/autodl-tmp/.autodl/iot/food_cv/class_names.xlsx", header=None)[1].tolist()

# === 参数设置 ===
NUM_CLASSES = 241
IMG_SIZE = 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 模型结构 ===
class GatedHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedHead, self).__init__()
        self.gate = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        gate_values = self.sigmoid(self.gate(x))
        gated_output = gate_values * x
        return self.fc(self.relu(gated_output))

class CombinedModel(nn.Module):
    def __init__(self, model1, model2, model3, gated_head):
        super(CombinedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.gated_head = gated_head
    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        combined = torch.cat((out1, out2, out3), dim=1)
        return self.gated_head(combined) + out1 / 3 + out2 / 3 + out3 / 3

# === 图像预处理 ===
transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# === 加载模型 ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction=self.reduction)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss

def load_model(model_path):
    model1 = regnet_y_16gf(weights=None)
    model2 = regnet_y_32gf(weights=None)
    model3 = regnet_y_128gf(weights=None)
    model1.fc = nn.Linear(model1.fc.in_features, NUM_CLASSES)
    model2.fc = nn.Linear(model2.fc.in_features, NUM_CLASSES)
    model3.fc = nn.Linear(model3.fc.in_features, NUM_CLASSES)
    for m in [model1, model2, model3]:
        for p in m.parameters():
            p.requires_grad = False
    gated_head = GatedHead(input_dim=3 * NUM_CLASSES, output_dim=NUM_CLASSES)
    model = CombinedModel(model1, model2, model3, gated_head)
    model = nn.DataParallel(model).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_dict' in checkpoint:
        state_dict = checkpoint['model_dict']
    else:
        state_dict = checkpoint  # 如果直接保存的是 state_dict 本身
    model.load_state_dict(state_dict)
    model.load_state_dict(checkpoint['model_dict'])
    model.eval()
    return model

# === 加载营养素标签 ===
def load_nutrient_labels(nutrient_txt_path):
    if not os.path.exists(nutrient_txt_path):
        print("?? 未检测到真实营养标签文件，使用随机模拟营养素数据")
        np.random.seed(42)
        return np.random.uniform(10, 200, size=(241, 24)).round(2)

    with open(nutrient_txt_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        s = line
        values = []
        while True:
            idx1 = s.find("=")
            idx2 = s.find(",")
            if idx1 == -1 or idx2 == -1:
                break
            val = float(s[idx1+1:idx2])
            values.append(round(val, 2))
            s = s[idx2+1:]
        indices = [2,3,4,6,7,8,9,10,12,15,16,17,18,19,23,24,25,26,27,28,29,30,31,32]
        values = np.array(values)[indices]
        data.append(values)
    return np.array(data)

# === 推理 ===
def predict(model, image_path, nutrient_label):
    image = Image.open(image_path).convert('RGB')
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        top3_indices = np.argsort(probs)[-3:][::-1]
        top3_probs = probs[top3_indices]

        print("\n? Top-3 预测结果：")
        for i in range(3):
            print(f"  - 类别 {top3_indices[i]}：{class_names[top3_indices[i]]}，概率：{top3_probs[i]:.4f}")

        norm_probs = top3_probs / top3_probs.sum()
        nutrient_estimate = np.dot(norm_probs, nutrient_label[top3_indices])
        print("\n? Top-3 融合营养素估计（24项）:")
        for idx, val in enumerate(nutrient_estimate):
            print(f"  [{idx+1:02}] = {val:.2f}")

        return top3_indices, top3_probs, nutrient_estimate

# === 主程序入口 ===
if __name__ == "__main__":
    MODEL_PATH = "/root/autodl-tmp/.autodl/iot/food_cv/SEP14_y128y32y16_best.pt"
    IMG_DIR = "/root/autodl-tmp/.autodl/iot/food_img/"
    NUTRIENT_TXT = "/root/autodl-tmp/.autodl/iot/food_cv/new nutrient_label 2.txt"

    print("? 加载模型中...")
    model = load_model(MODEL_PATH)

    print("? 加载营养素标签...")
    nutrient_label = load_nutrient_labels(NUTRIENT_TXT)

    print(f"? 扫描目录中的图像文件: {IMG_DIR}")
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if os.path.splitext(f)[1].lower() in valid_exts]

    if not image_files:
        print("?? 未找到任何图像文件，请确认路径是否正确。")
    else:
        for idx, img_path in enumerate(sorted(image_files)):
            print(f"\n===== ?? 第 {idx+1} 张图像：{os.path.basename(img_path)} =====")
            try:
                top3_indices, top3_probs, nutrient_estimate = predict(model, img_path, nutrient_label)
                print(f"? Top-1 是：{class_names[top3_indices[0]]}，概率：{top3_probs[0]:.4f}")
            except Exception as e:
                print(f"? 图像处理失败：{img_path}")
                print(f"错误信息：{str(e)}")
