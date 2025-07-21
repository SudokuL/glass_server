# -*- coding: gbk -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import regnet_y_16gf, regnet_y_32gf, regnet_y_128gf
from torchvision.transforms import v2, InterpolationMode
from PIL import Image
import numpy as np
import pandas as pd
import os
import pickle

# === ??? ===
class_names = pd.read_csv("/root/autodl-tmp/.autodl/iot/food_cv/class_names.xlsx", header=None)[1].tolist()

# === ???? ===
NUM_CLASSES = 241
IMG_SIZE = 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === ???? ===
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

# === ????? ===
transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# === ???? ===
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

    # Temporarily add FocalLoss to __main__ module to fix loading
    import sys
    import __main__
    if not hasattr(__main__, 'FocalLoss'):
        __main__.FocalLoss = FocalLoss
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        if 'model_dict' in checkpoint:
            state_dict = checkpoint['model_dict']
        else:
            state_dict = checkpoint  # ???????? state_dict ??
        model.load_state_dict(state_dict)
    finally:
        # Clean up the temporary reference
        if hasattr(__main__, 'FocalLoss'):
            delattr(__main__, 'FocalLoss')
    
    model.eval()
    return model

# === ??????? ===
def load_nutrient_labels(nutrient_txt_path):
    if not os.path.exists(nutrient_txt_path):
        print("?? ????????????????????????")
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

# === ?? ===
def predict(model, image_path, nutrient_label):
    image = Image.open(image_path).convert('RGB')
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        top3_indices = np.argsort(probs)[-3:][::-1]
        top3_probs = probs[top3_indices]

        # print("\n? Top-3 ?????")
        # for i in range(3):
        #     print(f"  - ?? {top3_indices[i]}?{class_names[top3_indices[i]]}????{top3_probs[i]:.4f}")

        norm_probs = top3_probs / top3_probs.sum()
        nutrient_estimate = np.dot(norm_probs, nutrient_label[top3_indices])
        # print("\n? Top-3 ????????24??:")
        # for idx, val in enumerate(nutrient_estimate):
        #     print(f"  [{idx+1:02}] = {val:.2f}")

        return top3_indices, top3_probs, nutrient_estimate

# === ???????????? ===
nutrient_names = [
    "????(kcal)", "??????(g)", "???(g)", "????????(g)", "??????(g)", "?????A(??g)", 
    "?????B1(mg)", "?????B2(mg)", "?????B6(mg)", "?????B12(??g)", "?????C(mg)", 
    "?????D(??g)", "?????E(mg)", "?????K(??g)", "??(mg)", "??(mg)", "??(mg)", 
    "??(??g)", "??(mg)", "??(mg)", "?(mg)", "??(mg)", "?(mg)", "??(mg)"
]

# === ????? ===
class FoodRecognizer:
    def __init__(self):
        self.MODEL_PATH = "/root/autodl-tmp/.autodl/iot/food_cv/SEP14_y128y32y16_best.pt"
        self.NUTRIENT_TXT = "/root/autodl-tmp/.autodl/iot/food_cv/new nutrient_label 2.txt"
        self.model = None
        self.nutrient_label = None
        self._load_model_and_nutrients()
    
    def _load_model_and_nutrients(self):
        """??????????"""
        print("[FoodCV] ?????...")
        self.model = load_model(self.MODEL_PATH)
        print("[FoodCV] ???????...")
        self.nutrient_label = load_nutrient_labels(self.NUTRIENT_TXT)
        print("[FoodCV] ??????????")
    
    def recognize_food(self, image_path):
        """?????????"""
        try:
            top3_indices, top3_probs, nutrient_estimate = predict(self.model, image_path, self.nutrient_label)
            
            # ??????
            result = {
                "dish_name": class_names[top3_indices[0]],
                "confidence": float(top3_probs[0]),
                "top3_predictions": [
                    {
                        "name": class_names[top3_indices[i]],
                        "confidence": float(top3_probs[i])
                    } for i in range(3)
                ],
                "nutrients": {
                    nutrient_names[i]: float(nutrient_estimate[i]) for i in range(len(nutrient_names))
                }
            }
            
            print(f"[FoodCV] ????: {result['dish_name']} (???: {result['confidence']:.4f})")
            return result
            
        except Exception as e:
            print(f"[FoodCV] ??????: {str(e)}")
            return None

# === ????? ===
if __name__ == "__main__":
    recognizer = FoodRecognizer()
    
    IMG_DIR = "/root/autodl-tmp/.autodl/iot/food_img/"
    print(f"? ??????????: {IMG_DIR}")
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if os.path.splitext(f)[1].lower() in valid_exts]

    if not image_files:
        print("?? ????????????????????")
    else:
        for idx, img_path in enumerate(sorted(image_files)):
            print(f"\n===== ?? ? {idx+1} ????{os.path.basename(img_path)} =====")
            result = recognizer.recognize_food(img_path)
            if result:
                print(f"? Top-1 ??{result['dish_name']}????{result['confidence']:.4f}")
