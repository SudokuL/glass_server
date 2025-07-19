import kagglehub

# Download latest version
path = kagglehub.model_download("yihfeng/chinese-dish-nutrient-estimation/pyTorch/default")

print("Path to model files:", "/root/autodl-tmp/.autodl/iot/food_cv/"+path)