
# -*- coding: gbk -*-
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
import torch

# ---- 离线模型目录 ----
local_model_path = "/root/autodl-tmp/.autodl/iot/qwen_7b_vl_offline"

# 加载 Processor
processor = AutoProcessor.from_pretrained(local_model_path)

# 加载 Qwen2.5-VL 多模态生成模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    local_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)

# 准备 LoRA 微调：冻结原模型权重，仅适配 k-bit
model = prepare_model_for_kbit_training(model)

# ---- 构造测试输入 ----
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            },
            {
                "type": "text",
                "text": "请描述这张图片。"
            }
        ]
    }
]

# 文本 + 图像 预处理
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)

# 推理并打印结果
with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=128)
    output_ids = generated[0][inputs.input_ids.shape[1]:]
    output_text = processor.batch_decode(
        [output_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

print("\n? 模型输出：", output_text)
