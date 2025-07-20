
# -*- coding: gbk -*-
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
import torch

# ---- ����ģ��Ŀ¼ ----
local_model_path = "/root/autodl-tmp/.autodl/iot/qwen_7b_vl_offline"

# ���� Processor
processor = AutoProcessor.from_pretrained(local_model_path)

# ���� Qwen2.5-VL ��ģ̬����ģ��
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    local_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)

# ׼�� LoRA ΢��������ԭģ��Ȩ�أ������� k-bit
model = prepare_model_for_kbit_training(model)

# ---- ����������� ----
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
                "text": "����������ͼƬ��"
            }
        ]
    }
]

# �ı� + ͼ�� Ԥ����
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

# ������ӡ���
with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=128)
    output_ids = generated[0][inputs.input_ids.shape[1]:]
    output_text = processor.batch_decode(
        [output_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

print("\n? ģ�������", output_text)
