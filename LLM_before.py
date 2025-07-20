# -*- coding: gbk -*-
# LLM�ı�������TTS�Խ�������
# ���ܣ���������ģ�������ı�����ͨ���ٶ�TTS�ӿ�ת��Ϊ�����ļ�

import os           # �����ļ�·���ͻ�����������
import torch        # PyTorch���ѧϰ��ܣ�����ģ�ͼ��غ�����
import re           # ������ʽģ�飬�����ı���ϴ

# ========== ·�����ã���ģ�ͻ��浽������ ==========
# ˵������Hugging Face��ػ���Ŀ¼���õ������̣�����ռ��ϵͳ�̿ռ�
data_disk_path = "/root/autodl-tmp/.autodl/hf_cache"
os.environ['TRANSFORMERS_CACHE'] = data_disk_path          # transformers�⻺��·��
os.environ['HF_HOME'] = data_disk_path                     # Hugging Face������·��
os.environ['HF_DATASETS_CACHE'] = os.path.join(data_disk_path, "datasets")  # ���ݼ�����·��
os.environ['HF_METRICS_CACHE'] = os.path.join(data_disk_path, "metrics")    # ָ�껺��·��
# ==================================================

# ʹ�ù��ھ������
# ˵��������Hugging Face���ھ��񣬼���ģ������
os.environ['HF_HUB_OFFLINE'] = '0'                         # �ر�����ģʽ
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'         # ���ù��ھ����ַ

from transformers import (
    AutoTokenizer,          # �Զ�����Ԥѵ���ִ���
    AutoModelForCausalLM,   # �Զ������������ģ��
    pipeline                # ��������ܵ�����
)
from baidu_tts import baidu_tts_test  # ����ٶ�TTS�����ϳɺ���

# ------------ ģ���������� ------------
# ˵��������Ҫʹ�õ�Ԥѵ������ģ��
model_name = "hfl/llama-3-chinese-8b-instruct-v3"  # ģ�����ƣ�����Llama-3 8Bָ��΢����
# -------------------------------------

print("�ӱ��ؼ��� tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=False,    # �����Զ�����أ��������û�У�
    trust_remote_code=True     # ����Զ�̴��루����Զ���ģ�ͣ�
)  # ���طִ��������ı�ת��Ϊģ�Ϳ�����token

print("�ӱ��ؼ���ģ�ͣ�FP16 ���ȣ�...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    local_files_only=False,        # �����Զ�����أ��������û�У�
    torch_dtype=torch.float16,     # ʹ�ð뾫��FP16�������Դ�ռ��
    device_map="auto",            # �Զ������豸��GPU���ȣ�
    trust_remote_code=True         # ����Զ�̴��루����Զ���ģ�ͣ�
)  # ����Ԥѵ������ģ�ͣ������ı���������

print("��������ܵ�...")
pipe = pipeline(
    task="text-generation",       # �������ͣ��ı�����
    model=model,                   # ʹ�ü��ص�����ģ��
    tokenizer=tokenizer            # ʹ�ü��صķִ���
)  # ��������ܵ������ı����ɹ��̵ĵ��ýӿ�

# �ı����ɲ���
prompt = "����һ�������"  # ������ʾ�ʣ�����ģ�������ض�����
result = pipe(
    prompt,
    max_new_tokens=200,            # ������ɳ��ȣ�200��token
    do_sample=True,                # ���ò������ɣ���ȷ���ԣ�
    temperature=0.7,               # �¶Ȳ�������������ԣ�0-1��ֵԽ��Խ�����
    return_full_text=False         # �����������ɵ��ı���������ԭʼprompt
)  # ��������ܵ������ı�

# �ı���ϴ���Ƴ�Emoji���Ʊ�����س��ͻ��з�
generated_text = re.sub(r'[\U00010000-\U0010ffff\t\r\n]', '', result[0]["generated_text"]).strip()
print(generated_text)  # ��ӡ��ϴ��������ı�
# ����TTS�ӿ�����MP3�ļ�
baidu_tts_test(generated_text)  # �����ɵ��ı�ת��Ϊ����������Ϊresult.mp3
