# -*- coding: gbk -*-
# LLM文本生成与TTS对接主程序
# 功能：加载语言模型生成文本，并通过百度TTS接口转换为语音文件

import os           # 用于文件路径和环境变量操作
import torch        # PyTorch深度学习框架，用于模型加载和推理
import re           # 正则表达式模块，用于文本清洗

# ========== 路径设置：将模型缓存到数据盘 ==========
# 说明：将Hugging Face相关缓存目录设置到数据盘，避免占用系统盘空间
data_disk_path = "/root/autodl-tmp/.autodl/hf_cache"
os.environ['TRANSFORMERS_CACHE'] = data_disk_path          # transformers库缓存路径
os.environ['HF_HOME'] = data_disk_path                     # Hugging Face主缓存路径
os.environ['HF_DATASETS_CACHE'] = os.path.join(data_disk_path, "datasets")  # 数据集缓存路径
os.environ['HF_METRICS_CACHE'] = os.path.join(data_disk_path, "metrics")    # 指标缓存路径
# ==================================================

# 使用国内镜像加速
# 说明：配置Hugging Face国内镜像，加速模型下载
os.environ['HF_HUB_OFFLINE'] = '0'                         # 关闭离线模式
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'         # 设置国内镜像地址

from transformers import (
    AutoTokenizer,          # 自动加载预训练分词器
    AutoModelForCausalLM,   # 自动加载因果语言模型
    pipeline                # 构建推理管道工具
)
from baidu_tts import baidu_tts_test  # 导入百度TTS语音合成函数

# ------------ 模型设置区域 ------------
# 说明：配置要使用的预训练语言模型
model_name = "hfl/llama-3-chinese-8b-instruct-v3"  # 模型名称：中文Llama-3 8B指令微调版
# -------------------------------------

print("从本地加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=False,    # 允许从远程下载（如果本地没有）
    trust_remote_code=True     # 信任远程代码（针对自定义模型）
)  # 加载分词器：将文本转换为模型可理解的token

print("从本地加载模型（FP16 精度）...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    local_files_only=False,        # 允许从远程下载（如果本地没有）
    torch_dtype=torch.float16,     # 使用半精度FP16，减少显存占用
    device_map="auto",            # 自动分配设备（GPU优先）
    trust_remote_code=True         # 信任远程代码（针对自定义模型）
)  # 加载预训练语言模型：用于文本生成推理

print("构建推理管道...")
pipe = pipeline(
    task="text-generation",       # 任务类型：文本生成
    model=model,                   # 使用加载的语言模型
    tokenizer=tokenizer            # 使用加载的分词器
)  # 创建推理管道：简化文本生成过程的调用接口

# 文本生成测试
prompt = "介绍一下刘瑞达"  # 输入提示词：引导模型生成特定内容
result = pipe(
    prompt,
    max_new_tokens=200,            # 最大生成长度：200个token
    do_sample=True,                # 启用采样生成（非确定性）
    temperature=0.7,               # 温度参数：控制随机性（0-1，值越高越随机）
    return_full_text=False         # 仅返回新生成的文本，不包含原始prompt
)  # 调用推理管道生成文本

# 文本清洗：移除Emoji、制表符、回车和换行符
generated_text = re.sub(r'[\U00010000-\U0010ffff\t\r\n]', '', result[0]["generated_text"]).strip()
print(generated_text)  # 打印清洗后的生成文本
# 调用TTS接口生成MP3文件
baidu_tts_test(generated_text)  # 将生成的文本转换为语音并保存为result.mp3
