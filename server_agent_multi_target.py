# -*- coding: gbk -*-
# server_agent.py  — 实时 PART+FINAL 输出 + Qwen2.5-VL 调用 + TTS
#!/usr/bin/env python3
#python /root/autodl-tmp/.autodl/iot/server_agent.py
import asyncio, websockets, json, os, time
from datetime import datetime
from vosk import Model, KaldiRecognizer
from aiohttp import web, web_runner
import aiofiles
import cv2

# --- Qwen2.5-VL 相关 ---
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# --- Baidu TTS ---
import baidu_tts

# --- 食品识别 ---
import sys
sys.path.append('/root/autodl-tmp/.autodl/iot/food_cv')
from food_cv.GatedRegNet_Food import FoodRecognizer

# --- 多目标食物检测 ---
from multi_target_food_detector import MultiTargetFoodDetector

# --- 视频处理 ---
import glob
import random

# 菜品营养数据库
DISH_NUTRITION_DATABASE = {
    # 高蛋白质菜品
    '红烧肉': { 'protein': 16.2, 'fat': 20.5, 'carbs': 8.1, 'fiber': 0.8, 'calcium': 15, 'iron': 3.2, 'category': 'meat' },
    '白切鸡': { 'protein': 23.3, 'fat': 9.3, 'carbs': 0.0, 'fiber': 0.0, 'calcium': 12, 'iron': 1.4, 'category': 'meat' },
    '蒸蛋羹': { 'protein': 13.1, 'fat': 11.2, 'carbs': 1.2, 'fiber': 0.0, 'calcium': 56, 'iron': 2.8, 'category': 'egg' },
    '鲫鱼汤': { 'protein': 17.1, 'fat': 2.7, 'carbs': 0.0, 'fiber': 0.0, 'calcium': 79, 'iron': 1.3, 'category': 'fish' },
    '牛肉炖土豆': { 'protein': 18.6, 'fat': 12.4, 'carbs': 15.2, 'fiber': 2.1, 'calcium': 18, 'iron': 3.8, 'category': 'meat' },
    
    # 高钙菜品
    '麻婆豆腐': { 'protein': 7.9, 'fat': 11.1, 'carbs': 6.0, 'fiber': 1.1, 'calcium': 68.6, 'iron': 2.2, 'category': 'tofu' },
    '虾仁豆腐': { 'protein': 15.8, 'fat': 8.3, 'carbs': 4.2, 'fiber': 0.5, 'calcium': 156, 'iron': 2.1, 'category': 'seafood' },
    '芝麻糊': { 'protein': 4.8, 'fat': 12.6, 'carbs': 22.1, 'fiber': 3.2, 'calcium': 218, 'iron': 4.6, 'category': 'dessert' },
    '紫菜蛋花汤': { 'protein': 8.2, 'fat': 4.1, 'carbs': 2.8, 'fiber': 1.8, 'calcium': 89, 'iron': 12.8, 'category': 'soup' },
    '酸奶水果': { 'protein': 5.2, 'fat': 3.8, 'carbs': 18.4, 'fiber': 2.1, 'calcium': 125, 'iron': 0.3, 'category': 'dairy' },
    
    # 高铁菜品
    '猪肝炒菠菜': { 'protein': 18.2, 'fat': 8.9, 'carbs': 6.4, 'fiber': 2.8, 'calcium': 28, 'iron': 18.6, 'category': 'organ' },
    '韭菜炒蛋': { 'protein': 11.8, 'fat': 12.3, 'carbs': 4.1, 'fiber': 2.4, 'calcium': 42, 'iron': 6.2, 'category': 'vegetable' },
    '黑木耳炒肉': { 'protein': 13.6, 'fat': 9.8, 'carbs': 8.2, 'fiber': 4.2, 'calcium': 38, 'iron': 8.9, 'category': 'fungus' },
    '红枣银耳汤': { 'protein': 2.8, 'fat': 0.4, 'carbs': 28.6, 'fiber': 5.1, 'calcium': 45, 'iron': 4.8, 'category': 'dessert' },
    '芝麻菠菜': { 'protein': 6.2, 'fat': 8.4, 'carbs': 9.1, 'fiber': 3.8, 'calcium': 86, 'iron': 7.2, 'category': 'vegetable' },
    
    # 高纤维菜品
    '蒜蓉西兰花': { 'protein': 4.3, 'fat': 2.1, 'carbs': 6.6, 'fiber': 4.2, 'calcium': 47, 'iron': 1.2, 'category': 'vegetable' },
    '凉拌芹菜': { 'protein': 2.2, 'fat': 0.8, 'carbs': 4.8, 'fiber': 3.2, 'calcium': 80, 'iron': 2.5, 'category': 'vegetable' },
    '木耳菜': { 'protein': 2.8, 'fat': 0.5, 'carbs': 5.4, 'fiber': 2.8, 'calcium': 166, 'iron': 3.2, 'category': 'vegetable' },
    '冬瓜汤': { 'protein': 1.2, 'fat': 0.2, 'carbs': 2.9, 'fiber': 1.8, 'calcium': 19, 'iron': 0.3, 'category': 'soup' },
    '萝卜丝汤': { 'protein': 1.6, 'fat': 0.3, 'carbs': 4.1, 'fiber': 2.4, 'calcium': 24, 'iron': 0.8, 'category': 'soup' },
    
    # 低脂低糖菜品
    '清蒸鲈鱼': { 'protein': 18.6, 'fat': 3.4, 'carbs': 0.0, 'fiber': 0.0, 'calcium': 138, 'iron': 1.2, 'category': 'fish' },
    '白灼菜心': { 'protein': 2.8, 'fat': 0.4, 'carbs': 3.2, 'fiber': 1.8, 'calcium': 108, 'iron': 1.8, 'category': 'vegetable' },
    '蒸南瓜': { 'protein': 1.2, 'fat': 0.1, 'carbs': 5.3, 'fiber': 1.4, 'calcium': 16, 'iron': 0.4, 'category': 'vegetable' },
    '黄瓜汤': { 'protein': 0.8, 'fat': 0.2, 'carbs': 2.0, 'fiber': 0.8, 'calcium': 15, 'iron': 0.3, 'category': 'soup' }
}

# 离线模型路径
QWEN_MODEL_PATH = "/root/autodl-tmp/.autodl/iot/qwen_7b_vl_offline"
# TTS 临时文件
TTS_FILE = "result.mp3"

print("[Srv] Loading Qwen2.5-VL model …")
processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)
qwen = prepare_model_for_kbit_training(qwen)
qwen.eval()
print("[Srv] Qwen2.5-VL ready on", qwen.device)

# 初始化食品识别器
print("[Srv] Loading FoodRecognizer...")
food_recognizer = FoodRecognizer()
print("[Srv] FoodRecognizer ready")

# 初始化多目标食物检测器
print("[Srv] Loading MultiTargetFoodDetector...")
multi_target_detector = MultiTargetFoodDetector(silent_mode=True)
print("[Srv] MultiTargetFoodDetector ready")

# 麦粒唤醒词模糊音识别
def detect_maili_wakeup(text):
    """检测麦粒唤醒词及其模糊音"""
    wakeup_words = ["麦粒", "卖力", "麦莉", "外力", "玛丽", "买力", "迈力", "麦力","麦蒂"]
    for word in wakeup_words:
        if word in text:
            return True
    return False


def query_qwen(messages):
    """调用 Qwen，messages 前注入系统 Prompt"""
    system_prompt = { "role": "system", "content": "你叫“麦粒”，嵌入在我的智能眼镜中，是我的私人语音助手。同时你是用户的私人营养师，精通各种营养学知识。你的目的是帮助用户健康的饮食，提出建议等。用户的文字很可能包含错别字，如果你理解不通顺请按照发音异议理解。" }
    all_msgs = [system_prompt] + messages
    text = processor.apply_chat_template(all_msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(all_msgs)
    inputs = processor(
        text=[text], images=imgs, videos=vids,
        padding=True, return_tensors="pt"
    ).to(qwen.device)
    with torch.no_grad():
        gen = qwen.generate(**inputs, max_new_tokens=512)
    out_ids = gen[0][inputs.input_ids.shape[1]:]
    return processor.batch_decode(
        [out_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

# --- Vosk 识别相关 ---
MODEL_PATH  = "model/vosk-model-cn-0.22"
SAMPLE_RATE = 16000
vosk_model = Model(MODEL_PATH)

PHOTO_DIR = "photos"; VIDEO_DIR = "video_frames"; MEETING_DIR = "meeting_recording"; FOOD_DIR = "food_recognition"; VIDEOS_DIR = "videos"; NUTRITION_DIR = "nutrition_analysis"
os.makedirs(PHOTO_DIR, exist_ok=True); os.makedirs(VIDEO_DIR, exist_ok=True); os.makedirs(MEETING_DIR, exist_ok=True); os.makedirs(FOOD_DIR, exist_ok=True); os.makedirs(VIDEOS_DIR, exist_ok=True); os.makedirs(NUTRITION_DIR, exist_ok=True)


def save_photo(img: bytes) -> str:
    fn = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
    path = os.path.join(PHOTO_DIR, fn)
    with open(path, "wb") as f: f.write(img)
    print(f"[Srv] ? saved {fn}")
    return path


def save_frame(img: bytes):
    fn = datetime.now().strftime("%Y%m%d_%H%M%S_%f.jpg")
    path = os.path.join(VIDEO_DIR, fn)
    with open(path, "wb") as f: f.write(img)
    return path

def create_video_from_frames(video_session_id: str) -> str:
    """将视频帧合成为视频文件"""
    frame_pattern = os.path.join(VIDEO_DIR, f"{video_session_id}_*.jpg")
    frame_files = sorted(glob.glob(frame_pattern))
    
    if len(frame_files) < 2:
        print(f"[Srv] 视频帧数量不足: {len(frame_files)}")
        return None
    
    # 读取第一帧获取尺寸
    first_frame = cv2.imread(frame_files[0])
    height, width, layers = first_frame.shape
    
    # 创建视频文件
    video_filename = f"{video_session_id}.mp4"
    video_path = os.path.join(VIDEOS_DIR, video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
    
    # 写入所有帧
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is not None:
            video_writer.write(frame)
    
    video_writer.release()
    
    # 清理临时帧文件
    for frame_file in frame_files:
        try:
            os.remove(frame_file)
        except:
            pass
    
    print(f"[Srv] 视频已生成: {video_path}")
    return video_path

def start_meeting_recording():
    """开始会议记录"""
    fn = datetime.now().strftime("%Y%m%d_%H%M%S_meeting.txt")
    path = os.path.join(MEETING_DIR, fn)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"会议记录开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")
    print(f"[Srv] 会议记录开始: {fn}")
    return path

def append_meeting_text(meeting_file: str, text: str):
    """追加会议记录内容"""
    if meeting_file and os.path.exists(meeting_file):
        with open(meeting_file, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] {text}\n")
        print(f"[Srv] 会议记录追加: {text}")

def end_meeting_recording(meeting_file: str):
    """结束会议记录"""
    if meeting_file and os.path.exists(meeting_file):
        with open(meeting_file, "a", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write(f"会议记录结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"[Srv] 会议记录结束: {meeting_file}")

def get_weekly_nutrition_summary():
    """获取本周营养摄入汇总"""
    from datetime import datetime, timedelta
    import glob
    
    # 计算本周的开始和结束时间
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # 获取本周的食品识别记录
    food_files = glob.glob(os.path.join(FOOD_DIR, "food_*.json"))
    weekly_nutrition = {
        "热量(kcal)": 0, "蛋白质(g)": 0, "脂肪(g)": 0, "碳水化合物(g)": 0,
        "膳食纤维(g)": 0, "维生素A(μg)": 0, "维生素C(mg)": 0, "钙(mg)": 0,
        "铁(mg)": 0, "锌(mg)": 0
    }
    food_records = []
    
    for food_file in food_files:
        try:
            with open(food_file, "r", encoding="utf-8") as f:
                food_data = json.load(f)
            
            # 解析时间戳
            food_time = datetime.strptime(food_data["timestamp"], "%Y%m%d_%H%M%S")
            
            # 检查是否在本周内
            if food_time >= week_start:
                nutrients = food_data.get("nutrients", {})
                food_records.append({
                    "dish_name": food_data.get("dish_name", "未知食品"),
                    "datetime": food_data.get("datetime", ""),
                    "nutrients": nutrients
                })
                
                # 累加营养素
                for nutrient, value in nutrients.items():
                    if nutrient in weekly_nutrition:
                        weekly_nutrition[nutrient] += float(value)
        except Exception as e:
            print(f"[Nutrition] 解析食品记录失败: {food_file}, {e}")
    
    return {
        "week_start": week_start.strftime("%Y-%m-%d"),
        "total_nutrition": weekly_nutrition,
        "food_records": food_records,
        "record_count": len(food_records)
    }

def generate_nutrition_evaluation(nutrition_summary):
    """生成营养评价"""
    total_nutrition = nutrition_summary["total_nutrition"]
    food_records = nutrition_summary["food_records"]
    record_count = nutrition_summary["record_count"]
    
    # 构建营养评价prompt
    nutrition_text = "\n".join([f"{k}: {v:.1f}" for k, v in total_nutrition.items()])
    food_list = "\n".join([f"- {record['dish_name']} ({record['datetime']})" for record in food_records[-10:]])  # 最近10条记录
    
    prompt = f"""作为专业营养师，请分析用户本周的营养摄入情况。

本周营养摄入汇总（共{record_count}次记录）：
{nutrition_text}

最近食物记录：
{food_list}

请提供营养摄入评价，分析当前营养结构是否均衡，指出哪些营养素充足、哪些不足，评估整体饮食健康状况。

注意：请用简洁明了的纯文本回答，不要使用任何Markdown格式（如**、#、-等），控制在400字以内，重点突出营养分析结果。"""
    
    try:
        # 调用Qwen模型生成评价
        evaluation = query_qwen([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
        return evaluation
    except Exception as e:
        print(f"[Nutrition] 生成营养评价失败: {e}")
        return "营养评价生成失败，请稍后重试。"

def generate_diet_recommendation(nutrition_summary):
    """生成饮食推荐"""
    total_nutrition = nutrition_summary["total_nutrition"]
    food_records = nutrition_summary["food_records"]
    record_count = nutrition_summary["record_count"]
    
    # 构建饮食推荐prompt
    nutrition_text = "\n".join([f"{k}: {v:.1f}" for k, v in total_nutrition.items()])
    food_list = "\n".join([f"- {record['dish_name']} ({record['datetime']})" for record in food_records[-10:]])  # 最近10条记录
    
    prompt = f"""作为专业营养师，请根据用户本周的营养摄入情况提供具体的饮食推荐建议。

本周营养摄入汇总（共{record_count}次记录）：
{nutrition_text}

最近食物记录：
{food_list}

请提供实用的饮食推荐建议，包括：针对营养不足或过量的情况给出具体改善建议，推荐2-3道适合的菜品并说明推荐理由，提供具体的饮食调整方案。

注意：请用简洁明了的纯文本回答，不要使用任何Markdown格式（如**、#、-等），控制在400字以内，重点突出实用性建议和具体菜品推荐。"""
    
    try:
        # 调用Qwen模型生成推荐
        recommendation = query_qwen([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
        return recommendation
    except Exception as e:
        print(f"[Nutrition] 生成饮食推荐失败: {e}")
        return "饮食推荐生成失败，请稍后重试。"

def generate_nutrition_advice(nutrition_summary):
    """生成完整的营养建议（包含评价和推荐）"""
    evaluation = generate_nutrition_evaluation(nutrition_summary)
    recommendation = generate_diet_recommendation(nutrition_summary)
    
    return {
        "evaluation": evaluation,
        "recommendation": recommendation
    }

def save_nutrition_advice(advice_data):
    """保存营养建议到文件"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    advice_file = os.path.join(NUTRITION_DIR, f"advice_{ts}.json")
    
    try:
        # 获取营养汇总数据
        nutrition_summary = get_weekly_nutrition_summary()
        
        # 处理advice_data，支持字符串和字典两种格式
        if isinstance(advice_data, str):
            # 兼容旧格式
            advice_content = {
                "evaluation": advice_data,
                "recommendation": "请重新生成获取饮食推荐"
            }
        else:
            # 新格式
            advice_content = advice_data
        
        full_advice_data = {
            "timestamp": ts,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation": advice_content.get("evaluation", ""),
            "recommendation": advice_content.get("recommendation", ""),
            "nutrition_summary": nutrition_summary
        }
        
        with open(advice_file, "w", encoding="utf-8") as f:
            json.dump(full_advice_data, f, ensure_ascii=False, indent=2)
        
        # 同时保存最新建议到固定文件名
        latest_file = os.path.join(NUTRITION_DIR, "latest_advice.json")
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(full_advice_data, f, ensure_ascii=False, indent=2)
        
        print(f"[Nutrition] 营养建议已保存: {advice_file}")
        
        # 同时生成健康预警
        generate_health_warnings(nutrition_summary)
        
        return advice_file
    except Exception as e:
        print(f"[Nutrition] 保存营养建议失败: {e}")
        return None

def generate_health_warnings(nutrition_summary):
    """生成健康预警分析"""
    try:
        # 慢性病风险评估配置
        chronic_diseases_config = {
            '糖尿病': {
                'thresholds': {
                    'low': {'carbs': [0, 150], 'fiber': [25, float('inf')]},
                    'medium': {'carbs': [151, 250], 'fiber': [15, 24]},
                    'high': {'carbs': [251, float('inf')], 'fiber': [0, 14]}
                },
                'symptoms': {
                    'low': ['血糖略有波动', '偶有疲劳感'],
                    'medium': ['血糖控制不稳', '易疲劳', '口渴增多'],
                    'high': ['血糖显著升高', '多饮多尿', '体重下降', '视力模糊']
                }
            },
            '高血压': {
                'thresholds': {
                    'low': {'sodium': [0, 2000], 'potassium': [3500, float('inf')]},
                    'medium': {'sodium': [2001, 3000], 'potassium': [2500, 3499]},
                    'high': {'sodium': [3001, float('inf')], 'potassium': [0, 2499]}
                },
                'symptoms': {
                    'low': ['血压偶有升高'],
                    'medium': ['头晕头痛', '心悸'],
                    'high': ['持续性头痛', '眩晕', '耳鸣', '心悸胸闷']
                }
            },
            '骨质疏松': {
                'thresholds': {
                    'low': {'calcium': [800, float('inf')], 'protein': [60, float('inf')]},
                    'medium': {'calcium': [600, 799], 'protein': [40, 59]},
                    'high': {'calcium': [0, 599], 'protein': [0, 39]}
                },
                'symptoms': {
                    'low': ['轻微骨密度下降'],
                    'medium': ['关节疼痛', '腰背酸痛'],
                    'high': ['骨痛明显', '容易骨折', '身高变矮', '驼背']
                }
            },
            '贫血': {
                'thresholds': {
                    'low': {'iron': [15, float('inf')], 'protein': [60, float('inf')]},
                    'medium': {'iron': [10, 14], 'protein': [40, 59]},
                    'high': {'iron': [0, 9], 'protein': [0, 39]}
                },
                'symptoms': {
                    'low': ['轻微疲劳'],
                    'medium': ['疲劳乏力', '面色苍白'],
                    'high': ['严重疲劳', '头晕心悸', '指甲苍白', '注意力不集中']
                }
            },
            '心血管疾病': {
                'thresholds': {
                    'low': {'fat': [0, 65], 'sodium': [0, 2000]},
                    'medium': {'fat': [66, 90], 'sodium': [2001, 3000]},
                    'high': {'fat': [91, float('inf')], 'sodium': [3001, float('inf')]}
                },
                'symptoms': {
                    'low': ['偶有胸闷'],
                    'medium': ['胸痛胸闷', '心悸'],
                    'high': ['胸痛加重', '呼吸困难', '心律不齐', '水肿']
                }
            }
        }
        
        # 分析慢性病风险
        warning_results = []
        total_nutrition = nutrition_summary['total_nutrition']
        
        # 转换营养数据（毫克转克，保持一致性）
        daily_nutrition = {
            'carbs': total_nutrition.get('碳水化合物(g)', 0) / 7,  # 周平均转日平均
            'fiber': total_nutrition.get('膳食纤维(g)', 0) / 7,
            'sodium': total_nutrition.get('钠(mg)', 0) / 7,
            'potassium': total_nutrition.get('钾(mg)', 0) / 7, 
            'calcium': total_nutrition.get('钙(mg)', 0) / 7,
            'protein': total_nutrition.get('蛋白质(g)', 0) / 7,
            'iron': total_nutrition.get('铁(mg)', 0) / 7,
            'fat': total_nutrition.get('脂肪(g)', 0) / 7
        }
        
        for disease_name, disease_config in chronic_diseases_config.items():
            risk_level = 'low'
            matched_criteria = []
            
            # 检查高风险阈值
            for nutrient, threshold in disease_config['thresholds']['high'].items():
                value = daily_nutrition.get(nutrient, 0)
                if threshold[0] <= value <= threshold[1]:
                    risk_level = 'high'
                    matched_criteria.append(f"{nutrient}: {value:.1f}")
                    break
            
            # 检查中风险阈值
            if risk_level == 'low':
                for nutrient, threshold in disease_config['thresholds']['medium'].items():
                    value = daily_nutrition.get(nutrient, 0)
                    if threshold[0] <= value <= threshold[1]:
                        risk_level = 'medium'
                        matched_criteria.append(f"{nutrient}: {value:.1f}")
                        break
            
            warning_results.append({
                'disease': disease_name,
                'risk_level': risk_level,
                'symptoms': disease_config['symptoms'][risk_level],
                'matched_criteria': matched_criteria
            })
        
        # 生成预警报告
        high_risk_diseases = [w for w in warning_results if w['risk_level'] == 'high']
        medium_risk_diseases = [w for w in warning_results if w['risk_level'] == 'medium']
        
        # 使用千问生成详细分析
        if high_risk_diseases or medium_risk_diseases:
            warning_prompt = f"""作为专业营养师和健康顾问，请分析以下慢性病风险评估结果：

高风险疾病：{', '.join([d['disease'] for d in high_risk_diseases]) if high_risk_diseases else '无'}
中等风险疾病：{', '.join([d['disease'] for d in medium_risk_diseases]) if medium_risk_diseases else '无'}

营养数据（日均值）：
- 碳水化合物: {daily_nutrition['carbs']:.1f}g
- 膳食纤维: {daily_nutrition['fiber']:.1f}g  
- 钠: {daily_nutrition['sodium']:.1f}mg
- 钙: {daily_nutrition['calcium']:.1f}mg
- 蛋白质: {daily_nutrition['protein']:.1f}g
- 铁: {daily_nutrition['iron']:.1f}mg
- 脂肪: {daily_nutrition['fat']:.1f}g

请提供：
1. 风险分析总结（150字以内）
2. 针对性预防建议（150字以内）

注意：使用简洁明了的文字回答，不要使用任何Markdown格式。"""

            analysis = query_qwen([{"role": "user", "content": [{"type": "text", "text": warning_prompt}]}])
        else:
            analysis = "根据营养数据分析，各项慢性病风险指标正常，请继续保持均衡饮食和健康生活方式。"
        
        # 保存预警数据
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        warning_data = {
            "timestamp": ts,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "warnings": warning_results,
            "analysis": analysis,
            "nutrition_data": daily_nutrition,
            "summary": {
                "high_risk_count": len(high_risk_diseases),
                "medium_risk_count": len(medium_risk_diseases),
                "low_risk_count": len([w for w in warning_results if w['risk_level'] == 'low'])
            }
        }
        
        # 保存到文件
        warning_file = os.path.join(NUTRITION_DIR, f"warnings_{ts}.json")
        with open(warning_file, "w", encoding="utf-8") as f:
            json.dump(warning_data, f, ensure_ascii=False, indent=2)
        
        # 更新最新预警文件
        latest_warning_file = os.path.join(NUTRITION_DIR, "latest_warnings.json")
        with open(latest_warning_file, "w", encoding="utf-8") as f:
            json.dump(warning_data, f, ensure_ascii=False, indent=2)
        
        print(f"[Warning] 健康预警已生成: {warning_file}")
        return warning_file
        
    except Exception as e:
        print(f"[Warning] 生成健康预警失败: {e}")
        return None

def get_recommended_dishes(warnings_data):
    """基于健康预警数据获取推荐菜品列表"""
    try:
        # 分析风险等级和营养需求
        high_risk_diseases = []
        medium_risk_diseases = []
        recommended_dishes = []
        
        # 解析预警数据
        if 'warnings' in warnings_data:
            warnings_list = warnings_data['warnings']
            for warning in warnings_list:
                if warning['risk_level'] == 'high':
                    high_risk_diseases.append(warning['disease'])
                elif warning['risk_level'] == 'medium':
                    medium_risk_diseases.append(warning['disease'])
        
        # 根据疾病风险选择合适的菜品
        if high_risk_diseases:
            for disease in high_risk_diseases:
                if '糖尿病' in disease:
                    # 选择低糖低GI食物
                    candidates = ['蒸蛋羹', '清蒸鲈鱼', '白灼菜心', '蒜蓉西兰花']
                    recommended_dishes.extend(random.sample(candidates, min(2, len(candidates))))
                elif '高血压' in disease:
                    # 选择低钠高钾食物
                    candidates = ['白灼菜心', '蒸南瓜', '冬瓜汤', '清蒸鲈鱼']
                    recommended_dishes.extend(random.sample(candidates, min(2, len(candidates))))
                elif '心血管疾病' in disease:
                    # 选择低脂食物
                    candidates = ['清蒸鲈鱼', '鲫鱼汤', '白灼菜心', '蒸蛋羹']
                    recommended_dishes.extend(random.sample(candidates, min(2, len(candidates))))
                elif '骨质疏松' in disease:
                    # 选择高钙食物
                    candidates = ['虾仁豆腐', '芝麻糊', '紫菜蛋花汤', '麻婆豆腐']
                    recommended_dishes.extend(random.sample(candidates, min(2, len(candidates))))
                elif '贫血' in disease:
                    # 选择富含铁质的食物
                    candidates = ['猪肝炒菠菜', '韭菜炒蛋', '黑木耳炒肉', '芝麻菠菜']
                    recommended_dishes.extend(random.sample(candidates, min(2, len(candidates))))
        
        elif medium_risk_diseases:
            # 中等风险，选择均衡营养的菜品
            candidates = ['蒸蛋羹', '鲫鱼汤', '蒜蓉西兰花', '麻婆豆腐', '白切鸡']
            recommended_dishes.extend(random.sample(candidates, min(3, len(candidates))))
        
        else:
            # 健康状况良好，随机推荐营养均衡的菜品
            all_dishes = list(DISH_NUTRITION_DATABASE.keys())
            recommended_dishes = random.sample(all_dishes, min(3, len(all_dishes)))
        
        # 去重并限制为3个菜品
        recommended_dishes = list(set(recommended_dishes))[:3]
        
        # 如果推荐菜品不足3个，补充一些健康菜品
        if len(recommended_dishes) < 3:
            healthy_dishes = ['蒸蛋羹', '清蒸鲈鱼', '蒜蓉西兰花', '麻婆豆腐', '白切鸡']
            for dish in healthy_dishes:
                if dish not in recommended_dishes and len(recommended_dishes) < 3:
                    recommended_dishes.append(dish)
        
        return recommended_dishes
        
    except Exception as e:
        print(f"[FOOD_REC] 获取推荐菜品失败: {e}")
        return ['蒸蛋羹', '清蒸鲈鱼', '蒜蓉西兰花']

def generate_food_recommendation(warnings_data):
    """基于健康预警数据生成菜品推荐"""
    try:
        # 获取推荐菜品列表
        recommended_dishes = get_recommended_dishes(warnings_data)
        
        # 分析需要补充的营养素
        nutrients_needed = []
        if 'warnings' in warnings_data:
            warnings_list = warnings_data['warnings']
            for warning in warnings_list:
                if warning['risk_level'] in ['high', 'medium']:
                    disease = warning['disease']
                    if '糖尿病' in disease:
                        nutrients_needed.append('膳食纤维')
                    elif '高血压' in disease:
                        nutrients_needed.append('钾')
                    elif '心血管疾病' in disease:
                        nutrients_needed.append('不饱和脂肪酸')
                    elif '骨质疏松' in disease:
                        nutrients_needed.append('钙')
                    elif '贫血' in disease:
                        nutrients_needed.append('铁')
        
        # 去重营养素
        nutrients_needed = list(set(nutrients_needed))
        
        # 构建简洁的推荐文本
        if len(recommended_dishes) >= 3:
            recommendation = f"根据您的营养健康状态，推荐1.{recommended_dishes[0]}，2.{recommended_dishes[1]}，3.{recommended_dishes[2]}三个菜"
        else:
            recommendation = f"根据您的营养健康状态，推荐{', '.join(recommended_dishes)}等菜品"
        
        if nutrients_needed:
            if len(nutrients_needed) == 1:
                recommendation += f"，补充{nutrients_needed[0]}营养素。"
            elif len(nutrients_needed) == 2:
                recommendation += f"，补充{nutrients_needed[0]}、{nutrients_needed[1]}两种营养素。"
            else:
                recommendation += f"，补充{', '.join(nutrients_needed[:2])}等营养素。"
        else:
            recommendation += "，保持营养均衡。"
        
        return recommendation
        
    except Exception as e:
        print(f"[FOOD_REC] 生成推荐失败: {e}")
        return "根据您的营养健康状态，推荐1.蒸蛋羹，2.清蒸鲈鱼，3.蒜蓉西兰花三个菜，补充蛋白质、维生素营养素。"

def evaluate_food_recommendation(detected_foods, warnings_data):
    """评估检测到的食物是否推荐食用"""
    try:
        if not warnings_data:
            return "暂无健康数据，建议适量食用。", "neutral"
        
        # 如果检测到两个及以上的菜，只推荐一个菜
        if len(detected_foods) >= 2:
            # 分析高风险疾病和缺乏的营养素
            high_risk_diseases = []
            lacking_nutrients = []
            
            for disease, data in warnings_data.items():
                if isinstance(data, dict) and data.get('risk_level') == '高风险':
                    high_risk_diseases.append(disease)
                    # 根据疾病类型推断缺乏的营养素
                    if disease == '糖尿病':
                        lacking_nutrients.extend(['膳食纤维', '蛋白质'])
                    elif disease == '高血压':
                        lacking_nutrients.extend(['钾', '镁'])
                    elif disease == '心血管疾病':
                        lacking_nutrients.extend(['Omega-3脂肪酸', '维生素E'])
                    elif disease == '骨质疏松':
                        lacking_nutrients.extend(['钙', '维生素D'])
                    elif disease == '贫血':
                        lacking_nutrients.extend(['铁', '维生素B12'])
            
            # 去重营养素
            lacking_nutrients = list(set(lacking_nutrients))
            
            # 评估每个食物的适宜性得分
            food_scores = []
            for food in detected_foods:
                food_name = food.get('dish_name', '')
                nutrients = food.get('nutrients', {})
                
                score = 0
                # 基础得分
                score += food.get('confidence', 0) * 100
                
                # 根据营养素含量调整得分
                protein = nutrients.get('蛋白质(g)', 0)
                calcium = nutrients.get('钙(mg)', 0)
                iron = nutrients.get('铁(mg)', 0)
                
                # 蛋白质丰富加分
                if protein > 10:
                    score += 20
                # 钙含量丰富加分
                if calcium > 100:
                    score += 15
                # 铁含量丰富加分
                if iron > 2:
                    score += 15
                
                # 不健康食物减分
                sugar = nutrients.get('碳水化合物(g)', 0)
                sodium = nutrients.get('钠(mg)', 0)
                fat = nutrients.get('脂肪(g)', 0)
                
                if sugar > 30:
                    score -= 30
                if sodium > 500:
                    score -= 25
                if fat > 20:
                    score -= 20
                
                food_scores.append((food_name, score))
            
            # 选择得分最高的食物
            best_food = max(food_scores, key=lambda x: x[1])[0]
            
            # 生成推荐文本
            if lacking_nutrients:
                nutrients_text = '、'.join(lacking_nutrients[:2])  # 最多显示两个营养素
                result_text = f"推荐您吃{best_food}，因为您缺乏{nutrients_text}"
            else:
                result_text = f"推荐您吃{best_food}，因为它营养均衡适合您"
            
            return result_text, "recommended"
        
        # 如果只有一个菜，按原逻辑处理
        else:
            # 分析高风险疾病
            high_risk_diseases = []
            for disease, data in warnings_data.items():
                if isinstance(data, dict) and data.get('risk_level') == '高风险':
                    high_risk_diseases.append(disease)
            
            # 简单的食物评估逻辑
            not_recommended = []
            recommended = []
            
            for food in detected_foods:
                food_name = food.get('dish_name', '').lower()
                nutrients = food.get('nutrients', {})
                
                # 获取关键营养素
                sugar = nutrients.get('碳水化合物(g)', 0)
                sodium = nutrients.get('钠(mg)', 0)
                fat = nutrients.get('脂肪(g)', 0)
                
                is_recommended = True
                reasons = []
                
                # 糖尿病风险评估
                if '糖尿病' in high_risk_diseases:
                    if sugar > 30 or '糖' in food_name or '甜' in food_name:
                        is_recommended = False
                        reasons.append('含糖量较高，不适合糖尿病风险人群')
                
                # 高血压风险评估
                if '高血压' in high_risk_diseases:
                    if sodium > 500 or '咸' in food_name or '腌' in food_name:
                        is_recommended = False
                        reasons.append('钠含量较高，不适合高血压风险人群')
                
                # 心血管疾病风险评估
                if '心血管疾病' in high_risk_diseases:
                    if fat > 20 or '炸' in food_name or '油' in food_name:
                        is_recommended = False
                        reasons.append('脂肪含量较高，不适合心血管疾病风险人群')
                
                if is_recommended:
                    recommended.append(food_name)
                else:
                    not_recommended.append((food_name, reasons))
            
            # 生成评估结果
            if not_recommended:
                result_text = "不建议食用以下食物：\n"
                for food, reasons in not_recommended:
                    result_text += f"? {food}: {', '.join(reasons)}\n"
                if recommended:
                    result_text += f"\n可以适量食用：{', '.join(recommended)}"
                return result_text, "not_recommended"
            elif recommended:
                return f"检测到的食物都比较适合您：{', '.join(recommended)}，建议适量食用。", "recommended"
            else:
                return "未能评估食物适宜性，建议咨询营养师。", "neutral"
            
    except Exception as e:
        print(f"[FOOD_EVAL] 评估失败: {e}")
        return "食物评估功能暂时不可用，建议适量食用。", "neutral"

async def ws_handler(ws):
    print("[Srv] ? new conn", ws.remote_address)
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    rec.SetWords(True)

    last_photo_ts = 0
    video_on = False
    last_partial = ""
    pending_text = ""
    awaiting_photo_model = False
    # 会议记录相关变量
    meeting_recording = False
    meeting_file = None
    # 视频录制会话ID
    video_session_id = None

    async for chunk in ws:
        if isinstance(chunk, bytes):
            if rec.AcceptWaveform(chunk):
                result = json.loads(rec.Result())["text"].replace(" ", "")
                if result:
                    print(f"[FINAL] {result}")
                    pending_text = result
                    
                    # 会议记录控制逻辑
                    if "会议记录开始" in result:
                        if not meeting_recording:
                            meeting_file = start_meeting_recording()
                            meeting_recording = True
                            print("[Srv] 会议记录模式已开启")
                        continue
                    elif "会议记录结束" in result or "会议识别结束" in result:
                        if meeting_recording:
                            end_meeting_recording(meeting_file)
                            meeting_recording = False
                            meeting_file = None
                            print("[Srv] 会议记录模式已关闭")
                        continue
                    
                    # 会议记录模式下，记录所有语音内容，不处理AI唤醒
                    if meeting_recording:
                        append_meeting_text(meeting_file, result)
                        continue
                    
                    # 独立拍照功能（不需要麦粒唤醒词）
                    if "拍照" in result and not detect_maili_wakeup(result):
                        if time.time() - last_photo_ts > 3:
                            await ws.send(json.dumps({"cmd": "photo"}))
                            last_photo_ts = time.time()
                            print("[Srv] ? 独立拍照 cmd")
                        awaiting_photo_model = "simple_photo"
                    
                    # 唤醒词和拍照逻辑
                    elif detect_maili_wakeup(result):
                        if "拍照" in result:
                            if time.time() - last_photo_ts > 3:
                                await ws.send(json.dumps({"cmd": "photo"}))
                                last_photo_ts = time.time()
                                print("[Srv] ? photo cmd")
                            awaiting_photo_model = True
                        elif "食物识别" in result or "实物识别" in result:
                            if time.time() - last_photo_ts > 3:
                                await ws.send(json.dumps({"cmd": "photo"}))
                                last_photo_ts = time.time()
                                print("[Srv] ? food recognition photo cmd")
                            awaiting_photo_model = "food_recommend"
                        elif "我要吃这个" in result:
                            if time.time() - last_photo_ts > 3:
                                await ws.send(json.dumps({"cmd": "photo"}))
                                last_photo_ts = time.time()
                                print("[Srv] ? food record photo cmd")
                            awaiting_photo_model = "food_record"
                        elif "菜品推荐" in result:
                            # 直接读取latest_warnings.json并推荐
                            try:
                                latest_warning_file = os.path.join(NUTRITION_DIR, "latest_warnings.json")
                                if os.path.exists(latest_warning_file):
                                    with open(latest_warning_file, "r", encoding="utf-8") as f:
                                        warnings_data = json.load(f)
                                    
                                    # 生成菜品推荐文本
                                    recommendation_text = generate_food_recommendation(warnings_data)
                                    print(f"[FOOD_REC] {recommendation_text}")
                                    
                                    # 获取推荐的三个具体菜品
                                    recommended_dishes = get_recommended_dishes(warnings_data)
                                    
                                    # 构建菜品数据结构
                                    dishes_data = []
                                    for dish_name in recommended_dishes:
                                        if dish_name in DISH_NUTRITION_DATABASE:
                                            dish_info = DISH_NUTRITION_DATABASE[dish_name]
                                            dishes_data.append({
                                                "name": dish_name,
                                                "protein": dish_info["protein"],
                                                "fat": dish_info["fat"],
                                                "carbs": dish_info["carbs"],
                                                "calcium": dish_info["calcium"],
                                                "iron": dish_info["iron"],
                                                "category": dish_info["category"]
                                            })
                                    
                                    baidu_tts.baidu_tts_test(recommendation_text)
                                    mp3 = open(TTS_FILE, "rb").read()
                                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    
                                    await ws.send(json.dumps({
                                        "tag": "response",
                                        "question": pending_text,
                                        "text": recommendation_text,
                                        "payload_hex": mp3.hex(),
                                        "filename": ts,
                                        "dishes": dishes_data
                                    }))
                                else:
                                    resp = "暂无健康预警数据，无法生成个性化菜品推荐。请先进行食物识别以建立健康档案。"
                                    baidu_tts.baidu_tts_test(resp)
                                    mp3 = open(TTS_FILE, "rb").read()
                                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    
                                    await ws.send(json.dumps({
                                        "tag": "response",
                                        "question": pending_text,
                                        "text": resp,
                                        "payload_hex": mp3.hex(),
                                        "filename": ts,
                                        "dishes": []
                                    }))
                            except Exception as e:
                                print(f"[FOOD_REC] 菜品推荐失败: {e}")
                                resp = "菜品推荐功能暂时不可用，请稍后再试。"
                                baidu_tts.baidu_tts_test(resp)
                                mp3 = open(TTS_FILE, "rb").read()
                                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                
                                await ws.send(json.dumps({
                                    "tag": "response",
                                    "question": pending_text,
                                    "text": resp,
                                    "payload_hex": mp3.hex(),
                                    "filename": ts,
                                    "dishes": []
                                }))
                            
                            awaiting_photo_model = False
                            pending_text = ""
                        else:
                            # 纯文本调用
                            print("[info] 触发 Qwen2.5-VL 文本推理 …")
                            resp = query_qwen([
                                {"role": "user", "content": [{"type": "text", "text": pending_text}]}])
                            print(f"[QWEN] 文本回复：{resp}")
                            # 生成 TTS 并保存历史 TXT
                            baidu_tts.baidu_tts_test(resp)
                            mp3 = open(TTS_FILE, "rb").read()
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            os.makedirs("history", exist_ok=True)
                            with open(os.path.join("history", f"response_{ts}.txt"), "w", encoding="utf-8") as f:
                                f.write(f"Q: {pending_text}\nA: {resp}\n")
                            # 发送给 Pi
                            await ws.send(json.dumps({
                                "tag": "response",
                                "question": pending_text,
                                "text": resp,
                                "payload_hex": mp3.hex(),
                                "filename": ts
                            }))
                            awaiting_photo_model = False
                            pending_text = ""
                    # 录像控制
                    if "录像暂停" in result or "停止录像" in result:
                        if video_on:
                            await ws.send(json.dumps({"cmd": "video_stop"}))
                            video_on = False
                            # 生成视频文件
                            if video_session_id:
                                video_path = create_video_from_frames(video_session_id)
                                if video_path:
                                    print(f"[Srv] 视频已保存: {video_path}")
                                video_session_id = None
                            print("[Srv] ?? video_stop")
                    elif "录像" in result and not video_on:
                        video_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                        await ws.send(json.dumps({"cmd": "video_start"}))
                        video_on = True
                        print(f"[Srv] ?? video_start, session: {video_session_id}")
                last_partial = ""
            else:
                # 保留 PART 输出
                partial = json.loads(rec.PartialResult())["partial"].replace(" ", "")
                if partial and partial != last_partial:
                    print(f"[PART]  {partial}", end="\r", flush=True)
                    last_partial = partial
        else:
            data = json.loads(chunk)
            tag = data.get("tag")
            if tag == "periodic_detection":
                # 处理定时检测照片
                img_bytes = bytes.fromhex(data["payload_hex"])
                
                # 保存原始照片到临时位置
                temp_photo_path = os.path.join("/tmp", f"periodic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                with open(temp_photo_path, "wb") as f:
                    f.write(img_bytes)
                
                # 读取图像进行多目标检测
                image = cv2.imread(temp_photo_path)
                if image is not None:
                    try:
                        # 运行多目标检测
                        results = multi_target_detector.process_frame(image)
                        
                        # 可视化结果
                        annotated_image = multi_target_detector.visualize_results(image, results)
                        
                        # 保存结果图片到主目录（覆盖上一次的结果）
                        result_path = "/root/autodl-tmp/.autodl/iot/periodic_detection_result.jpg"
                        cv2.imwrite(result_path, annotated_image)
                        print(f"[定时检测] 已完成并保存结果")
                        
                        # 清理临时文件
                        os.remove(temp_photo_path)
                        
                    except Exception as e:
                        print(f"[定时检测] 处理失败: {e}")
                        # 清理临时文件
                        if os.path.exists(temp_photo_path):
                            os.remove(temp_photo_path)
                else:
                    print("[定时检测] 图像读取失败")
                    # 清理临时文件
                    if os.path.exists(temp_photo_path):
                        os.remove(temp_photo_path)
                        
            elif tag == "photo":
                img_bytes = bytes.fromhex(data["payload_hex"])
                photo_path = save_photo(img_bytes)
                
                # 独立拍照功能（仅保存照片，不调用AI）
                if awaiting_photo_model == "simple_photo":
                    print(f"[Srv] 独立拍照已保存: {photo_path}")
                    awaiting_photo_model = False
                    pending_text = ""
                elif awaiting_photo_model == "food_recommend" and detect_maili_wakeup(pending_text) and ("食物识别" in pending_text or "实物识别" in pending_text):
                     # 多目标食品识别调用
                     print("[info] 触发多目标食品识别 …")
                     
                     # 读取图像
                     image = cv2.imread(photo_path)
                     if image is None:
                         resp = "抱歉，无法读取图像，请重试"
                         print(f"[FOOD] {resp}")
                         baidu_tts.baidu_tts_test(resp)
                         mp3 = open(TTS_FILE, "rb").read()
                         await ws.send(json.dumps({
                             "tag": "response",
                             "question": pending_text,
                             "text": resp,
                             "payload_hex": mp3.hex(),
                             "filename": datetime.now().strftime("%Y%m%d_%H%M%S")
                         }))
                         awaiting_photo_model = False
                         pending_text = ""
                         continue
                     
                     # 进行多目标检测
                     detection_results = multi_target_detector.process_frame(image)
                     
                     if detection_results['total_detected'] > 0:
                         # 生成可视化结果
                         annotated_image = multi_target_detector.visualize_results(image, detection_results)
                         
                         # 保存可视化结果
                         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                         result_image_path = os.path.join(PHOTO_DIR, f"multi_detection_{ts}.jpg")
                         cv2.imwrite(result_image_path, annotated_image)
                         
                         # 获取所有检测到的食物信息
                         all_foods = [{
                             'dish_name': food['chinese_food_result']['dish_name'],
                             'confidence': food['chinese_food_result']['confidence'],
                             'nutrients': food['chinese_food_result']['nutrients']
                         } for food in detection_results['food_detections']]
                         
                         # 读取健康预警数据进行推荐评估
                         try:
                             latest_warning_file = os.path.join(NUTRITION_DIR, "latest_warnings.json")
                             warnings_data = {}
                             if os.path.exists(latest_warning_file):
                                 with open(latest_warning_file, "r", encoding="utf-8") as f:
                                     warnings_data = json.load(f)
                             
                             # 评估食物推荐
                             recommendation_text, recommendation_status = evaluate_food_recommendation(all_foods, warnings_data)
                             
                             # 构建检测结果描述
                             food_names = [food['dish_name'] for food in all_foods]
                             detection_desc = f"检测到{len(all_foods)}种食物：{', '.join(food_names)}。"
                             
                             full_resp = f"{detection_desc}\n\n{recommendation_text}"
                             print(f"[FOOD_REC] {full_resp}")
                             
                         except Exception as e:
                             print(f"[FOOD_REC] 推荐评估失败: {e}")
                             food_names = [food['dish_name'] for food in all_foods]
                             full_resp = f"检测到{len(all_foods)}种食物：{', '.join(food_names)}。建议适量食用。"
                         
                         baidu_tts.baidu_tts_test(full_resp)
                         mp3 = open(TTS_FILE, "rb").read()
                         
                         await ws.send(json.dumps({
                             "tag": "response",
                             "question": pending_text,
                             "text": full_resp,
                             "payload_hex": mp3.hex(),
                             "filename": ts,
                             "food_result": {
                                 "foods": all_foods,
                                 "detection_type": "food_recommend",
                                 "total_detected": detection_results['total_detected']
                             }
                         }))
                     else:
                         resp = "抱歉，未检测到任何食物目标，请重试"
                         print(f"[FOOD] {resp}")
                         baidu_tts.baidu_tts_test(resp)
                         mp3 = open(TTS_FILE, "rb").read()
                         await ws.send(json.dumps({
                             "tag": "response",
                             "question": pending_text,
                             "text": resp,
                             "payload_hex": mp3.hex(),
                             "filename": datetime.now().strftime("%Y%m%d_%H%M%S")
                         }))
                     
                     awaiting_photo_model = False
                     pending_text = ""
                elif awaiting_photo_model == "food_record":
                     # 食物记录功能
                     print("[info] 触发食物记录功能 …")
                     
                     # 读取图像
                     image = cv2.imread(photo_path)
                     if image is None:
                         resp = "抱歉，无法读取图像，请重试"
                         print(f"[FOOD_RECORD] {resp}")
                         baidu_tts.baidu_tts_test(resp)
                         mp3 = open(TTS_FILE, "rb").read()
                         await ws.send(json.dumps({
                             "tag": "response",
                             "question": pending_text,
                             "text": resp,
                             "payload_hex": mp3.hex(),
                             "filename": datetime.now().strftime("%Y%m%d_%H%M%S")
                         }))
                         awaiting_photo_model = False
                         pending_text = ""
                         continue
                     
                     # 进行多目标检测
                     detection_results = multi_target_detector.process_frame(image)
                     
                     if detection_results['total_detected'] > 0:
                         # 生成可视化结果
                         annotated_image = multi_target_detector.visualize_results(image, detection_results)
                         
                         # 保存可视化结果
                         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                         result_image_path = os.path.join(PHOTO_DIR, f"food_record_{ts}.jpg")
                         cv2.imwrite(result_image_path, annotated_image)
                         
                         # 选择要记录的食物
                         selected_food = detection_results.get('selected_food')
                         if selected_food:
                             # 手势选中了食物
                             dish_name = selected_food['chinese_food_result']['dish_name']
                             confidence = selected_food['chinese_food_result']['confidence']
                             nutrients = selected_food['chinese_food_result']['nutrients']
                             
                             resp = f"已将{dish_name}菜品记录，置信度{confidence:.2f}"
                         else:
                             # 没有手势选择，选择屏幕上最大的菜品（置信度最高）
                             best_food = max(detection_results['food_detections'], 
                                            key=lambda x: x['chinese_food_result']['confidence'])
                             dish_name = best_food['chinese_food_result']['dish_name']
                             confidence = best_food['chinese_food_result']['confidence']
                             nutrients = best_food['chinese_food_result']['nutrients']
                             
                             if detection_results['gesture']:
                                 resp = f"手势未指向明确菜品，已记录屏幕上最大的菜品：{dish_name}，置信度{confidence:.2f}"
                             else:
                                 resp = f"未检测到手势，已记录屏幕上最大的菜品：{dish_name}，置信度{confidence:.2f}"
                         
                         # 保存食物记录
                         food_result_path = os.path.join(FOOD_DIR, f"food_{ts}.json")
                         with open(food_result_path, "w", encoding="utf-8") as f:
                             json.dump({
                                 "timestamp": ts,
                                 "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                 "photo_path": photo_path,
                                 "result_image_path": result_image_path,
                                 "dish_name": dish_name,
                                 "confidence": confidence,
                                 "nutrients": nutrients,
                                 "detection_type": "food_record",
                                 "total_detected": detection_results['total_detected'],
                                 "has_gesture": detection_results['gesture'] is not None,
                                 "gesture_selected": selected_food is not None,
                                 "full_result": {
                                     "food_detections": [{
                                         "dish_name": food['chinese_food_result']['dish_name'],
                                         "confidence": food['chinese_food_result']['confidence'],
                                         "bbox": food['bbox'],
                                         "yolo_class": food['yolo_class']
                                     } for food in detection_results['food_detections']]
                                 }
                             }, f, ensure_ascii=False, indent=2)
                         
                         # 添加营养信息
                         key_nutrients = ["热量(kcal)", "蛋白质(g)", "脂肪(g)", "碳水化合物(g)"]
                         nutrient_summary = ", ".join([f"{k}:{nutrients.get(k, 0):.1f}" for k in key_nutrients if k in nutrients])
                         
                         full_resp = f"{resp}。营养成分：{nutrient_summary}。记录已保存。"
                         print(f"[FOOD_RECORD] {full_resp}")
                         
                         baidu_tts.baidu_tts_test(full_resp)
                         mp3 = open(TTS_FILE, "rb").read()
                         
                         await ws.send(json.dumps({
                             "tag": "response",
                             "question": pending_text,
                             "text": full_resp,
                             "payload_hex": mp3.hex(),
                             "filename": ts,
                             "food_result": {
                                 "dish_name": dish_name,
                                 "confidence": confidence,
                                 "nutrients": nutrients,
                                 "detection_type": "food_record",
                                 "total_detected": detection_results['total_detected']
                             }
                         }))
                         
                         # 生成营养建议
                         try:
                             print("[Nutrition] 正在生成营养建议...")
                             nutrition_summary = get_weekly_nutrition_summary()
                             advice = generate_nutrition_advice(nutrition_summary)
                             save_nutrition_advice(advice)
                             print("[Nutrition] 营养建议已生成")
                         except Exception as e:
                             print(f"[Nutrition] 营养建议生成失败: {e}")
                     else:
                         resp = "抱歉，未检测到任何食物目标，请重试"
                         print(f"[FOOD_RECORD] {resp}")
                         baidu_tts.baidu_tts_test(resp)
                         mp3 = open(TTS_FILE, "rb").read()
                         await ws.send(json.dumps({
                             "tag": "response",
                             "question": pending_text,
                             "text": resp,
                             "payload_hex": mp3.hex(),
                             "filename": datetime.now().strftime("%Y%m%d_%H%M%S")
                         }))
                     
                     awaiting_photo_model = False
                     pending_text = ""
                elif awaiting_photo_model == True and detect_maili_wakeup(pending_text) and "拍照" in pending_text:
                    # 图文调用
                    print("[info] 触发 Qwen2.5-VL 图文推理 …")
                    resp = query_qwen([{
                        "role": "user",
                        "content":[
                            {"type":"image","image":f"file://{photo_path}"},
                            {"type":"text","text":pending_text}
                        ]
                    }])
                    print(f"[QWEN] 图文回复：{resp}")
                    baidu_tts.baidu_tts_test(resp)
                    mp3 = open(TTS_FILE, "rb").read()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs("history", exist_ok=True)
                    with open(os.path.join("history", f"response_{ts}.txt"), "w", encoding="utf-8") as f:
                                f.write(f"Q: {pending_text}\nA: {resp}\nPhoto: {photo_path}\n")
                    await ws.send(json.dumps({
                        "tag": "response",
                        "question": pending_text,
                        "text": resp,
                        "payload_hex": mp3.hex(),
                        "filename": ts
                    }))
                    awaiting_photo_model = False
                    pending_text = ""
            elif tag == "video_frame":
                if video_session_id:
                    # 为视频帧添加会话ID前缀
                    img_bytes = bytes.fromhex(data["payload_hex"])
                    fn = f"{video_session_id}_{datetime.now().strftime('%H%M%S_%f')}.jpg"
                    path = os.path.join(VIDEO_DIR, fn)
                    with open(path, "wb") as f:
                        f.write(img_bytes)
                else:
                    save_frame(bytes.fromhex(data["payload_hex"]))
            else:
                print("[Srv] txt:", data)


# HTTP服务器处理函数
async def handle_photos(request):
    """处理照片目录请求"""
    try:
        files = os.listdir(PHOTO_DIR)
        photo_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))]
        
        # 生成简单的HTML目录列表
        html = "<html><body><h1>Photos Directory</h1><ul>"
        for file in sorted(photo_files, reverse=True):
            html += f'<li><a href="{file}">{file}</a></li>'
        html += "</ul></body></html>"
        
        return web.Response(text=html, content_type='text/html')
    except Exception as e:
        return web.Response(text=f"Error: {str(e)}", status=500)

async def handle_videos(request):
    """处理视频目录请求"""
    try:
        files = os.listdir(VIDEOS_DIR)
        video_files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv'))]
        
        # 生成简单的HTML目录列表
        html = "<html><body><h1>Videos Directory</h1><ul>"
        for file in sorted(video_files, reverse=True):
            html += f'<li><a href="{file}">{file}</a></li>'
        html += "</ul></body></html>"
        
        return web.Response(text=html, content_type='text/html')
    except Exception as e:
        return web.Response(text=f"Error: {str(e)}", status=500)

async def handle_photo_file(request):
    """处理单个照片文件请求"""
    filename = request.match_info['filename']
    file_path = os.path.join(PHOTO_DIR, filename)
    
    if not os.path.exists(file_path):
        return web.Response(text="File not found", status=404)
    
    try:
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
        
        # 根据文件扩展名设置Content-Type
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            content_type = 'image/jpeg'
        elif filename.lower().endswith('.png'):
            content_type = 'image/png'
        elif filename.lower().endswith('.gif'):
            content_type = 'image/gif'
        elif filename.lower().endswith('.webp'):
            content_type = 'image/webp'
        else:
            content_type = 'application/octet-stream'
        
        return web.Response(body=content, content_type=content_type)
    except Exception as e:
        return web.Response(text=f"Error reading file: {str(e)}", status=500)

async def handle_video_file(request):
    """处理单个视频文件请求"""
    filename = request.match_info['filename']
    file_path = os.path.join(VIDEOS_DIR, filename)
    
    if not os.path.exists(file_path):
        return web.Response(text="File not found", status=404)
    
    try:
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
        
        # 根据文件扩展名设置Content-Type
        if filename.lower().endswith('.mp4'):
            content_type = 'video/mp4'
        elif filename.lower().endswith('.avi'):
            content_type = 'video/x-msvideo'
        elif filename.lower().endswith('.mov'):
            content_type = 'video/quicktime'
        elif filename.lower().endswith('.webm'):
            content_type = 'video/webm'
        elif filename.lower().endswith('.mkv'):
            content_type = 'video/x-matroska'
        else:
            content_type = 'application/octet-stream'
        
        return web.Response(body=content, content_type=content_type)
    except Exception as e:
        return web.Response(text=f"Error reading file: {str(e)}", status=500)

async def handle_nutrition_advice(request):
    """处理营养建议请求"""
    try:
        latest_file = os.path.join(NUTRITION_DIR, "latest_advice.json")
        if os.path.exists(latest_file):
            async with aiofiles.open(latest_file, 'r', encoding='utf-8') as f:
                content = await f.read()
            return web.Response(text=content, content_type='application/json; charset=utf-8')
        else:
            # 如果没有最新建议，返回默认消息
            default_advice = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation": "暂无营养评价，请先进行食物识别以生成个性化建议。",
                "recommendation": "暂无饮食推荐，请先进行食物识别以生成个性化建议。",
                "nutrition_summary": {
                    "week_start": datetime.now().strftime("%Y-%m-%d"),
                    "total_nutrition": {},
                    "food_records": [],
                    "record_count": 0
                }
            }
            return web.Response(text=json.dumps(default_advice, ensure_ascii=False), content_type='application/json; charset=utf-8')
    except Exception as e:
        return web.Response(text=f"Error: {str(e)}", status=500)


async def create_http_app():
    """创建HTTP应用"""
    app = web.Application()
    
    # 添加路由
    app.router.add_get('/photos/', handle_photos)
    app.router.add_get('/videos/', handle_videos)
    app.router.add_get('/photos/{filename}', handle_photo_file)
    app.router.add_get('/videos/{filename}', handle_video_file)
    app.router.add_get('/api/nutrition-advice', handle_nutrition_advice)

    
    # 静态文件服务
    app.router.add_static('/web/', path='web', name='web')
    app.router.add_static('/food_recognition/', path=FOOD_DIR, name='food_recognition')
    app.router.add_static('/meeting_recording/', path=MEETING_DIR, name='meeting_recording')
    app.router.add_static('/history/', path='history', name='history')
    app.router.add_static('/nutrition_analysis/', path=NUTRITION_DIR, name='nutrition_analysis')
    
    return app

async def main():
    # 启动HTTP服务器
    http_app = await create_http_app()
    http_runner = web_runner.AppRunner(http_app)
    await http_runner.setup()
    http_site = web_runner.TCPSite(http_runner, '0.0.0.0', 8080)
    await http_site.start()
    print("[Srv] HTTP server started on http://0.0.0.0:8080")
    
    # 启动WebSocket服务器
    async with websockets.serve(ws_handler, "0.0.0.0", 6006, ping_interval=20, max_size=None):
        print("[Srv] ? ws://0.0.0.0:6006/ws (vosk + Qwen + TTS ready)")
        print("[Srv] 媒体文件访问: http://0.0.0.0:8080/photos/ 和 http://0.0.0.0:8080/videos/")
        await asyncio.Future()

if __name__ == "__main__": asyncio.run(main())