# -*- coding: gbk -*-
# server_agent.py  �� ʵʱ PART+FINAL ��� + Qwen2.5-VL ���� + TTS
#!/usr/bin/env python3
#python /root/autodl-tmp/.autodl/iot/server_agent.py
import asyncio, websockets, json, os, time
from datetime import datetime
from vosk import Model, KaldiRecognizer
from aiohttp import web, web_runner
import aiofiles
import cv2

# --- Qwen2.5-VL ��� ---
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# --- Baidu TTS ---
import baidu_tts

# --- ʳƷʶ�� ---
import sys
sys.path.append('/root/autodl-tmp/.autodl/iot/food_cv')
from food_cv.GatedRegNet_Food import FoodRecognizer

# --- ��Ŀ��ʳ���� ---
from multi_target_food_detector import MultiTargetFoodDetector

# --- ��Ƶ���� ---
import glob
import random

# ��ƷӪ�����ݿ�
DISH_NUTRITION_DATABASE = {
    # �ߵ����ʲ�Ʒ
    '������': { 'protein': 16.2, 'fat': 20.5, 'carbs': 8.1, 'fiber': 0.8, 'calcium': 15, 'iron': 3.2, 'category': 'meat' },
    '���м�': { 'protein': 23.3, 'fat': 9.3, 'carbs': 0.0, 'fiber': 0.0, 'calcium': 12, 'iron': 1.4, 'category': 'meat' },
    '������': { 'protein': 13.1, 'fat': 11.2, 'carbs': 1.2, 'fiber': 0.0, 'calcium': 56, 'iron': 2.8, 'category': 'egg' },
    '������': { 'protein': 17.1, 'fat': 2.7, 'carbs': 0.0, 'fiber': 0.0, 'calcium': 79, 'iron': 1.3, 'category': 'fish' },
    'ţ��������': { 'protein': 18.6, 'fat': 12.4, 'carbs': 15.2, 'fiber': 2.1, 'calcium': 18, 'iron': 3.8, 'category': 'meat' },
    
    # �߸Ʋ�Ʒ
    '���Ŷ���': { 'protein': 7.9, 'fat': 11.1, 'carbs': 6.0, 'fiber': 1.1, 'calcium': 68.6, 'iron': 2.2, 'category': 'tofu' },
    'Ϻ�ʶ���': { 'protein': 15.8, 'fat': 8.3, 'carbs': 4.2, 'fiber': 0.5, 'calcium': 156, 'iron': 2.1, 'category': 'seafood' },
    '֥���': { 'protein': 4.8, 'fat': 12.6, 'carbs': 22.1, 'fiber': 3.2, 'calcium': 218, 'iron': 4.6, 'category': 'dessert' },
    '�ϲ˵�����': { 'protein': 8.2, 'fat': 4.1, 'carbs': 2.8, 'fiber': 1.8, 'calcium': 89, 'iron': 12.8, 'category': 'soup' },
    '����ˮ��': { 'protein': 5.2, 'fat': 3.8, 'carbs': 18.4, 'fiber': 2.1, 'calcium': 125, 'iron': 0.3, 'category': 'dairy' },
    
    # ������Ʒ
    '��γ�����': { 'protein': 18.2, 'fat': 8.9, 'carbs': 6.4, 'fiber': 2.8, 'calcium': 28, 'iron': 18.6, 'category': 'organ' },
    '�²˳���': { 'protein': 11.8, 'fat': 12.3, 'carbs': 4.1, 'fiber': 2.4, 'calcium': 42, 'iron': 6.2, 'category': 'vegetable' },
    '��ľ������': { 'protein': 13.6, 'fat': 9.8, 'carbs': 8.2, 'fiber': 4.2, 'calcium': 38, 'iron': 8.9, 'category': 'fungus' },
    '����������': { 'protein': 2.8, 'fat': 0.4, 'carbs': 28.6, 'fiber': 5.1, 'calcium': 45, 'iron': 4.8, 'category': 'dessert' },
    '֥�鲤��': { 'protein': 6.2, 'fat': 8.4, 'carbs': 9.1, 'fiber': 3.8, 'calcium': 86, 'iron': 7.2, 'category': 'vegetable' },
    
    # ����ά��Ʒ
    '����������': { 'protein': 4.3, 'fat': 2.1, 'carbs': 6.6, 'fiber': 4.2, 'calcium': 47, 'iron': 1.2, 'category': 'vegetable' },
    '�����۲�': { 'protein': 2.2, 'fat': 0.8, 'carbs': 4.8, 'fiber': 3.2, 'calcium': 80, 'iron': 2.5, 'category': 'vegetable' },
    'ľ����': { 'protein': 2.8, 'fat': 0.5, 'carbs': 5.4, 'fiber': 2.8, 'calcium': 166, 'iron': 3.2, 'category': 'vegetable' },
    '������': { 'protein': 1.2, 'fat': 0.2, 'carbs': 2.9, 'fiber': 1.8, 'calcium': 19, 'iron': 0.3, 'category': 'soup' },
    '�ܲ�˿��': { 'protein': 1.6, 'fat': 0.3, 'carbs': 4.1, 'fiber': 2.4, 'calcium': 24, 'iron': 0.8, 'category': 'soup' },
    
    # ��֬���ǲ�Ʒ
    '��������': { 'protein': 18.6, 'fat': 3.4, 'carbs': 0.0, 'fiber': 0.0, 'calcium': 138, 'iron': 1.2, 'category': 'fish' },
    '���Ʋ���': { 'protein': 2.8, 'fat': 0.4, 'carbs': 3.2, 'fiber': 1.8, 'calcium': 108, 'iron': 1.8, 'category': 'vegetable' },
    '���Ϲ�': { 'protein': 1.2, 'fat': 0.1, 'carbs': 5.3, 'fiber': 1.4, 'calcium': 16, 'iron': 0.4, 'category': 'vegetable' },
    '�ƹ���': { 'protein': 0.8, 'fat': 0.2, 'carbs': 2.0, 'fiber': 0.8, 'calcium': 15, 'iron': 0.3, 'category': 'soup' }
}

# ����ģ��·��
QWEN_MODEL_PATH = "/root/autodl-tmp/.autodl/iot/qwen_7b_vl_offline"
# TTS ��ʱ�ļ�
TTS_FILE = "result.mp3"

print("[Srv] Loading Qwen2.5-VL model ��")
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

# ��ʼ��ʳƷʶ����
print("[Srv] Loading FoodRecognizer...")
food_recognizer = FoodRecognizer()
print("[Srv] FoodRecognizer ready")

# ��ʼ����Ŀ��ʳ������
print("[Srv] Loading MultiTargetFoodDetector...")
multi_target_detector = MultiTargetFoodDetector(silent_mode=True)
print("[Srv] MultiTargetFoodDetector ready")

# �������Ѵ�ģ����ʶ��
def detect_maili_wakeup(text):
    """����������Ѵʼ���ģ����"""
    wakeup_words = ["����", "����", "����", "����", "����", "����", "����", "����","���"]
    for word in wakeup_words:
        if word in text:
            return True
    return False


def query_qwen(messages):
    """���� Qwen��messages ǰע��ϵͳ Prompt"""
    system_prompt = { "role": "system", "content": "��С���������Ƕ�����ҵ������۾��У����ҵ�˽���������֡�ͬʱ�����û���˽��Ӫ��ʦ����ͨ����Ӫ��ѧ֪ʶ�����Ŀ���ǰ����û���������ʳ���������ȡ��û������ֺܿ��ܰ�������֣��������ⲻͨ˳�밴�շ���������⡣" }
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

# --- Vosk ʶ����� ---
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
    """����Ƶ֡�ϳ�Ϊ��Ƶ�ļ�"""
    frame_pattern = os.path.join(VIDEO_DIR, f"{video_session_id}_*.jpg")
    frame_files = sorted(glob.glob(frame_pattern))
    
    if len(frame_files) < 2:
        print(f"[Srv] ��Ƶ֡��������: {len(frame_files)}")
        return None
    
    # ��ȡ��һ֡��ȡ�ߴ�
    first_frame = cv2.imread(frame_files[0])
    height, width, layers = first_frame.shape
    
    # ������Ƶ�ļ�
    video_filename = f"{video_session_id}.mp4"
    video_path = os.path.join(VIDEOS_DIR, video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
    
    # д������֡
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is not None:
            video_writer.write(frame)
    
    video_writer.release()
    
    # ������ʱ֡�ļ�
    for frame_file in frame_files:
        try:
            os.remove(frame_file)
        except:
            pass
    
    print(f"[Srv] ��Ƶ������: {video_path}")
    return video_path

def start_meeting_recording():
    """��ʼ�����¼"""
    fn = datetime.now().strftime("%Y%m%d_%H%M%S_meeting.txt")
    path = os.path.join(MEETING_DIR, fn)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"�����¼��ʼʱ��: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")
    print(f"[Srv] �����¼��ʼ: {fn}")
    return path

def append_meeting_text(meeting_file: str, text: str):
    """׷�ӻ����¼����"""
    if meeting_file and os.path.exists(meeting_file):
        with open(meeting_file, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] {text}\n")
        print(f"[Srv] �����¼׷��: {text}")

def end_meeting_recording(meeting_file: str):
    """���������¼"""
    if meeting_file and os.path.exists(meeting_file):
        with open(meeting_file, "a", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write(f"�����¼����ʱ��: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"[Srv] �����¼����: {meeting_file}")

def get_weekly_nutrition_summary():
    """��ȡ����Ӫ���������"""
    from datetime import datetime, timedelta
    import glob
    
    # ���㱾�ܵĿ�ʼ�ͽ���ʱ��
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # ��ȡ���ܵ�ʳƷʶ���¼
    food_files = glob.glob(os.path.join(FOOD_DIR, "food_*.json"))
    weekly_nutrition = {
        "����(kcal)": 0, "������(g)": 0, "֬��(g)": 0, "̼ˮ������(g)": 0,
        "��ʳ��ά(g)": 0, "ά����A(��g)": 0, "ά����C(mg)": 0, "��(mg)": 0,
        "��(mg)": 0, "п(mg)": 0
    }
    food_records = []
    
    for food_file in food_files:
        try:
            with open(food_file, "r", encoding="utf-8") as f:
                food_data = json.load(f)
            
            # ����ʱ���
            food_time = datetime.strptime(food_data["timestamp"], "%Y%m%d_%H%M%S")
            
            # ����Ƿ��ڱ�����
            if food_time >= week_start:
                nutrients = food_data.get("nutrients", {})
                food_records.append({
                    "dish_name": food_data.get("dish_name", "δ֪ʳƷ"),
                    "datetime": food_data.get("datetime", ""),
                    "nutrients": nutrients
                })
                
                # �ۼ�Ӫ����
                for nutrient, value in nutrients.items():
                    if nutrient in weekly_nutrition:
                        weekly_nutrition[nutrient] += float(value)
        except Exception as e:
            print(f"[Nutrition] ����ʳƷ��¼ʧ��: {food_file}, {e}")
    
    return {
        "week_start": week_start.strftime("%Y-%m-%d"),
        "total_nutrition": weekly_nutrition,
        "food_records": food_records,
        "record_count": len(food_records)
    }

def generate_nutrition_evaluation(nutrition_summary):
    """����Ӫ������"""
    total_nutrition = nutrition_summary["total_nutrition"]
    food_records = nutrition_summary["food_records"]
    record_count = nutrition_summary["record_count"]
    
    # ����Ӫ������prompt
    nutrition_text = "\n".join([f"{k}: {v:.1f}" for k, v in total_nutrition.items()])
    food_list = "\n".join([f"- {record['dish_name']} ({record['datetime']})" for record in food_records[-10:]])  # ���10����¼
    
    prompt = f"""��ΪרҵӪ��ʦ��������û����ܵ�Ӫ�����������

����Ӫ��������ܣ���{record_count}�μ�¼����
{nutrition_text}

���ʳ���¼��
{food_list}

���ṩӪ���������ۣ�������ǰӪ���ṹ�Ƿ���⣬ָ����ЩӪ���س��㡢��Щ���㣬����������ʳ����״����

ע�⣺���ü�����˵Ĵ��ı��ش𣬲�Ҫʹ���κ�Markdown��ʽ����**��#��-�ȣ���������400�����ڣ��ص�ͻ��Ӫ�����������"""
    
    try:
        # ����Qwenģ����������
        evaluation = query_qwen([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
        return evaluation
    except Exception as e:
        print(f"[Nutrition] ����Ӫ������ʧ��: {e}")
        return "Ӫ����������ʧ�ܣ����Ժ����ԡ�"

def generate_diet_recommendation(nutrition_summary):
    """������ʳ�Ƽ�"""
    total_nutrition = nutrition_summary["total_nutrition"]
    food_records = nutrition_summary["food_records"]
    record_count = nutrition_summary["record_count"]
    
    # ������ʳ�Ƽ�prompt
    nutrition_text = "\n".join([f"{k}: {v:.1f}" for k, v in total_nutrition.items()])
    food_list = "\n".join([f"- {record['dish_name']} ({record['datetime']})" for record in food_records[-10:]])  # ���10����¼
    
    prompt = f"""��ΪרҵӪ��ʦ��������û����ܵ�Ӫ����������ṩ�������ʳ�Ƽ����顣

����Ӫ��������ܣ���{record_count}�μ�¼����
{nutrition_text}

���ʳ���¼��
{food_list}

���ṩʵ�õ���ʳ�Ƽ����飬���������Ӫ�������������������������ƽ��飬�Ƽ�2-3���ʺϵĲ�Ʒ��˵���Ƽ����ɣ��ṩ�������ʳ����������

ע�⣺���ü�����˵Ĵ��ı��ش𣬲�Ҫʹ���κ�Markdown��ʽ����**��#��-�ȣ���������400�����ڣ��ص�ͻ��ʵ���Խ���;����Ʒ�Ƽ���"""
    
    try:
        # ����Qwenģ�������Ƽ�
        recommendation = query_qwen([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
        return recommendation
    except Exception as e:
        print(f"[Nutrition] ������ʳ�Ƽ�ʧ��: {e}")
        return "��ʳ�Ƽ�����ʧ�ܣ����Ժ����ԡ�"

def generate_nutrition_advice(nutrition_summary):
    """����������Ӫ�����飨�������ۺ��Ƽ���"""
    evaluation = generate_nutrition_evaluation(nutrition_summary)
    recommendation = generate_diet_recommendation(nutrition_summary)
    
    return {
        "evaluation": evaluation,
        "recommendation": recommendation
    }

def save_nutrition_advice(advice_data):
    """����Ӫ�����鵽�ļ�"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    advice_file = os.path.join(NUTRITION_DIR, f"advice_{ts}.json")
    
    try:
        # ��ȡӪ����������
        nutrition_summary = get_weekly_nutrition_summary()
        
        # ����advice_data��֧���ַ������ֵ����ָ�ʽ
        if isinstance(advice_data, str):
            # ���ݾɸ�ʽ
            advice_content = {
                "evaluation": advice_data,
                "recommendation": "���������ɻ�ȡ��ʳ�Ƽ�"
            }
        else:
            # �¸�ʽ
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
        
        # ͬʱ�������½��鵽�̶��ļ���
        latest_file = os.path.join(NUTRITION_DIR, "latest_advice.json")
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(full_advice_data, f, ensure_ascii=False, indent=2)
        
        print(f"[Nutrition] Ӫ�������ѱ���: {advice_file}")
        
        # ͬʱ���ɽ���Ԥ��
        generate_health_warnings(nutrition_summary)
        
        return advice_file
    except Exception as e:
        print(f"[Nutrition] ����Ӫ������ʧ��: {e}")
        return None

def generate_health_warnings(nutrition_summary):
    """���ɽ���Ԥ������"""
    try:
        # ���Բ�������������
        chronic_diseases_config = {
            '����': {
                'thresholds': {
                    'low': {'carbs': [0, 150], 'fiber': [25, float('inf')]},
                    'medium': {'carbs': [151, 250], 'fiber': [15, 24]},
                    'high': {'carbs': [251, float('inf')], 'fiber': [0, 14]}
                },
                'symptoms': {
                    'low': ['Ѫ�����в���', 'ż��ƣ�͸�'],
                    'medium': ['Ѫ�ǿ��Ʋ���', '��ƣ��', '�ڿ�����'],
                    'high': ['Ѫ����������', '��������', '�����½�', '����ģ��']
                }
            },
            '��Ѫѹ': {
                'thresholds': {
                    'low': {'sodium': [0, 2000], 'potassium': [3500, float('inf')]},
                    'medium': {'sodium': [2001, 3000], 'potassium': [2500, 3499]},
                    'high': {'sodium': [3001, float('inf')], 'potassium': [0, 2499]}
                },
                'symptoms': {
                    'low': ['Ѫѹż������'],
                    'medium': ['ͷ��ͷʹ', '�ļ�'],
                    'high': ['������ͷʹ', 'ѣ��', '����', '�ļ�����']
                }
            },
            '��������': {
                'thresholds': {
                    'low': {'calcium': [800, float('inf')], 'protein': [60, float('inf')]},
                    'medium': {'calcium': [600, 799], 'protein': [40, 59]},
                    'high': {'calcium': [0, 599], 'protein': [0, 39]}
                },
                'symptoms': {
                    'low': ['��΢���ܶ��½�'],
                    'medium': ['�ؽ���ʹ', '������ʹ'],
                    'high': ['��ʹ����', '���׹���', '��߱䰫', '�ձ�']
                }
            },
            'ƶѪ': {
                'thresholds': {
                    'low': {'iron': [15, float('inf')], 'protein': [60, float('inf')]},
                    'medium': {'iron': [10, 14], 'protein': [40, 59]},
                    'high': {'iron': [0, 9], 'protein': [0, 39]}
                },
                'symptoms': {
                    'low': ['��΢ƣ��'],
                    'medium': ['ƣ�ͷ���', '��ɫ�԰�'],
                    'high': ['����ƣ��', 'ͷ���ļ�', 'ָ�ײ԰�', 'ע����������']
                }
            },
            '��Ѫ�ܼ���': {
                'thresholds': {
                    'low': {'fat': [0, 65], 'sodium': [0, 2000]},
                    'medium': {'fat': [66, 90], 'sodium': [2001, 3000]},
                    'high': {'fat': [91, float('inf')], 'sodium': [3001, float('inf')]}
                },
                'symptoms': {
                    'low': ['ż������'],
                    'medium': ['��ʹ����', '�ļ�'],
                    'high': ['��ʹ����', '��������', '���ɲ���', 'ˮ��']
                }
            }
        }
        
        # �������Բ�����
        warning_results = []
        total_nutrition = nutrition_summary['total_nutrition']
        
        # ת��Ӫ�����ݣ�����ת�ˣ�����һ���ԣ�
        daily_nutrition = {
            'carbs': total_nutrition.get('̼ˮ������(g)', 0) / 7,  # ��ƽ��ת��ƽ��
            'fiber': total_nutrition.get('��ʳ��ά(g)', 0) / 7,
            'sodium': total_nutrition.get('��(mg)', 0) / 7,
            'potassium': total_nutrition.get('��(mg)', 0) / 7, 
            'calcium': total_nutrition.get('��(mg)', 0) / 7,
            'protein': total_nutrition.get('������(g)', 0) / 7,
            'iron': total_nutrition.get('��(mg)', 0) / 7,
            'fat': total_nutrition.get('֬��(g)', 0) / 7
        }
        
        for disease_name, disease_config in chronic_diseases_config.items():
            risk_level = 'low'
            matched_criteria = []
            
            # ���߷�����ֵ
            for nutrient, threshold in disease_config['thresholds']['high'].items():
                value = daily_nutrition.get(nutrient, 0)
                if threshold[0] <= value <= threshold[1]:
                    risk_level = 'high'
                    matched_criteria.append(f"{nutrient}: {value:.1f}")
                    break
            
            # ����з�����ֵ
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
        
        # ����Ԥ������
        high_risk_diseases = [w for w in warning_results if w['risk_level'] == 'high']
        medium_risk_diseases = [w for w in warning_results if w['risk_level'] == 'medium']
        
        # ʹ��ǧ��������ϸ����
        if high_risk_diseases or medium_risk_diseases:
            warning_prompt = f"""��ΪרҵӪ��ʦ�ͽ������ʣ�������������Բ��������������

�߷��ռ�����{', '.join([d['disease'] for d in high_risk_diseases]) if high_risk_diseases else '��'}
�еȷ��ռ�����{', '.join([d['disease'] for d in medium_risk_diseases]) if medium_risk_diseases else '��'}

Ӫ�����ݣ��վ�ֵ����
- ̼ˮ������: {daily_nutrition['carbs']:.1f}g
- ��ʳ��ά: {daily_nutrition['fiber']:.1f}g  
- ��: {daily_nutrition['sodium']:.1f}mg
- ��: {daily_nutrition['calcium']:.1f}mg
- ������: {daily_nutrition['protein']:.1f}g
- ��: {daily_nutrition['iron']:.1f}mg
- ֬��: {daily_nutrition['fat']:.1f}g

���ṩ��
1. ���շ����ܽᣨ150�����ڣ�
2. �����Ԥ�����飨150�����ڣ�

ע�⣺ʹ�ü�����˵����ֻش𣬲�Ҫʹ���κ�Markdown��ʽ��"""

            analysis = query_qwen([{"role": "user", "content": [{"type": "text", "text": warning_prompt}]}])
        else:
            analysis = "����Ӫ�����ݷ������������Բ�����ָ����������������־�����ʳ�ͽ������ʽ��"
        
        # ����Ԥ������
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
        
        # ���浽�ļ�
        warning_file = os.path.join(NUTRITION_DIR, f"warnings_{ts}.json")
        with open(warning_file, "w", encoding="utf-8") as f:
            json.dump(warning_data, f, ensure_ascii=False, indent=2)
        
        # ��������Ԥ���ļ�
        latest_warning_file = os.path.join(NUTRITION_DIR, "latest_warnings.json")
        with open(latest_warning_file, "w", encoding="utf-8") as f:
            json.dump(warning_data, f, ensure_ascii=False, indent=2)
        
        print(f"[Warning] ����Ԥ��������: {warning_file}")
        return warning_file
        
    except Exception as e:
        print(f"[Warning] ���ɽ���Ԥ��ʧ��: {e}")
        return None

def get_recommended_dishes(warnings_data):
    """���ڽ���Ԥ�����ݻ�ȡ�Ƽ���Ʒ�б�"""
    try:
        # �������յȼ���Ӫ������
        high_risk_diseases = []
        medium_risk_diseases = []
        recommended_dishes = []
        
        # ����Ԥ������
        if 'warnings' in warnings_data:
            warnings_list = warnings_data['warnings']
            for warning in warnings_list:
                if warning['risk_level'] == 'high':
                    high_risk_diseases.append(warning['disease'])
                elif warning['risk_level'] == 'medium':
                    medium_risk_diseases.append(warning['disease'])
        
        # ���ݼ�������ѡ����ʵĲ�Ʒ
        if high_risk_diseases:
            for disease in high_risk_diseases:
                if '����' in disease:
                    # ѡ����ǵ�GIʳ��
                    candidates = ['������', '��������', '���Ʋ���', '����������']
                    recommended_dishes.extend(random.sample(candidates, min(2, len(candidates))))
                elif '��Ѫѹ' in disease:
                    # ѡ����Ƹ߼�ʳ��
                    candidates = ['���Ʋ���', '���Ϲ�', '������', '��������']
                    recommended_dishes.extend(random.sample(candidates, min(2, len(candidates))))
                elif '��Ѫ�ܼ���' in disease:
                    # ѡ���֬ʳ��
                    candidates = ['��������', '������', '���Ʋ���', '������']
                    recommended_dishes.extend(random.sample(candidates, min(2, len(candidates))))
                elif '��������' in disease:
                    # ѡ��߸�ʳ��
                    candidates = ['Ϻ�ʶ���', '֥���', '�ϲ˵�����', '���Ŷ���']
                    recommended_dishes.extend(random.sample(candidates, min(2, len(candidates))))
                elif 'ƶѪ' in disease:
                    # ѡ�񸻺����ʵ�ʳ��
                    candidates = ['��γ�����', '�²˳���', '��ľ������', '֥�鲤��']
                    recommended_dishes.extend(random.sample(candidates, min(2, len(candidates))))
        
        elif medium_risk_diseases:
            # �еȷ��գ�ѡ�����Ӫ���Ĳ�Ʒ
            candidates = ['������', '������', '����������', '���Ŷ���', '���м�']
            recommended_dishes.extend(random.sample(candidates, min(3, len(candidates))))
        
        else:
            # ����״�����ã�����Ƽ�Ӫ������Ĳ�Ʒ
            all_dishes = list(DISH_NUTRITION_DATABASE.keys())
            recommended_dishes = random.sample(all_dishes, min(3, len(all_dishes)))
        
        # ȥ�ز�����Ϊ3����Ʒ
        recommended_dishes = list(set(recommended_dishes))[:3]
        
        # ����Ƽ���Ʒ����3��������һЩ������Ʒ
        if len(recommended_dishes) < 3:
            healthy_dishes = ['������', '��������', '����������', '���Ŷ���', '���м�']
            for dish in healthy_dishes:
                if dish not in recommended_dishes and len(recommended_dishes) < 3:
                    recommended_dishes.append(dish)
        
        return recommended_dishes
        
    except Exception as e:
        print(f"[FOOD_REC] ��ȡ�Ƽ���Ʒʧ��: {e}")
        return ['������', '��������', '����������']

def generate_food_recommendation(warnings_data):
    """���ڽ���Ԥ���������ɲ�Ʒ�Ƽ�"""
    try:
        # ��ȡ�Ƽ���Ʒ�б�
        recommended_dishes = get_recommended_dishes(warnings_data)
        
        # ������Ҫ�����Ӫ����
        nutrients_needed = []
        if 'warnings' in warnings_data:
            warnings_list = warnings_data['warnings']
            for warning in warnings_list:
                if warning['risk_level'] in ['high', 'medium']:
                    disease = warning['disease']
                    if '����' in disease:
                        nutrients_needed.append('��ʳ��ά')
                    elif '��Ѫѹ' in disease:
                        nutrients_needed.append('��')
                    elif '��Ѫ�ܼ���' in disease:
                        nutrients_needed.append('������֬����')
                    elif '��������' in disease:
                        nutrients_needed.append('��')
                    elif 'ƶѪ' in disease:
                        nutrients_needed.append('��')
        
        # ȥ��Ӫ����
        nutrients_needed = list(set(nutrients_needed))
        
        # ���������Ƽ��ı�
        if len(recommended_dishes) >= 3:
            recommendation = f"��������Ӫ������״̬���Ƽ�1.{recommended_dishes[0]}��2.{recommended_dishes[1]}��3.{recommended_dishes[2]}������"
        else:
            recommendation = f"��������Ӫ������״̬���Ƽ�{', '.join(recommended_dishes)}�Ȳ�Ʒ"
        
        if nutrients_needed:
            if len(nutrients_needed) == 1:
                recommendation += f"������{nutrients_needed[0]}Ӫ���ء�"
            elif len(nutrients_needed) == 2:
                recommendation += f"������{nutrients_needed[0]}��{nutrients_needed[1]}����Ӫ���ء�"
            else:
                recommendation += f"������{', '.join(nutrients_needed[:2])}��Ӫ���ء�"
        else:
            recommendation += "������Ӫ�����⡣"
        
        return recommendation
        
    except Exception as e:
        print(f"[FOOD_REC] �����Ƽ�ʧ��: {e}")
        return "��������Ӫ������״̬���Ƽ�1.��������2.�������㣬3.���������������ˣ����䵰���ʡ�ά����Ӫ���ء�"

def evaluate_food_recommendation(detected_foods, warnings_data):
    """������⵽��ʳ���Ƿ��Ƽ�ʳ��"""
    try:
        if not warnings_data:
            return "���޽������ݣ���������ʳ�á�", "neutral"
        
        # �����⵽���������ϵĲˣ�ֻ�Ƽ�һ����
        if len(detected_foods) >= 2:
            # �����߷��ռ�����ȱ����Ӫ����
            high_risk_diseases = []
            lacking_nutrients = []
            
            for disease, data in warnings_data.items():
                if isinstance(data, dict) and data.get('risk_level') == '�߷���':
                    high_risk_diseases.append(disease)
                    # ���ݼ��������ƶ�ȱ����Ӫ����
                    if disease == '����':
                        lacking_nutrients.extend(['��ʳ��ά', '������'])
                    elif disease == '��Ѫѹ':
                        lacking_nutrients.extend(['��', 'þ'])
                    elif disease == '��Ѫ�ܼ���':
                        lacking_nutrients.extend(['Omega-3֬����', 'ά����E'])
                    elif disease == '��������':
                        lacking_nutrients.extend(['��', 'ά����D'])
                    elif disease == 'ƶѪ':
                        lacking_nutrients.extend(['��', 'ά����B12'])
            
            # ȥ��Ӫ����
            lacking_nutrients = list(set(lacking_nutrients))
            
            # ����ÿ��ʳ��������Ե÷�
            food_scores = []
            for food in detected_foods:
                food_name = food.get('dish_name', '')
                nutrients = food.get('nutrients', {})
                
                score = 0
                # �����÷�
                score += food.get('confidence', 0) * 100
                
                # ����Ӫ���غ��������÷�
                protein = nutrients.get('������(g)', 0)
                calcium = nutrients.get('��(mg)', 0)
                iron = nutrients.get('��(mg)', 0)
                
                # �����ʷḻ�ӷ�
                if protein > 10:
                    score += 20
                # �ƺ����ḻ�ӷ�
                if calcium > 100:
                    score += 15
                # �������ḻ�ӷ�
                if iron > 2:
                    score += 15
                
                # ������ʳ�����
                sugar = nutrients.get('̼ˮ������(g)', 0)
                sodium = nutrients.get('��(mg)', 0)
                fat = nutrients.get('֬��(g)', 0)
                
                if sugar > 30:
                    score -= 30
                if sodium > 500:
                    score -= 25
                if fat > 20:
                    score -= 20
                
                food_scores.append((food_name, score))
            
            # ѡ��÷���ߵ�ʳ��
            best_food = max(food_scores, key=lambda x: x[1])[0]
            
            # �����Ƽ��ı�
            if lacking_nutrients:
                nutrients_text = '��'.join(lacking_nutrients[:2])  # �����ʾ����Ӫ����
                result_text = f"�Ƽ�����{best_food}����Ϊ��ȱ��{nutrients_text}"
            else:
                result_text = f"�Ƽ�����{best_food}����Ϊ��Ӫ�������ʺ���"
            
            return result_text, "recommended"
        
        # ���ֻ��һ���ˣ���ԭ�߼�����
        else:
            # �����߷��ռ���
            high_risk_diseases = []
            for disease, data in warnings_data.items():
                if isinstance(data, dict) and data.get('risk_level') == '�߷���':
                    high_risk_diseases.append(disease)
            
            # �򵥵�ʳ�������߼�
            not_recommended = []
            recommended = []
            
            for food in detected_foods:
                food_name = food.get('dish_name', '').lower()
                nutrients = food.get('nutrients', {})
                
                # ��ȡ�ؼ�Ӫ����
                sugar = nutrients.get('̼ˮ������(g)', 0)
                sodium = nutrients.get('��(mg)', 0)
                fat = nutrients.get('֬��(g)', 0)
                
                is_recommended = True
                reasons = []
                
                # ���򲡷�������
                if '����' in high_risk_diseases:
                    if sugar > 30 or '��' in food_name or '��' in food_name:
                        is_recommended = False
                        reasons.append('�������ϸߣ����ʺ����򲡷�����Ⱥ')
                
                # ��Ѫѹ��������
                if '��Ѫѹ' in high_risk_diseases:
                    if sodium > 500 or '��' in food_name or '��' in food_name:
                        is_recommended = False
                        reasons.append('�ƺ����ϸߣ����ʺϸ�Ѫѹ������Ⱥ')
                
                # ��Ѫ�ܼ�����������
                if '��Ѫ�ܼ���' in high_risk_diseases:
                    if fat > 20 or 'ը' in food_name or '��' in food_name:
                        is_recommended = False
                        reasons.append('֬�������ϸߣ����ʺ���Ѫ�ܼ���������Ⱥ')
                
                if is_recommended:
                    recommended.append(food_name)
                else:
                    not_recommended.append((food_name, reasons))
            
            # �����������
            if not_recommended:
                result_text = "������ʳ������ʳ�\n"
                for food, reasons in not_recommended:
                    result_text += f"? {food}: {', '.join(reasons)}\n"
                if recommended:
                    result_text += f"\n��������ʳ�ã�{', '.join(recommended)}"
                return result_text, "not_recommended"
            elif recommended:
                return f"��⵽��ʳ�ﶼ�Ƚ��ʺ�����{', '.join(recommended)}����������ʳ�á�", "recommended"
            else:
                return "δ������ʳ�������ԣ�������ѯӪ��ʦ��", "neutral"
            
    except Exception as e:
        print(f"[FOOD_EVAL] ����ʧ��: {e}")
        return "ʳ������������ʱ�����ã���������ʳ�á�", "neutral"

async def ws_handler(ws):
    print("[Srv] ? new conn", ws.remote_address)
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    rec.SetWords(True)

    last_photo_ts = 0
    video_on = False
    last_partial = ""
    pending_text = ""
    awaiting_photo_model = False
    # �����¼��ر���
    meeting_recording = False
    meeting_file = None
    # ��Ƶ¼�ƻỰID
    video_session_id = None

    async for chunk in ws:
        if isinstance(chunk, bytes):
            if rec.AcceptWaveform(chunk):
                result = json.loads(rec.Result())["text"].replace(" ", "")
                if result:
                    print(f"[FINAL] {result}")
                    pending_text = result
                    
                    # �����¼�����߼�
                    if "�����¼��ʼ" in result:
                        if not meeting_recording:
                            meeting_file = start_meeting_recording()
                            meeting_recording = True
                            print("[Srv] �����¼ģʽ�ѿ���")
                        continue
                    elif "�����¼����" in result or "����ʶ�����" in result:
                        if meeting_recording:
                            end_meeting_recording(meeting_file)
                            meeting_recording = False
                            meeting_file = None
                            print("[Srv] �����¼ģʽ�ѹر�")
                        continue
                    
                    # �����¼ģʽ�£���¼�����������ݣ�������AI����
                    if meeting_recording:
                        append_meeting_text(meeting_file, result)
                        continue
                    
                    # �������չ��ܣ�����Ҫ�������Ѵʣ�
                    if "����" in result and not detect_maili_wakeup(result):
                        if time.time() - last_photo_ts > 3:
                            await ws.send(json.dumps({"cmd": "photo"}))
                            last_photo_ts = time.time()
                            print("[Srv] ? �������� cmd")
                        awaiting_photo_model = "simple_photo"
                    
                    # ���Ѵʺ������߼�
                    elif detect_maili_wakeup(result):
                        if "����" in result:
                            if time.time() - last_photo_ts > 3:
                                await ws.send(json.dumps({"cmd": "photo"}))
                                last_photo_ts = time.time()
                                print("[Srv] ? photo cmd")
                            awaiting_photo_model = True
                        elif "ʳ��ʶ��" in result or "ʵ��ʶ��" in result:
                            if time.time() - last_photo_ts > 3:
                                await ws.send(json.dumps({"cmd": "photo"}))
                                last_photo_ts = time.time()
                                print("[Srv] ? food recognition photo cmd")
                            awaiting_photo_model = "food_recommend"
                        elif "��Ҫ�����" in result:
                            if time.time() - last_photo_ts > 3:
                                await ws.send(json.dumps({"cmd": "photo"}))
                                last_photo_ts = time.time()
                                print("[Srv] ? food record photo cmd")
                            awaiting_photo_model = "food_record"
                        elif "��Ʒ�Ƽ�" in result:
                            # ֱ�Ӷ�ȡlatest_warnings.json���Ƽ�
                            try:
                                latest_warning_file = os.path.join(NUTRITION_DIR, "latest_warnings.json")
                                if os.path.exists(latest_warning_file):
                                    with open(latest_warning_file, "r", encoding="utf-8") as f:
                                        warnings_data = json.load(f)
                                    
                                    # ���ɲ�Ʒ�Ƽ��ı�
                                    recommendation_text = generate_food_recommendation(warnings_data)
                                    print(f"[FOOD_REC] {recommendation_text}")
                                    
                                    # ��ȡ�Ƽ������������Ʒ
                                    recommended_dishes = get_recommended_dishes(warnings_data)
                                    
                                    # ������Ʒ���ݽṹ
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
                                    resp = "���޽���Ԥ�����ݣ��޷����ɸ��Ի���Ʒ�Ƽ������Ƚ���ʳ��ʶ���Խ�������������"
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
                                print(f"[FOOD_REC] ��Ʒ�Ƽ�ʧ��: {e}")
                                resp = "��Ʒ�Ƽ�������ʱ�����ã����Ժ����ԡ�"
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
                            # ���ı�����
                            print("[info] ���� Qwen2.5-VL �ı����� ��")
                            resp = query_qwen([
                                {"role": "user", "content": [{"type": "text", "text": pending_text}]}])
                            print(f"[QWEN] �ı��ظ���{resp}")
                            # ���� TTS ��������ʷ TXT
                            baidu_tts.baidu_tts_test(resp)
                            mp3 = open(TTS_FILE, "rb").read()
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            os.makedirs("history", exist_ok=True)
                            with open(os.path.join("history", f"response_{ts}.txt"), "w", encoding="utf-8") as f:
                                f.write(f"Q: {pending_text}\nA: {resp}\n")
                            # ���͸� Pi
                            await ws.send(json.dumps({
                                "tag": "response",
                                "question": pending_text,
                                "text": resp,
                                "payload_hex": mp3.hex(),
                                "filename": ts
                            }))
                            awaiting_photo_model = False
                            pending_text = ""
                    # ¼�����
                    if "¼����ͣ" in result or "ֹͣ¼��" in result:
                        if video_on:
                            await ws.send(json.dumps({"cmd": "video_stop"}))
                            video_on = False
                            # ������Ƶ�ļ�
                            if video_session_id:
                                video_path = create_video_from_frames(video_session_id)
                                if video_path:
                                    print(f"[Srv] ��Ƶ�ѱ���: {video_path}")
                                video_session_id = None
                            print("[Srv] ?? video_stop")
                    elif "¼��" in result and not video_on:
                        video_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                        await ws.send(json.dumps({"cmd": "video_start"}))
                        video_on = True
                        print(f"[Srv] ?? video_start, session: {video_session_id}")
                last_partial = ""
            else:
                # ���� PART ���
                partial = json.loads(rec.PartialResult())["partial"].replace(" ", "")
                if partial and partial != last_partial:
                    print(f"[PART]  {partial}", end="\r", flush=True)
                    last_partial = partial
        else:
            data = json.loads(chunk)
            tag = data.get("tag")
            if tag == "periodic_detection":
                # ����ʱ�����Ƭ
                img_bytes = bytes.fromhex(data["payload_hex"])
                
                # ����ԭʼ��Ƭ����ʱλ��
                temp_photo_path = os.path.join("/tmp", f"periodic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                with open(temp_photo_path, "wb") as f:
                    f.write(img_bytes)
                
                # ��ȡͼ����ж�Ŀ����
                image = cv2.imread(temp_photo_path)
                if image is not None:
                    try:
                        # ���ж�Ŀ����
                        results = multi_target_detector.process_frame(image)
                        
                        # ���ӻ����
                        annotated_image = multi_target_detector.visualize_results(image, results)
                        
                        # ������ͼƬ����Ŀ¼��������һ�εĽ����
                        result_path = "/root/autodl-tmp/.autodl/iot/periodic_detection_result.jpg"
                        cv2.imwrite(result_path, annotated_image)
                        print(f"[��ʱ���] ����ɲ�������")
                        
                        # ������ʱ�ļ�
                        os.remove(temp_photo_path)
                        
                    except Exception as e:
                        print(f"[��ʱ���] ����ʧ��: {e}")
                        # ������ʱ�ļ�
                        if os.path.exists(temp_photo_path):
                            os.remove(temp_photo_path)
                else:
                    print("[��ʱ���] ͼ���ȡʧ��")
                    # ������ʱ�ļ�
                    if os.path.exists(temp_photo_path):
                        os.remove(temp_photo_path)
                        
            elif tag == "photo":
                img_bytes = bytes.fromhex(data["payload_hex"])
                photo_path = save_photo(img_bytes)
                
                # �������չ��ܣ���������Ƭ��������AI��
                if awaiting_photo_model == "simple_photo":
                    print(f"[Srv] ���������ѱ���: {photo_path}")
                    awaiting_photo_model = False
                    pending_text = ""
                elif awaiting_photo_model == "food_recommend" and detect_maili_wakeup(pending_text) and ("ʳ��ʶ��" in pending_text or "ʵ��ʶ��" in pending_text):
                     # ��Ŀ��ʳƷʶ�����
                     print("[info] ������Ŀ��ʳƷʶ�� ��")
                     
                     # ��ȡͼ��
                     image = cv2.imread(photo_path)
                     if image is None:
                         resp = "��Ǹ���޷���ȡͼ��������"
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
                     
                     # ���ж�Ŀ����
                     detection_results = multi_target_detector.process_frame(image)
                     
                     if detection_results['total_detected'] > 0:
                         # ���ɿ��ӻ����
                         annotated_image = multi_target_detector.visualize_results(image, detection_results)
                         
                         # ������ӻ����
                         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                         result_image_path = os.path.join(PHOTO_DIR, f"multi_detection_{ts}.jpg")
                         cv2.imwrite(result_image_path, annotated_image)
                         
                         # ��ȡ���м�⵽��ʳ����Ϣ
                         all_foods = [{
                             'dish_name': food['chinese_food_result']['dish_name'],
                             'confidence': food['chinese_food_result']['confidence'],
                             'nutrients': food['chinese_food_result']['nutrients']
                         } for food in detection_results['food_detections']]
                         
                         # ��ȡ����Ԥ�����ݽ����Ƽ�����
                         try:
                             latest_warning_file = os.path.join(NUTRITION_DIR, "latest_warnings.json")
                             warnings_data = {}
                             if os.path.exists(latest_warning_file):
                                 with open(latest_warning_file, "r", encoding="utf-8") as f:
                                     warnings_data = json.load(f)
                             
                             # ����ʳ���Ƽ�
                             recommendation_text, recommendation_status = evaluate_food_recommendation(all_foods, warnings_data)
                             
                             # �������������
                             food_names = [food['dish_name'] for food in all_foods]
                             detection_desc = f"��⵽{len(all_foods)}��ʳ�{', '.join(food_names)}��"
                             
                             full_resp = f"{detection_desc}\n\n{recommendation_text}"
                             print(f"[FOOD_REC] {full_resp}")
                             
                         except Exception as e:
                             print(f"[FOOD_REC] �Ƽ�����ʧ��: {e}")
                             food_names = [food['dish_name'] for food in all_foods]
                             full_resp = f"��⵽{len(all_foods)}��ʳ�{', '.join(food_names)}����������ʳ�á�"
                         
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
                         resp = "��Ǹ��δ��⵽�κ�ʳ��Ŀ�꣬������"
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
                     # ʳ���¼����
                     print("[info] ����ʳ���¼���� ��")
                     
                     # ��ȡͼ��
                     image = cv2.imread(photo_path)
                     if image is None:
                         resp = "��Ǹ���޷���ȡͼ��������"
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
                     
                     # ���ж�Ŀ����
                     detection_results = multi_target_detector.process_frame(image)
                     
                     if detection_results['total_detected'] > 0:
                         # ���ɿ��ӻ����
                         annotated_image = multi_target_detector.visualize_results(image, detection_results)
                         
                         # ������ӻ����
                         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                         result_image_path = os.path.join(PHOTO_DIR, f"food_record_{ts}.jpg")
                         cv2.imwrite(result_image_path, annotated_image)
                         
                         # ѡ��Ҫ��¼��ʳ��
                         selected_food = detection_results.get('selected_food')
                         if selected_food:
                             # ����ѡ����ʳ��
                             dish_name = selected_food['chinese_food_result']['dish_name']
                             confidence = selected_food['chinese_food_result']['confidence']
                             nutrients = selected_food['chinese_food_result']['nutrients']
                             
                             resp = f"�ѽ�{dish_name}��Ʒ��¼�����Ŷ�{confidence:.2f}"
                         else:
                             # û������ѡ��ѡ����Ļ�����Ĳ�Ʒ�����Ŷ���ߣ�
                             best_food = max(detection_results['food_detections'], 
                                            key=lambda x: x['chinese_food_result']['confidence'])
                             dish_name = best_food['chinese_food_result']['dish_name']
                             confidence = best_food['chinese_food_result']['confidence']
                             nutrients = best_food['chinese_food_result']['nutrients']
                             
                             if detection_results['gesture']:
                                 resp = f"����δָ����ȷ��Ʒ���Ѽ�¼��Ļ�����Ĳ�Ʒ��{dish_name}�����Ŷ�{confidence:.2f}"
                             else:
                                 resp = f"δ��⵽���ƣ��Ѽ�¼��Ļ�����Ĳ�Ʒ��{dish_name}�����Ŷ�{confidence:.2f}"
                         
                         # ����ʳ���¼
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
                         
                         # ���Ӫ����Ϣ
                         key_nutrients = ["����(kcal)", "������(g)", "֬��(g)", "̼ˮ������(g)"]
                         nutrient_summary = ", ".join([f"{k}:{nutrients.get(k, 0):.1f}" for k in key_nutrients if k in nutrients])
                         
                         full_resp = f"{resp}��Ӫ���ɷ֣�{nutrient_summary}����¼�ѱ��档"
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
                         
                         # ����Ӫ������
                         try:
                             print("[Nutrition] ��������Ӫ������...")
                             nutrition_summary = get_weekly_nutrition_summary()
                             advice = generate_nutrition_advice(nutrition_summary)
                             save_nutrition_advice(advice)
                             print("[Nutrition] Ӫ������������")
                         except Exception as e:
                             print(f"[Nutrition] Ӫ����������ʧ��: {e}")
                     else:
                         resp = "��Ǹ��δ��⵽�κ�ʳ��Ŀ�꣬������"
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
                elif awaiting_photo_model == True and detect_maili_wakeup(pending_text) and "����" in pending_text:
                    # ͼ�ĵ���
                    print("[info] ���� Qwen2.5-VL ͼ������ ��")
                    resp = query_qwen([{
                        "role": "user",
                        "content":[
                            {"type":"image","image":f"file://{photo_path}"},
                            {"type":"text","text":pending_text}
                        ]
                    }])
                    print(f"[QWEN] ͼ�Ļظ���{resp}")
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
                    # Ϊ��Ƶ֡��ӻỰIDǰ׺
                    img_bytes = bytes.fromhex(data["payload_hex"])
                    fn = f"{video_session_id}_{datetime.now().strftime('%H%M%S_%f')}.jpg"
                    path = os.path.join(VIDEO_DIR, fn)
                    with open(path, "wb") as f:
                        f.write(img_bytes)
                else:
                    save_frame(bytes.fromhex(data["payload_hex"]))
            else:
                print("[Srv] txt:", data)


# HTTP������������
async def handle_photos(request):
    """������ƬĿ¼����"""
    try:
        files = os.listdir(PHOTO_DIR)
        photo_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))]
        
        # ���ɼ򵥵�HTMLĿ¼�б�
        html = "<html><body><h1>Photos Directory</h1><ul>"
        for file in sorted(photo_files, reverse=True):
            html += f'<li><a href="{file}">{file}</a></li>'
        html += "</ul></body></html>"
        
        return web.Response(text=html, content_type='text/html')
    except Exception as e:
        return web.Response(text=f"Error: {str(e)}", status=500)

async def handle_videos(request):
    """������ƵĿ¼����"""
    try:
        files = os.listdir(VIDEOS_DIR)
        video_files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv'))]
        
        # ���ɼ򵥵�HTMLĿ¼�б�
        html = "<html><body><h1>Videos Directory</h1><ul>"
        for file in sorted(video_files, reverse=True):
            html += f'<li><a href="{file}">{file}</a></li>'
        html += "</ul></body></html>"
        
        return web.Response(text=html, content_type='text/html')
    except Exception as e:
        return web.Response(text=f"Error: {str(e)}", status=500)

async def handle_photo_file(request):
    """��������Ƭ�ļ�����"""
    filename = request.match_info['filename']
    file_path = os.path.join(PHOTO_DIR, filename)
    
    if not os.path.exists(file_path):
        return web.Response(text="File not found", status=404)
    
    try:
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
        
        # �����ļ���չ������Content-Type
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
    """��������Ƶ�ļ�����"""
    filename = request.match_info['filename']
    file_path = os.path.join(VIDEOS_DIR, filename)
    
    if not os.path.exists(file_path):
        return web.Response(text="File not found", status=404)
    
    try:
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
        
        # �����ļ���չ������Content-Type
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
    """����Ӫ����������"""
    try:
        latest_file = os.path.join(NUTRITION_DIR, "latest_advice.json")
        if os.path.exists(latest_file):
            async with aiofiles.open(latest_file, 'r', encoding='utf-8') as f:
                content = await f.read()
            return web.Response(text=content, content_type='application/json; charset=utf-8')
        else:
            # ���û�����½��飬����Ĭ����Ϣ
            default_advice = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation": "����Ӫ�����ۣ����Ƚ���ʳ��ʶ�������ɸ��Ի����顣",
                "recommendation": "������ʳ�Ƽ������Ƚ���ʳ��ʶ�������ɸ��Ի����顣",
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
    """����HTTPӦ��"""
    app = web.Application()
    
    # ���·��
    app.router.add_get('/photos/', handle_photos)
    app.router.add_get('/videos/', handle_videos)
    app.router.add_get('/photos/{filename}', handle_photo_file)
    app.router.add_get('/videos/{filename}', handle_video_file)
    app.router.add_get('/api/nutrition-advice', handle_nutrition_advice)

    
    # ��̬�ļ�����
    app.router.add_static('/web/', path='web', name='web')
    app.router.add_static('/food_recognition/', path=FOOD_DIR, name='food_recognition')
    app.router.add_static('/meeting_recording/', path=MEETING_DIR, name='meeting_recording')
    app.router.add_static('/history/', path='history', name='history')
    app.router.add_static('/nutrition_analysis/', path=NUTRITION_DIR, name='nutrition_analysis')
    
    return app

async def main():
    # ����HTTP������
    http_app = await create_http_app()
    http_runner = web_runner.AppRunner(http_app)
    await http_runner.setup()
    http_site = web_runner.TCPSite(http_runner, '0.0.0.0', 8080)
    await http_site.start()
    print("[Srv] HTTP server started on http://0.0.0.0:8080")
    
    # ����WebSocket������
    async with websockets.serve(ws_handler, "0.0.0.0", 6006, ping_interval=20, max_size=None):
        print("[Srv] ? ws://0.0.0.0:6006/ws (vosk + Qwen + TTS ready)")
        print("[Srv] ý���ļ�����: http://0.0.0.0:8080/photos/ �� http://0.0.0.0:8080/videos/")
        await asyncio.Future()

if __name__ == "__main__": asyncio.run(main())