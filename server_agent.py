# -*- coding: gbk -*-
# server_agent.py  �� ʵʱ PART+FINAL ��� + Qwen2.5-VL ���� + TTS
#!/usr/bin/env python3

import asyncio, websockets, json, os, time
from datetime import datetime
from vosk import Model, KaldiRecognizer
from aiohttp import web, web_runner
import aiofiles

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

# --- ��Ƶ���� ---
import cv2
import glob

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
        return advice_file
    except Exception as e:
        print(f"[Nutrition] ����Ӫ������ʧ��: {e}")
        return None

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
                            awaiting_photo_model = "food"
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
            if tag == "photo":
                img_bytes = bytes.fromhex(data["payload_hex"])
                photo_path = save_photo(img_bytes)
                
                # �������չ��ܣ���������Ƭ��������AI��
                if awaiting_photo_model == "simple_photo":
                    print(f"[Srv] ���������ѱ���: {photo_path}")
                    awaiting_photo_model = False
                    pending_text = ""
                elif awaiting_photo_model == "food" and detect_maili_wakeup(pending_text) and ("ʳ��ʶ��" in pending_text or "ʵ��ʶ��" in pending_text):
                     # ʳƷʶ�����
                     print("[info] ����ʳƷʶ�� ��")
                     food_result = food_recognizer.recognize_food(photo_path)
                     
                     if food_result:
                         dish_name = food_result.get('dish_name', 'δ֪ʳƷ')
                         confidence = food_result.get('confidence', 0.0)
                         nutrients = food_result.get('nutrients', {})
                         
                         # ����ʳƷʶ����
                         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                         food_result_path = os.path.join(FOOD_DIR, f"food_{ts}.json")
                         with open(food_result_path, "w", encoding="utf-8") as f:
                             json.dump({
                                 "timestamp": ts,
                                 "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                 "photo_path": photo_path,
                                 "dish_name": dish_name,
                                 "confidence": confidence,
                                 "nutrients": nutrients,
                                 "full_result": food_result
                             }, f, ensure_ascii=False, indent=2)
                         
                         # ����Ӫ����ϢժҪ
                         key_nutrients = ["����(kcal)", "������(g)", "֬��(g)", "̼ˮ������(g)"]
                         nutrient_summary = ", ".join([f"{k}:{nutrients.get(k, 0):.1f}" for k in key_nutrients if k in nutrients])
                         
                         resp = f"ʶ��ʳƷ��{dish_name}�����Ŷȣ�{confidence:.2f}����ҪӪ���ɷ֣�{nutrient_summary}"
                         print(f"[FOOD] {resp}")
                         
                         baidu_tts.baidu_tts_test(resp)
                         mp3 = open(TTS_FILE, "rb").read()
                         
                         await ws.send(json.dumps({
                             "tag": "response",
                             "question": pending_text,
                             "text": resp,
                             "payload_hex": mp3.hex(),
                             "filename": ts,
                             "food_result": food_result
                         }))
                         
                         # �Զ�����Ӫ������
                         try:
                             print("[Nutrition] ��ʼ����Ӫ������...")
                             nutrition_summary = get_weekly_nutrition_summary()
                             advice = generate_nutrition_advice(nutrition_summary)
                             save_nutrition_advice(advice)
                             print("[Nutrition] Ӫ�������������")
                         except Exception as e:
                             print(f"[Nutrition] Ӫ����������ʧ��: {e}")
                     else:
                         resp = "��Ǹ��ʳƷʶ��ʧ�ܣ�������"
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
