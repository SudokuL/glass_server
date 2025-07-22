# -*- coding: gbk -*-
# server_agent.py  — 实时 PART+FINAL 输出 + Qwen2.5-VL 调用 + TTS
#!/usr/bin/env python3

import asyncio, websockets, json, os, time
from datetime import datetime
from vosk import Model, KaldiRecognizer
from aiohttp import web, web_runner
import aiofiles

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

# --- 视频处理 ---
import cv2
import glob

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
        return advice_file
    except Exception as e:
        print(f"[Nutrition] 保存营养建议失败: {e}")
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
                            awaiting_photo_model = "food"
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
            if tag == "photo":
                img_bytes = bytes.fromhex(data["payload_hex"])
                photo_path = save_photo(img_bytes)
                
                # 独立拍照功能（仅保存照片，不调用AI）
                if awaiting_photo_model == "simple_photo":
                    print(f"[Srv] 独立拍照已保存: {photo_path}")
                    awaiting_photo_model = False
                    pending_text = ""
                elif awaiting_photo_model == "food" and detect_maili_wakeup(pending_text) and ("食物识别" in pending_text or "实物识别" in pending_text):
                     # 食品识别调用
                     print("[info] 触发食品识别 …")
                     food_result = food_recognizer.recognize_food(photo_path)
                     
                     if food_result:
                         dish_name = food_result.get('dish_name', '未知食品')
                         confidence = food_result.get('confidence', 0.0)
                         nutrients = food_result.get('nutrients', {})
                         
                         # 保存食品识别结果
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
                         
                         # 构建营养信息摘要
                         key_nutrients = ["热量(kcal)", "蛋白质(g)", "脂肪(g)", "碳水化合物(g)"]
                         nutrient_summary = ", ".join([f"{k}:{nutrients.get(k, 0):.1f}" for k in key_nutrients if k in nutrients])
                         
                         resp = f"识别到食品：{dish_name}，置信度：{confidence:.2f}。主要营养成分：{nutrient_summary}"
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
                         
                         # 自动生成营养建议
                         try:
                             print("[Nutrition] 开始生成营养建议...")
                             nutrition_summary = get_weekly_nutrition_summary()
                             advice = generate_nutrition_advice(nutrition_summary)
                             save_nutrition_advice(advice)
                             print("[Nutrition] 营养建议生成完成")
                         except Exception as e:
                             print(f"[Nutrition] 营养建议生成失败: {e}")
                     else:
                         resp = "抱歉，食品识别失败，请重试"
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
