# -*- coding: gbk -*-
# server_agent.py  ?? ?? PART+FINAL ??? + Qwen2.5-VL ???? + TTS
#!/usr/bin/env python3

import asyncio, websockets, json, os, time
from datetime import datetime
from vosk import Model, KaldiRecognizer

# --- Qwen2.5-VL ??? ---
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# --- Baidu TTS ---
import baidu_tts

# --- ????? ---
import sys
sys.path.append('/root/autodl-tmp/.autodl/iot/food_cv')
from food_cv.GatedRegNet_Food import FoodRecognizer

# ???????????
QWEN_MODEL_PATH = "/root/autodl-tmp/.autodl/iot/qwen_7b_vl_offline"
# TTS ??????
TTS_FILE = "result.mp3"

print("[Srv] Loading Qwen2.5-VL model ??")
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

# ????????????
print("[Srv] Loading FoodRecognizer...")
food_recognizer = FoodRecognizer()
print("[Srv] FoodRecognizer ready")


def query_qwen(messages):
    """???? Qwen??messages ?????? Prompt"""
    system_prompt = { "role": "system", "content": "???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????" }
    all_msgs = [system_prompt] + messages
    text = processor.apply_chat_template(all_msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(all_msgs)
    inputs = processor(
        text=[text], images=imgs, videos=vids,
        padding=True, return_tensors="pt"
    ).to(qwen.device)
    with torch.no_grad():
        gen = qwen.generate(**inputs, max_new_tokens=128)
    out_ids = gen[0][inputs.input_ids.shape[1]:]
    return processor.batch_decode(
        [out_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

# --- Vosk ?????? ---
MODEL_PATH  = "model/vosk-model-cn-0.22"
SAMPLE_RATE = 16000
vosk_model = Model(MODEL_PATH)

PHOTO_DIR = "photos"; VIDEO_DIR = "video_frames"; MEETING_DIR = "meeting_recording"; FOOD_DIR = "food_recognition"
os.makedirs(PHOTO_DIR, exist_ok=True); os.makedirs(VIDEO_DIR, exist_ok=True); os.makedirs(MEETING_DIR, exist_ok=True); os.makedirs(FOOD_DIR, exist_ok=True)


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

def start_meeting_recording():
    """?????????"""
    fn = datetime.now().strftime("%Y%m%d_%H%M%S_meeting.txt")
    path = os.path.join(MEETING_DIR, fn)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"????????????: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")
    print(f"[Srv] ?????????: {fn}")
    return path

def append_meeting_text(meeting_file: str, text: str):
    """????????????"""
    if meeting_file and os.path.exists(meeting_file):
        with open(meeting_file, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] {text}\n")
        print(f"[Srv] ?????????: {text}")

def end_meeting_recording(meeting_file: str):
    """??????????"""
    if meeting_file and os.path.exists(meeting_file):
        with open(meeting_file, "a", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write(f"?????????????: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"[Srv] ??????????: {meeting_file}")

async def ws_handler(ws):
    print("[Srv] ? new conn", ws.remote_address)
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    rec.SetWords(True)

    last_photo_ts = 0
    video_on = False
    last_partial = ""
    pending_text = ""
    awaiting_photo_model = False
    # ????????????
    meeting_recording = False
    meeting_file = None

    async for chunk in ws:
        if isinstance(chunk, bytes):
            if rec.AcceptWaveform(chunk):
                result = json.loads(rec.Result())["text"].replace(" ", "")
                if result:
                    print(f"[FINAL] {result}")
                    pending_text = result
                    
                    # ?????????????
                    if "?????????" in result:
                        if not meeting_recording:
                            meeting_file = start_meeting_recording()
                            meeting_recording = True
                            print("[Srv] ?????????????")
                        continue
                    elif "??????????" in result or "??????????" in result:
                        if meeting_recording:
                            end_meeting_recording(meeting_file)
                            meeting_recording = False
                            meeting_file = None
                            print("[Srv] ????????????")
                        continue
                    
                    # ?????????????????????????????????AI????
                    if meeting_recording:
                        append_meeting_text(meeting_file, result)
                        continue
                    
                    # ?????????????
                    if "????" in result:
                        if "????" in result:
                            if time.time() - last_photo_ts > 3:
                                await ws.send(json.dumps({"cmd": "photo"}))
                                last_photo_ts = time.time()
                                print("[Srv] ? photo cmd")
                            awaiting_photo_model = True
                        elif "??????" in result or "??????" in result:
                            if time.time() - last_photo_ts > 3:
                                await ws.send(json.dumps({"cmd": "photo"}))
                                last_photo_ts = time.time()
                                print("[Srv] ? food recognition photo cmd")
                            awaiting_photo_model = "food"
                        else:
                            # ?????????
                            print("[info] ???? Qwen2.5-VL ??????? ??")
                            resp = query_qwen([
                                {"role": "user", "content": [{"type": "text", "text": pending_text}]}])
                            print(f"[QWEN] ????????{resp}")
                            # ???? TTS ????????? TXT
                            baidu_tts.baidu_tts_test(resp)
                            mp3 = open(TTS_FILE, "rb").read()
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            os.makedirs("history", exist_ok=True)
                            with open(os.path.join("history", f"response_{ts}.txt"), "w", encoding="utf-8") as f:
                                f.write(f"Q: {pending_text}\nA: {resp}\n")
                            # ????? Pi
                            await ws.send(json.dumps({
                                "tag": "response",
                                "question": pending_text,
                                "text": resp,
                                "payload_hex": mp3.hex(),
                                "filename": ts
                            }))
                            awaiting_photo_model = False
                            pending_text = ""
                    # ??????
                    if "??????" in result or "?????" in result:
                        if video_on:
                            await ws.send(json.dumps({"cmd": "video_stop"}))
                            video_on = False
                            print("[Srv] ?? video_stop")
                    elif "???" in result and not video_on:
                        await ws.send(json.dumps({"cmd": "video_start"}))
                        video_on = True
                        print("[Srv] ?? video_start")
                last_partial = ""
            else:
                # ???? PART ???
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
                if awaiting_photo_model == "food" and "????" in pending_text and ("??????" in pending_text or "??????" in pending_text):
                     # ????????
                     print("[info] ????????? ??")
                     food_result = food_recognizer.recognize_food(photo_path)
                     
                     if food_result:
                         dish_name = food_result.get('dish_name', '?????')
                         confidence = food_result.get('confidence', 0.0)
                         nutrients = food_result.get('nutrients', {})
                         
                         # ???????????
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
                         
                         # ????????????
                         key_nutrients = ["????(kcal)", "??????(g)", "???(g)", "????????(g)"]
                         nutrient_summary = ", ".join([f"{k}:{nutrients.get(k, 0):.1f}" for k in key_nutrients if k in nutrients])
                         
                         resp = f"???????{dish_name}????????{confidence:.2f}????????????{nutrient_summary}"
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
                     else:
                         resp = "????????????????????"
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
                elif awaiting_photo_model == True and "????" in pending_text and "????" in pending_text:
                    # ??????
                    print("[info] ???? Qwen2.5-VL ??????? ??")
                    resp = query_qwen([{
                        "role": "user",
                        "content":[
                            {"type":"image","image":f"file://{photo_path}"},
                            {"type":"text","text":pending_text}
                        ]
                    }])
                    print(f"[QWEN] ???????{resp}")
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
                save_frame(bytes.fromhex(data["payload_hex"]))
            else:
                print("[Srv] txt:", data)


async def main():
    async with websockets.serve(ws_handler, "0.0.0.0", 6006, ping_interval=20, max_size=None):
        print("[Srv] ? ws://0.0.0.0:6006/ws (vosk + Qwen + TTS ready)")
        await asyncio.Future()

if __name__ == "__main__": asyncio.run(main())
