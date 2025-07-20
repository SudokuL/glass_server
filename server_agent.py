# -*- coding: gbk -*-
# server_agent.py  �� ʵʱ PART+FINAL ��� + Qwen2.5-VL ���� + TTS
#!/usr/bin/env python3

import asyncio, websockets, json, os, time
from datetime import datetime
from vosk import Model, KaldiRecognizer

# --- Qwen2.5-VL ��� ---
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# --- Baidu TTS ---
import baidu_tts

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


def query_qwen(messages):
    """���� Qwen��messages ǰע��ϵͳ Prompt"""
    system_prompt = { "role": "system", "content": "��С�Ī˹�ơ������ҵ�˽���������֡�" }
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

# --- Vosk ʶ����� ---
MODEL_PATH  = "model/vosk-model-cn-0.22"
SAMPLE_RATE = 16000
vosk_model = Model(MODEL_PATH)

PHOTO_DIR = "photos"; VIDEO_DIR = "video_frames"
os.makedirs(PHOTO_DIR, exist_ok=True); os.makedirs(VIDEO_DIR, exist_ok=True)


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

async def ws_handler(ws):
    print("[Srv] ? new conn", ws.remote_address)
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    rec.SetWords(True)

    last_photo_ts = 0
    video_on = False
    last_partial = ""
    pending_text = ""
    awaiting_photo_model = False

    async for chunk in ws:
        if isinstance(chunk, bytes):
            if rec.AcceptWaveform(chunk):
                result = json.loads(rec.Result())["text"].replace(" ", "")
                if result:
                    print(f"[FINAL] {result}")
                    pending_text = result
                    # ���Ѵʺ������߼�
                    if "Ī˹��" in result:
                        if "����" in result:
                            if time.time() - last_photo_ts > 3:
                                await ws.send(json.dumps({"cmd": "photo"}))
                                last_photo_ts = time.time()
                                print("[Srv] ? photo cmd")
                            awaiting_photo_model = True
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
                            print("[Srv] ?? video_stop")
                    elif "¼��" in result and not video_on:
                        await ws.send(json.dumps({"cmd": "video_start"}))
                        video_on = True
                        print("[Srv] ?? video_start")
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
                if awaiting_photo_model and "Ī˹��" in pending_text and "����" in pending_text:
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
                save_frame(bytes.fromhex(data["payload_hex"]))
            else:
                print("[Srv] txt:", data)


async def main():
    async with websockets.serve(ws_handler, "0.0.0.0", 6006, ping_interval=20, max_size=None):
        print("[Srv] ? ws://0.0.0.0:6006/ws (vosk + Qwen + TTS ready)")
        await asyncio.Future()

if __name__ == "__main__": asyncio.run(main())
