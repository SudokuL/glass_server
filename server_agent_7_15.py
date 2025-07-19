#!/usr/bin/env python3
# -*- coding: gbk -*-
"""
server_agent.py  �� ʵʱ PART+FINAL ���
"""
import asyncio, websockets, json, os, time
from datetime import datetime
from vosk import Model, KaldiRecognizer

MODEL_PATH  = "model/vosk-model-cn-0.22"
SAMPLE_RATE = 16000
model = Model(MODEL_PATH)

PHOTO_DIR = "photos"; VIDEO_DIR = "video_frames"
os.makedirs(PHOTO_DIR, exist_ok=True); os.makedirs(VIDEO_DIR, exist_ok=True)

def save_photo(img: bytes):
    fn = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
    open(os.path.join(PHOTO_DIR, fn), "wb").write(img)
    print(f"[Srv] ? saved {fn}")

def save_frame(img: bytes):
    fn = datetime.now().strftime("%Y%m%d_%H%M%S_%f.jpg")
    open(os.path.join(VIDEO_DIR, fn), "wb").write(img)

async def ws_handler(ws):
    print("[Srv] ? new conn", ws.remote_address)
    rec = KaldiRecognizer(model, SAMPLE_RATE); rec.SetWords(True)

    last_photo_ts = 0
    video_on = False
    last_partial = ""              # ��ֹ�ظ���ӡ

    async for chunk in ws:
        # ---------- ��Ƶ ----------
        if isinstance(chunk, bytes):
            if rec.AcceptWaveform(chunk):          # һ�����
                result = json.loads(rec.Result())["text"].replace(" ", "")
                if result:
                    print(f"[FINAL] {result}")
                    # ---- ָ���ж����� FINAL ----
                    if "����" in result and time.time() - last_photo_ts > 3:
                        await ws.send(json.dumps({"cmd": "photo"}))
                        last_photo_ts = time.time()
                        print("[Srv] ? photo cmd")
                    if "¼����ͣ" in result or "ֹͣ¼��" in result:
                        if video_on:
                            await ws.send(json.dumps({"cmd": "video_stop"}))
                            video_on = False; print("[Srv] ? video_stop")
                    elif "¼��" in result and not video_on:
                        await ws.send(json.dumps({"cmd": "video_start"}))
                        video_on = True;  print("[Srv] ? video_start")
                last_partial = ""                  # ��� partial ����
            else:                                  # �м�̬ partial
                partial = json.loads(rec.PartialResult())["partial"].replace(" ", "")
                if partial and partial != last_partial:
                    print(f"[PART]  {partial}", end="\r", flush=True)
                    last_partial = partial

        # ---------- �ı�����Ƭ / ��Ƶ֡�� ----------
        else:
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                print("[Srv] txt:", chunk); continue
            tag = data.get("tag")
            if tag == "photo":
                save_photo(bytes.fromhex(data["payload_hex"]))
            elif tag == "video_frame":
                save_frame(bytes.fromhex(data["payload_hex"]))
            else:
                print("[Srv] txt:", data)

async def main():
    async with websockets.serve(ws_handler, "0.0.0.0", 6006,
                                ping_interval=20, max_size=None):
        print("[Srv] ? ws://0.0.0.0:6006/ws (vosk ready)")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
