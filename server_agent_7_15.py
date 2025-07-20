#!/usr/bin/env python3
# -*- coding: gbk -*-
"""
server_agent.py  — 实时 PART+FINAL 输出
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
    last_partial = ""              # 防止重复打印

    async for chunk in ws:
        # ---------- 音频 ----------
        if isinstance(chunk, bytes):
            if rec.AcceptWaveform(chunk):          # 一句完成
                result = json.loads(rec.Result())["text"].replace(" ", "")
                if result:
                    print(f"[FINAL] {result}")
                    # ---- 指令判定仅用 FINAL ----
                    if "拍照" in result and time.time() - last_photo_ts > 3:
                        await ws.send(json.dumps({"cmd": "photo"}))
                        last_photo_ts = time.time()
                        print("[Srv] ? photo cmd")
                    if "录像暂停" in result or "停止录像" in result:
                        if video_on:
                            await ws.send(json.dumps({"cmd": "video_stop"}))
                            video_on = False; print("[Srv] ? video_stop")
                    elif "录像" in result and not video_on:
                        await ws.send(json.dumps({"cmd": "video_start"}))
                        video_on = True;  print("[Srv] ? video_start")
                last_partial = ""                  # 清空 partial 缓存
            else:                                  # 中间态 partial
                partial = json.loads(rec.PartialResult())["partial"].replace(" ", "")
                if partial and partial != last_partial:
                    print(f"[PART]  {partial}", end="\r", flush=True)
                    last_partial = partial

        # ---------- 文本（照片 / 视频帧） ----------
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
