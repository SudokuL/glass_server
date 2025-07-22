#!/usr/bin/env python3
# -*- coding: gbk -*-
import asyncio, json, websockets, pyaudio, os, traceback, cv2
# /usr/bin/python3 /home/arglasses/iot/pi_agent.py
AUDIO_HOST=os.getenv("AUDIO_HOST","127.0.0.1")
AUDIO_PORT=int(os.getenv("AUDIO_PORT",6006))
WS_URI=f"ws://{AUDIO_HOST}:{AUDIO_PORT}/ws"

CHUNK=1024; FORMAT=pyaudio.paInt16; CHANNELS=1; RATE=16000

# ---------- ??? ----------
async def send_audio(ws):
    pa=pyaudio.PyAudio()
    st=pa.open(format=FORMAT,channels=CHANNELS,rate=RATE,
               input=True,frames_per_buffer=CHUNK)
    print("[Pi] ? start audio")
    try:
        while True:
            await ws.send(st.read(CHUNK,exception_on_overflow=False))
            await asyncio.sleep(0)
    except Exception as e:
        print("[Pi] audio err",e); traceback.print_exc()
    finally:
        st.stop_stream(); st.close(); pa.terminate()

# ---------- ???? ----------
def _snap() -> bytes:
    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): raise RuntimeError("cam open fail")
    
    ok,frm=cap.read(); cap.release()
    if not ok: raise RuntimeError("snap fail")
    ok,buf=cv2.imencode(".jpg",frm,[int(cv2.IMWRITE_JPEG_QUALITY),90])
    if not ok: raise RuntimeError("encode fail")
    return buf.tobytes()

async def take_a_photo(ws):
    try:
        img=await asyncio.get_running_loop().run_in_executor(None,_snap)
        await ws.send(json.dumps({"tag":"photo","payload_hex":img.hex()}))
        print("[Pi] ? photo sent",len(img))
    except Exception as e:
        print("[Pi] photo err",e)

# ======== ????????? ========
async def _video_loop(ws, fps=10):
    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): raise RuntimeError("cam open fail")
    try:
        interval=1.0/fps
        while True:
            ok,frm=cap.read()
            if not ok: continue
            ok,buf=cv2.imencode(".jpg",frm,[int(cv2.IMWRITE_JPEG_QUALITY),80])
            if not ok: continue
            await ws.send(json.dumps({"tag":"video_frame",
                                       "payload_hex":buf.tobytes().hex()}))
            await asyncio.sleep(interval)
    finally:
        cap.release()

video_task=None  # ???????????

async def recv_cmd(ws):
    global video_task

    async for msg in ws:
        data = json.loads(msg)

        # ?????? text+audio response
        if data.get("tag") == "response":
            fn = data["filename"]
            question = data.get("question", "")
            answer   = data.get("text", "")
            # ????????? txt
            txt_fn = f"response_{fn}.txt"
            with open(txt_fn, "w", encoding="utf-8") as f:
                f.write(f"Q: {question}\nA: {answer}")
            print(f"[Pi] ? Saved Q&A: {txt_fn}")
            # ???????
            mp3_fn = f"response_{fn}.mp3"
            mp3_bytes = bytes.fromhex(data.get("payload_hex", ""))
            with open(mp3_fn, "wb") as f:
                f.write(mp3_bytes)
            print(f"[Pi] ? Saved audio: {mp3_fn}")
            continue

        # ???????????????/?????
        cmd = data.get("cmd")
        if cmd == "photo":
            asyncio.create_task(take_a_photo(ws))
        elif cmd == "video_start" and video_task is None:
            video_task = asyncio.create_task(_video_loop(ws))
            print("[Pi] ?? start streaming")
        elif cmd == "video_stop" and video_task:
            video_task.cancel()
            video_task = None
            print("[Pi] ?? stop streaming")
        else:
            print("[Pi] ??????:", data)



# ---------- ????? ----------
async def main():
    while True:
        try:
            print("[Pi] ? connect",WS_URI)
            async with websockets.connect(WS_URI,ping_interval=20) as ws:
                await asyncio.gather(send_audio(ws),recv_cmd(ws))
        except Exception as e:
            print("[Pi] reconnect in 5s",e); await asyncio.sleep(5)

if __name__=="__main__":
    asyncio.run(main())
