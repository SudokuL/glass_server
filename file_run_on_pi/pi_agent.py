#!/usr/bin/env python3
# -*- coding: gbk -*-
import asyncio, json, websockets, pyaudio, os, traceback, cv2, numpy as np
# /usr/bin/python3 /home/arglasses/iot/pi_agent.py
AUDIO_HOST=os.getenv("AUDIO_HOST","127.0.0.1")
AUDIO_PORT=int(os.getenv("AUDIO_PORT",6006))
WS_URI=f"ws://{AUDIO_HOST}:{AUDIO_PORT}/ws"

CHUNK=1024; FORMAT=pyaudio.paInt16; CHANNELS=1; RATE=16000

async def async_ar(text):
    from ar import ar
    import asyncio

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, ar, text)

import socket, sys, time
def wait_tunnel(port=6006, timeout=10):
    t0=time.time()
    while time.time()-t0 < timeout:
        s=socket.socket(); ok=s.connect_ex(("127.0.0.1", port)); s.close()
        if ok==0: return
        time.sleep(1)
    sys.exit(f"[Pi] ? ���ض˿� {port} δ������SSH ���δ����")

wait_tunnel()

# ---------- ��Ƶ ----------
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

# ---------- ���� ----------
def _find_camera():
    """���ҿ��õ�����ͷ�豸"""
    print("[Pi] ���ڼ������ͷ�豸...")
    available_cameras = []
    
    for i in range(10):  # ���0-9���豸
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # �����Ƿ��ܶ�ȡ֡
            ret, frame = cap.read()
            if ret:
                # ��ȡ����ͷ��Ϣ
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"[Pi] �ҵ���������ͷ: /dev/video{i} ({width}x{height}, {fps}fps)")
                available_cameras.append(i)
                cap.release()
                return i  # ���ص�һ�����õ�����ͷ
            else:
                print(f"[Pi] /dev/video{i} ���ڵ��޷���ȡ֡")
        cap.release()
    
    if not available_cameras:
        print("[Pi] δ�ҵ��κο��õ�����ͷ�豸")
        print("[Pi] ����:")
        print("[Pi] 1. ����ͷ�Ƿ���ȷ����")
        print("[Pi] 2. �Ƿ����㹻��Ȩ�޷�������ͷ�豸")
        print("[Pi] 3. ��������: ls -la /dev/video*")
        print("[Pi] 4. ��������: lsusb (�鿴USB����ͷ)")
    
    return None

def _snap() -> bytes:
    # ���ҿ�������ͷ
    cam_index = _find_camera()
    if cam_index is None:
        raise RuntimeError("δ�ҵ���������ͷ�豸")
    
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened(): 
        raise RuntimeError(f"�޷�������ͷ /dev/video{cam_index}")
    
    try:
        # ���û����ֱ���
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # ���ͷֱ�������ȶ���
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   
        cap.set(cv2.CAP_PROP_FPS, 10)             # ��һ������֡��
        
        # ƽ����ع����
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # �����Զ��ع�
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)         # ���е��ع�ֵ
        cap.set(cv2.CAP_PROP_GAIN, 10)             # �ʵ�����
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)      # �������
        cap.set(cv2.CAP_PROP_CONTRAST, 1.0)        # ��׼�Աȶ�
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)           # �����Զ���ƽ��
        
        # ֱ��ʹ���ֶ��Խ�
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # �ر��Զ��Խ�
        
        print("[Pi] ����ͷ������ɣ���ʼ�ֶ��Խ�...")
        
        # �ֶ��Խ�����
        import time
        time.sleep(1.0)  # ��ʼ�ȶ�
        
        # �����ֶ��Խ� - ���ٲ��Ե����̵ȴ�
        focus_values = [80, 120]  # ֻ����2����öԽ���
        best_focus = 80  # Ĭ���еȶԽ�ֵ
        best_clarity = 0
        
        print("[Pi] ��ʼ���ٶԽ�ɨ��...")
        for focus_val in focus_values:
            cap.set(cv2.CAP_PROP_FOCUS, focus_val)
            time.sleep(0.3)  # ��һ�����ٵȴ�ʱ��
            
            # ֻ��ȡ1֡���ټ��
            ret, test_frame = cap.read()
            
            if ret:
                # ����ͼ�������ȣ�������˹���
                gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
                clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
                print(f"[Pi] �Խ�ֵ {focus_val}: ������ {clarity:.1f}")
                
                if clarity > best_clarity:
                    best_clarity = clarity
                    best_focus = focus_val
                    
                # ������ǰ������ֵ������������
                if clarity > 100:
                    print(f"[Pi] �����ȴ�꣬��ǰ����")
                    break
        
        # ������ѶԽ�ֵ
        print(f"[Pi] ��ѶԽ�ֵ: {best_focus}, ������: {best_clarity:.1f}")
        cap.set(cv2.CAP_PROP_FOCUS, best_focus)
        time.sleep(0.4)  # ��һ�����������ȶ�ʱ��
        
        print("[Pi] �ֶ��Խ���ɣ�׼������")
        
        # �������� - ������������
        best_frame = None
        best_frame_clarity = 0
        
        for shot in range(3):  # ���ٵ�3��
            ok, frm = cap.read()
            time.sleep(0.1)  # �������ռ��
            
            if ok:
                # ����������
                gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
                print(f"[Pi] ���� {shot+1}/3, ������: {clarity:.1f}")
                
                if clarity > best_frame_clarity:
                    best_frame_clarity = clarity
                    best_frame = frm.copy()
                    
                # ��������Ⱥܺã���ǰ��������
                if clarity > 120:
                    print(f"[Pi] ���������ȴ�꣬��ǰ����")
                    break
        
        if best_frame is None:
            raise RuntimeError("�޷�������ͷ��ȡͼ��")
        
        print(f"[Pi] ѡ����������Ƭ��������: {best_frame_clarity:.1f}")
        print(f"[Pi] ԭʼͼ��ߴ�: {best_frame.shape}")
        print(f"[Pi] ԭʼͼ������: {best_frame.dtype}")
        
        # ��ȫ�����κκ���ֱ�ӱ���
        print("[Pi] ֱ�ӱ��룬�޺���")
        
        # ʹ�ýϸߵ�JPEG����
        ok, buf = cv2.imencode(".jpg", best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok: 
            raise RuntimeError("ͼ�����ʧ��")
        return buf.tobytes()
    finally:
        cap.release()

async def take_a_photo(ws):
    try:
        img=await asyncio.get_running_loop().run_in_executor(None,_snap)
        await ws.send(json.dumps({"tag":"photo","payload_hex":img.hex()}))
        print("[Pi] ? photo sent",len(img))
    except Exception as e:
        print("[Pi] photo err",e)

# ======== ��Ƶ��Э�� ========
async def _video_loop(ws, fps=10):
    # ���ҿ�������ͷ
    cam_index = _find_camera()
    if cam_index is None:
        print("[Pi] ��Ƶ������: δ�ҵ���������ͷ�豸")
        return
    
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened(): 
        print(f"[Pi] ��Ƶ������: �޷�������ͷ /dev/video{cam_index}")
        return
    
    try:
        # ������Ƶ����������Խϵ͵ķֱ����Ա�֤�����ԣ�
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # ��Ƶ��ʹ�ýϵͷֱ���
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        
        interval = 1.0/fps
        consecutive_failures = 0
        max_failures = 10
        
        while True:
            ok, frm = cap.read()
            if not ok: 
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"[Pi] ��Ƶ������: ����{max_failures}�ζ�ȡʧ�ܣ�ֹͣ��Ƶ��")
                    break
                await asyncio.sleep(0.1)
                continue
            
            consecutive_failures = 0  # ����ʧ�ܼ���
            
            # ��Ƶ��ʹ�ýϸߵ�ѹ�����Լ��ٴ���
            ok, buf = cv2.imencode(".jpg", frm, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok: 
                continue
                
            try:
                await ws.send(json.dumps({"tag": "video_frame",
                                         "payload_hex": buf.tobytes().hex()}))
            except Exception as e:
                print(f"[Pi] ��Ƶ�����ʹ���: {e}")
                break
                
            await asyncio.sleep(interval)
    except Exception as e:
        print(f"[Pi] ��Ƶ���쳣: {e}")
    finally:
        cap.release()
        print("[Pi] ��Ƶ����ֹͣ")

video_task=None  # ȫ����������

async def recv_cmd(ws):
    global video_task

    async for msg in ws:
        data = json.loads(msg)

        # ͳһ���� text+audio response
        if data.get("tag") == "response":
            fn = data["filename"]
            question = data.get("question", "")
            answer   = data.get("text", "")
            # �����ʴ�ͬһ txt
            txt_fn = f"response_{fn}.txt"
            with open(txt_fn, "w", encoding="utf-8") as f:
                f.write(f"Q: {question}\nA: {answer}")
            print(f"[Pi] ? Saved Q&A: {txt_fn}")
            
            # ������Ƶ
            mp3_fn = f"response_{fn}.mp3"
            mp3_bytes = bytes.fromhex(data.get("payload_hex", ""))
            with open(mp3_fn, "wb") as f:
                f.write(mp3_bytes)
            print(f"[Pi] ? Saved audio: {mp3_fn}")
            asyncio.create_task(async_ar(answer))
            continue

        # ԭ�����������/��Ƶ��
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
            print("[Pi] δ֪��Ϣ:", data)



# ---------- ��ѭ�� ----------
async def main():
    while True:
        try:
            print("[Pi] ? connect",WS_URI)
            async with websockets.connect(WS_URI,ping_interval=20) as ws:
                await asyncio.gather(send_audio(ws),recv_cmd(ws))
        except Exception as e:
            print("[Pi] reconnect in 5s",e); await asyncio.sleep(5)

if __name__=="__main__":
    # ---------- ����ͷ��� ----------
    print("[Pi] ����ʱ�������ͷ...")
    cam_available = _find_camera() is not None
    if cam_available:
        print("[Pi] ? ����ͷ���ɹ�")
    else:
        print("[Pi] ? ����ͷ���ʧ�ܣ����պ���Ƶ���ܽ�������")
    
    asyncio.run(main())
