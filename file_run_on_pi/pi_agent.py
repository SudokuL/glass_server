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
    sys.exit(f"[Pi] ? 本地端口 {port} 未监听，SSH 隧道未启动")

wait_tunnel()

# ---------- 音频 ----------
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

# ---------- 摄像 ----------
def _find_camera():
    """查找可用的摄像头设备"""
    print("[Pi] 正在检测摄像头设备...")
    available_cameras = []
    
    for i in range(10):  # 检查0-9号设备
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # 测试是否能读取帧
            ret, frame = cap.read()
            if ret:
                # 获取摄像头信息
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"[Pi] 找到可用摄像头: /dev/video{i} ({width}x{height}, {fps}fps)")
                available_cameras.append(i)
                cap.release()
                return i  # 返回第一个可用的摄像头
            else:
                print(f"[Pi] /dev/video{i} 存在但无法读取帧")
        cap.release()
    
    if not available_cameras:
        print("[Pi] 未找到任何可用的摄像头设备")
        print("[Pi] 请检查:")
        print("[Pi] 1. 摄像头是否正确连接")
        print("[Pi] 2. 是否有足够的权限访问摄像头设备")
        print("[Pi] 3. 尝试运行: ls -la /dev/video*")
        print("[Pi] 4. 尝试运行: lsusb (查看USB摄像头)")
    
    return None

def _snap() -> bytes:
    # 查找可用摄像头
    cam_index = _find_camera()
    if cam_index is None:
        raise RuntimeError("未找到可用摄像头设备")
    
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened(): 
        raise RuntimeError(f"无法打开摄像头 /dev/video{cam_index}")
    
    try:
        # 设置基本分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # 降低分辨率提高稳定性
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   
        cap.set(cv2.CAP_PROP_FPS, 10)             # 进一步降低帧率
        
        # 平衡的曝光控制
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 部分自动曝光
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)         # 适中的曝光值
        cap.set(cv2.CAP_PROP_GAIN, 10)             # 适当增益
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)      # 提高亮度
        cap.set(cv2.CAP_PROP_CONTRAST, 1.0)        # 标准对比度
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)           # 启用自动白平衡
        
        # 直接使用手动对焦
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 关闭自动对焦
        
        print("[Pi] 摄像头设置完成，开始手动对焦...")
        
        # 手动对焦策略
        import time
        time.sleep(1.0)  # 初始稳定
        
        # 极速手动对焦 - 最少测试点和最短等待
        focus_values = [80, 120]  # 只测试2个最常用对焦点
        best_focus = 80  # 默认中等对焦值
        best_clarity = 0
        
        print("[Pi] 开始极速对焦扫描...")
        for focus_val in focus_values:
            cap.set(cv2.CAP_PROP_FOCUS, focus_val)
            time.sleep(0.3)  # 进一步减少等待时间
            
            # 只读取1帧快速检测
            ret, test_frame = cap.read()
            
            if ret:
                # 计算图像清晰度（拉普拉斯方差）
                gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
                clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
                print(f"[Pi] 对焦值 {focus_val}: 清晰度 {clarity:.1f}")
                
                if clarity > best_clarity:
                    best_clarity = clarity
                    best_focus = focus_val
                    
                # 降低提前结束阈值，更容易满足
                if clarity > 100:
                    print(f"[Pi] 清晰度达标，提前结束")
                    break
        
        # 设置最佳对焦值
        print(f"[Pi] 最佳对焦值: {best_focus}, 清晰度: {best_clarity:.1f}")
        cap.set(cv2.CAP_PROP_FOCUS, best_focus)
        time.sleep(0.4)  # 进一步减少最终稳定时间
        
        print("[Pi] 手动对焦完成，准备拍照")
        
        # 极速拍照 - 减少拍照数量
        best_frame = None
        best_frame_clarity = 0
        
        for shot in range(3):  # 减少到3张
            ok, frm = cap.read()
            time.sleep(0.1)  # 减少拍照间隔
            
            if ok:
                # 计算清晰度
                gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
                print(f"[Pi] 拍照 {shot+1}/3, 清晰度: {clarity:.1f}")
                
                if clarity > best_frame_clarity:
                    best_frame_clarity = clarity
                    best_frame = frm.copy()
                    
                # 如果清晰度很好，提前结束拍照
                if clarity > 120:
                    print(f"[Pi] 拍照清晰度达标，提前结束")
                    break
        
        if best_frame is None:
            raise RuntimeError("无法从摄像头读取图像")
        
        print(f"[Pi] 选择最清晰照片，清晰度: {best_frame_clarity:.1f}")
        print(f"[Pi] 原始图像尺寸: {best_frame.shape}")
        print(f"[Pi] 原始图像类型: {best_frame.dtype}")
        
        # 完全不做任何后处理，直接编码
        print("[Pi] 直接编码，无后处理")
        
        # 使用较高的JPEG质量
        ok, buf = cv2.imencode(".jpg", best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok: 
            raise RuntimeError("图像编码失败")
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

# ======== 视频流协程 ========
async def _video_loop(ws, fps=10):
    # 查找可用摄像头
    cam_index = _find_camera()
    if cam_index is None:
        print("[Pi] 视频流错误: 未找到可用摄像头设备")
        return
    
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened(): 
        print(f"[Pi] 视频流错误: 无法打开摄像头 /dev/video{cam_index}")
        return
    
    try:
        # 设置视频流参数（相对较低的分辨率以保证流畅性）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # 视频流使用较低分辨率
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
                    print(f"[Pi] 视频流错误: 连续{max_failures}次读取失败，停止视频流")
                    break
                await asyncio.sleep(0.1)
                continue
            
            consecutive_failures = 0  # 重置失败计数
            
            # 视频流使用较高的压缩率以减少带宽
            ok, buf = cv2.imencode(".jpg", frm, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok: 
                continue
                
            try:
                await ws.send(json.dumps({"tag": "video_frame",
                                         "payload_hex": buf.tobytes().hex()}))
            except Exception as e:
                print(f"[Pi] 视频流发送错误: {e}")
                break
                
            await asyncio.sleep(interval)
    except Exception as e:
        print(f"[Pi] 视频流异常: {e}")
    finally:
        cap.release()
        print("[Pi] 视频流已停止")

video_task=None  # 全局任务引用

async def recv_cmd(ws):
    global video_task

    async for msg in ws:
        data = json.loads(msg)

        # 统一处理 text+audio response
        if data.get("tag") == "response":
            fn = data["filename"]
            question = data.get("question", "")
            answer   = data.get("text", "")
            # 保存问答到同一 txt
            txt_fn = f"response_{fn}.txt"
            with open(txt_fn, "w", encoding="utf-8") as f:
                f.write(f"Q: {question}\nA: {answer}")
            print(f"[Pi] ? Saved Q&A: {txt_fn}")
            
            # 保存音频
            mp3_fn = f"response_{fn}.mp3"
            mp3_bytes = bytes.fromhex(data.get("payload_hex", ""))
            with open(mp3_fn, "wb") as f:
                f.write(mp3_bytes)
            print(f"[Pi] ? Saved audio: {mp3_fn}")
            asyncio.create_task(async_ar(answer))
            continue

        # 原有命令处理（拍照/视频）
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
            print("[Pi] 未知消息:", data)



# ---------- 主循环 ----------
async def main():
    while True:
        try:
            print("[Pi] ? connect",WS_URI)
            async with websockets.connect(WS_URI,ping_interval=20) as ws:
                await asyncio.gather(send_audio(ws),recv_cmd(ws))
        except Exception as e:
            print("[Pi] reconnect in 5s",e); await asyncio.sleep(5)

if __name__=="__main__":
    # ---------- 摄像头检测 ----------
    print("[Pi] 启动时检测摄像头...")
    cam_available = _find_camera() is not None
    if cam_available:
        print("[Pi] ? 摄像头检测成功")
    else:
        print("[Pi] ? 摄像头检测失败，拍照和视频功能将不可用")
    
    asyncio.run(main())
