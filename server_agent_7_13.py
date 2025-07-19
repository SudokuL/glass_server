#!/usr/bin/env python3  # ָ��Python������·��
# -*- coding: gbk -*-  # �����ļ�����ΪGBK
"""
server_agent.py
���ܣ�
1. ���� ws://0.0.0.0:6006/ws
2. ����������ݮ���ϴ��� 16 kHz / 16-bit / mono PCM ��Ƶ��
   - �� Vosk ʵʱ����ʶ�𣬴�ӡ partial / final ���
3. �ն����� `video` / `photo`����Զ������ݮ������Ƶ������
   - TODO: �� on_video_stream() / on_photo() ������������߼�
"""

# �����Ҫ�Ŀ�
import asyncio      # �첽IO�⣬���ڴ���������
import websockets   # WebSocket�⣬��������ͨ��
import json         # JSON����⣬�����������л��ͷ����л�
import sys          # ϵͳ�⣬���ڷ���ϵͳ��ع���
import os           # ����ϵͳ�ӿڿ⣬�����ļ���Ŀ¼����
import time         # ʱ�䴦��⣬����ʱ����ز���
from vosk import Model, KaldiRecognizer  # ����Vosk����ʶ��ģ�ͺ�ʶ����

# ==== Vosk ģ������ ====
MODEL_PATH = "model/vosk-model-cn-0.22"   # ��������ʶ��ģ��·�����ɸ�����Ҫ�޸�
SAMPLE_RATE = 16000                       # ���������ã���������ݮ�ɶ˱���һ��
model = Model(MODEL_PATH)                 # ��������ʶ��ģ��

AUDIO_OUT = "incoming_audio.raw"          # ԭʼ��Ƶ�����ļ��������ڽ����յ�����Ƶ���ݳ־û��洢

# ������ Զ�̹��ܻص����� ������������������������������������������������������������������������������������
def on_video_stream_start(ws):
    """��ݮ�ɿ�ʼ����Ƶ��ʱ�����á�
    
    Args:
        ws: WebSocket���Ӷ��󣬿���������ݮ��ͨ��
    """
    print("[Srv] ? <TODO ������յ���Ƶ��>")  # ��ӡ��ʾ��Ϣ����Ҫʵ����Ƶ�������߼�

def on_photo_received(image_bytes):
    """��ݮ�����ջش�ʱ���á�
    
    Args:
        image_bytes: ���յ���ͼƬ����������
    """
    print("[Srv] ? <TODO ����/������Ƭ> len=", len(image_bytes))  # ��ӡͼƬ��С����Ҫʵ����Ƭ����/�����߼�
# ������������������������������������������������������������������������������������������������

async def ws_handler(ws):          # WebSocket���Ӵ�������websockets��汾��14ʱֻ����һ��websocket����
    """��������ݮ�ɵ�WebSocket���ӡ�
    
    Args:
        ws: WebSocket���Ӷ���
    """
    print(f"[Srv] ? new conn {ws.remote_address}")  # ��ӡ�����ӵ�Զ�̵�ַ��Ϣ

    # ��ʼ������ʶ����
    rec = KaldiRecognizer(model, SAMPLE_RATE)  # ����Kaldi����ʶ����ʵ����ʹ��Ԥ���ص�ģ�ͺͲ�����
    rec.SetWords(True)                         # ���ôʼ���ʶ�𣬿ɻ�ȡ����ϸ��ʶ����

    # �����׼��������������ڽ����û�����
    async def stdin_task():
        """������׼������첽�������ڽ����û�������͸���ݮ�ɡ�"""
        loop = asyncio.get_running_loop()  # ��ȡ��ǰ�¼�ѭ��
        while True:  # ���������û�����
            # ���¼�ѭ�����첽��ȡ��׼���룬�����������߳�
            cmd = (await loop.run_in_executor(None, sys.stdin.readline)).strip()  # ��ȡ�û����벢ȥ����β�հ�
            if cmd in ("video", "photo"):  # �������Ч����
                await ws.send(json.dumps({"cmd": cmd}))  # ������תΪJSON��ʽ���͸���ݮ��
                print(f"[Srv] ? sent cmd: {cmd}")  # ��ӡ���͵�����
            elif cmd:  # �����������������
                print("valid cmds: video / photo")  # ��ʾ��Ч������

    # ������������׼�����������
    cmd_task = asyncio.create_task(stdin_task())  # ��stdin_task��Ϊ�첽��������

    # ����Ƶ�ļ����ڱ�����յ���ԭʼ��Ƶ����
    audio_file = open(AUDIO_OUT, "ab", buffering=0)  # �Զ�����׷��ģʽ���ļ����޻���

    try:
        # ѭ������WebSocket���յ�����Ϣ
        async for msg in ws:  # �첽�������յ�����Ϣ
            # ==========================================================
            # 1) ������ݮ�ɷ��͵���Ƶ��������
            # ==========================================================
            if isinstance(msg, bytes):  # �����Ϣ�Ƕ��������ݣ���Ƶ����
                audio_file.write(msg)   # ����Ƶ����д���ļ�����

                # ����Ƶ������������ʶ�������д���
                if rec.AcceptWaveform(msg):  # ���ʶ����ȷ������һ������������Ƭ��
                    res = json.loads(rec.Result())  # ��ȡ����ʶ����������JSON
                    print(f"[ASR] ? {res.get('text', '')}")  # ��ӡʶ������ı�
                else:  # �������Ƭ����δ����
                    res = json.loads(rec.PartialResult())  # ��ȡ����ʶ����
                    partial = res.get("partial")  # ��ȡ����ʶ���ı�
                    if partial:  # ����в���ʶ����
                        # ��ͬһ�и�����ʾ����ʶ����
                        print(f"\r[ASR] �� {partial}", end="", flush=True)

            # ==========================================================
            # 2) ������ݮ�ɷ��͵��ı���Ϣ��������Ƶ/��Ƭ�����Ԫ��Ϣ��
            # ==========================================================
            else:  # �����Ϣ���Ƕ��������ݣ���Ϊ�ı���Ϣ
                try:
                    data = json.loads(msg)  # ���Խ���JSON��ʽ����Ϣ
                except json.JSONDecodeError:  # �������ʧ�ܣ�������Ч��JSON��
                    print("[Srv] ? text:", msg)  # ֱ�Ӵ�ӡԭʼ�ı�
                    continue  # ����������һ����Ϣ

                # ������Ϣ��ǩ���ͽ��в�ͬ����
                tag = data.get("tag")  # ��ȡ��Ϣ��ǩ
                if tag == "video_stream_start":  # �������Ƶ����ʼ��֪ͨ
                    on_video_stream_start(ws)  # ������Ƶ��������
                elif tag == "photo":  # �������Ƭ����
                    # ��ʮ�������ַ���ת��Ϊ���������ݣ���������Ƭ������
                    on_photo_received(bytes.fromhex(data["payload_hex"]))
                else:  # �������͵���Ϣ
                    print("[Srv] txt:", data)  # ��ӡ��Ϣ����

    except websockets.ConnectionClosed:  # ����WebSocket���ӹر��쳣
        print("[Srv] ? Pi disconnected")  # ��ӡ���ӶϿ���Ϣ

    finally:  # �������������������쳣����ִ���������
        cmd_task.cancel()  # ȡ����׼�����������
        audio_file.close()  # �ر���Ƶ�ļ�
        # ����������յ�����ʶ����
        final_res = json.loads(rec.FinalResult()).get("text", "")  # ��ȡ����ʣ���ʶ����
        if final_res:  # ��������ս��
            print(f"\n[ASR] ? {final_res}")  # ��ӡ����ʶ����

async def main():
    """������������WebSocket��������"""
    # ����������WebSocket������
    async with websockets.serve(
        ws_handler,           # ���Ӵ�����
        "0.0.0.0",           # ������������ӿ�
        6006,                # �����˿�
        ping_interval=20,    # ������������룩���������ӻ�Ծ
        max_size=None        # ��������Ϣ��С��������մ��Ͷ���������
    ):
        # ��ӡ������������Ϣ
        print("[Srv] ? ws://0.0.0.0:6006/ws UP  (Vosk model loaded)")
        # ���ַ��������У�ֱ�������ֶ���ֹ
        await asyncio.Future()  # ����һ��������ɵ�Future��ʹ�����������

# ������ڵ�
if __name__ == "__main__":  # ���ű�ֱ������ʱִ��
    asyncio.run(main())  # �����첽������
