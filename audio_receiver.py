#!/usr/bin/env python3
# coding=utf-8
import socket
import threading
import json
from vosk import Model, KaldiRecognizer

HOST = "0.0.0.0"
PORT = 6006
RATE = 16000          # 与树莓派保持一致
SAMPLE_WIDTH = 2      # 16-bit
CHUNK = 1024

model = Model("/root/autodl-tmp/.autodl/iot/model/vosk-model-cn-0.22")  # 换成自己的模型
print("[INIT] Model loaded")

def handle(conn, addr):
    print(f"[CONN] {addr} connected")
    rec = KaldiRecognizer(model, RATE)
    rec.SetWords(True)

    buf = b""
    MIN_PACKET = 4000            # 0.125 s @ 16 kHz 16-bit mono
    bytes_total = 0

    try:
        while True:
            chunk = conn.recv(CHUNK * SAMPLE_WIDTH)
            if not chunk:
                break
            buf += chunk
            bytes_total += len(chunk)

            # 每收到足够数据就切一段送给识别器
            while len(buf) >= MIN_PACKET:
                part, buf = buf[:MIN_PACKET], buf[MIN_PACKET:]

                if rec.AcceptWaveform(part):
                    # 本句话结束
                    final_txt = json.loads(rec.Result())["text"]
                    if final_txt:
                        final_txt = final_txt.replace(" ", "")
                        print("[FINAL]", final_txt)
                    rec.Reset()                 # 关键：重置识别器
                else:
                    p = json.loads(rec.PartialResult())["partial"]
                    if p:
                        print("[PART]", p)

    finally:
        # 处理残余缓冲
        if buf:
            rec.AcceptWaveform(buf)
            tail = json.loads(rec.FinalResult())["text"]
            if tail:
                tail = tail.replace(" ", "")
                print("[FINAL]", tail)

        print(f"[CONN] {addr} closed, bytes={bytes_total}")
        conn.close()

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[LISTEN] {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    main()
