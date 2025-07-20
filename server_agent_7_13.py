#!/usr/bin/env python3  # 指定Python解释器路径
# -*- coding: gbk -*-  # 设置文件编码为GBK
"""
server_agent.py
功能：
1. 监听 ws://0.0.0.0:6006/ws
2. 持续接收树莓派上传的 16 kHz / 16-bit / mono PCM 音频流
   - 用 Vosk 实时语音识别，打印 partial / final 结果
3. 终端输入 `video` / `photo`，可远程让树莓派推视频或拍照
   - TODO: 在 on_video_stream() / on_photo() 填入后续处理逻辑
"""

# 导入必要的库
import asyncio      # 异步IO库，用于处理并发操作
import websockets   # WebSocket库，用于网络通信
import json         # JSON处理库，用于数据序列化和反序列化
import sys          # 系统库，用于访问系统相关功能
import os           # 操作系统接口库，用于文件和目录操作
import time         # 时间处理库，用于时间相关操作
from vosk import Model, KaldiRecognizer  # 导入Vosk语音识别模型和识别器

# ==== Vosk 模型配置 ====
MODEL_PATH = "model/vosk-model-cn-0.22"   # 中文语音识别模型路径，可根据需要修改
SAMPLE_RATE = 16000                       # 采样率设置，必须与树莓派端保持一致
model = Model(MODEL_PATH)                 # 加载语音识别模型

AUDIO_OUT = "incoming_audio.raw"          # 原始音频保存文件名，用于将接收到的音频数据持久化存储

# ——— 远程功能回调函数 ——————————————————————————————————————————
def on_video_stream_start(ws):
    """树莓派开始推视频流时将调用。
    
    Args:
        ws: WebSocket连接对象，可用于与树莓派通信
    """
    print("[Srv] ? <TODO 处理接收的视频流>")  # 打印提示信息，需要实现视频流处理逻辑

def on_photo_received(image_bytes):
    """树莓派拍照回传时调用。
    
    Args:
        image_bytes: 接收到的图片二进制数据
    """
    print("[Srv] ? <TODO 保存/处理照片> len=", len(image_bytes))  # 打印图片大小，需要实现照片保存/处理逻辑
# ————————————————————————————————————————————————

async def ws_handler(ws):          # WebSocket连接处理函数，websockets库版本≥14时只接收一个websocket参数
    """处理与树莓派的WebSocket连接。
    
    Args:
        ws: WebSocket连接对象
    """
    print(f"[Srv] ? new conn {ws.remote_address}")  # 打印新连接的远程地址信息

    # 初始化语音识别器
    rec = KaldiRecognizer(model, SAMPLE_RATE)  # 创建Kaldi语音识别器实例，使用预加载的模型和采样率
    rec.SetWords(True)                         # 启用词级别识别，可获取更详细的识别结果

    # 定义标准输入监听任务，用于接收用户命令
    async def stdin_task():
        """监听标准输入的异步任务，用于接收用户命令并发送给树莓派。"""
        loop = asyncio.get_running_loop()  # 获取当前事件循环
        while True:  # 持续监听用户输入
            # 在事件循环中异步读取标准输入，避免阻塞主线程
            cmd = (await loop.run_in_executor(None, sys.stdin.readline)).strip()  # 读取用户输入并去除首尾空白
            if cmd in ("video", "photo"):  # 如果是有效命令
                await ws.send(json.dumps({"cmd": cmd}))  # 将命令转为JSON格式发送给树莓派
                print(f"[Srv] ? sent cmd: {cmd}")  # 打印发送的命令
            elif cmd:  # 如果输入了其他命令
                print("valid cmds: video / photo")  # 提示有效的命令

    # 创建并启动标准输入监听任务
    cmd_task = asyncio.create_task(stdin_task())  # 将stdin_task作为异步任务启动

    # 打开音频文件用于保存接收到的原始音频数据
    audio_file = open(AUDIO_OUT, "ab", buffering=0)  # 以二进制追加模式打开文件，无缓冲

    try:
        # 循环处理WebSocket接收到的消息
        async for msg in ws:  # 异步迭代接收到的消息
            # ==========================================================
            # 1) 处理树莓派发送的音频二进制流
            # ==========================================================
            if isinstance(msg, bytes):  # 如果消息是二进制数据（音频流）
                audio_file.write(msg)   # 将音频数据写入文件保存

                # 将音频数据送入语音识别器进行处理
                if rec.AcceptWaveform(msg):  # 如果识别器确认这是一个完整的语音片段
                    res = json.loads(rec.Result())  # 获取最终识别结果并解析JSON
                    print(f"[ASR] ? {res.get('text', '')}")  # 打印识别出的文本
                else:  # 如果语音片段尚未结束
                    res = json.loads(rec.PartialResult())  # 获取部分识别结果
                    partial = res.get("partial")  # 提取部分识别文本
                    if partial:  # 如果有部分识别结果
                        # 在同一行更新显示部分识别结果
                        print(f"\r[ASR] … {partial}", end="", flush=True)

            # ==========================================================
            # 2) 处理树莓派发送的文本消息（例如视频/照片传输的元信息）
            # ==========================================================
            else:  # 如果消息不是二进制数据，则为文本消息
                try:
                    data = json.loads(msg)  # 尝试解析JSON格式的消息
                except json.JSONDecodeError:  # 如果解析失败（不是有效的JSON）
                    print("[Srv] ? text:", msg)  # 直接打印原始文本
                    continue  # 继续处理下一条消息

                # 根据消息标签类型进行不同处理
                tag = data.get("tag")  # 获取消息标签
                if tag == "video_stream_start":  # 如果是视频流开始的通知
                    on_video_stream_start(ws)  # 调用视频流处理函数
                elif tag == "photo":  # 如果是照片数据
                    # 将十六进制字符串转换为二进制数据，并传给照片处理函数
                    on_photo_received(bytes.fromhex(data["payload_hex"]))
                else:  # 其他类型的消息
                    print("[Srv] txt:", data)  # 打印消息内容

    except websockets.ConnectionClosed:  # 捕获WebSocket连接关闭异常
        print("[Srv] ? Pi disconnected")  # 打印连接断开信息

    finally:  # 无论是正常结束还是异常，都执行清理操作
        cmd_task.cancel()  # 取消标准输入监听任务
        audio_file.close()  # 关闭音频文件
        # 处理并输出最终的语音识别结果
        final_res = json.loads(rec.FinalResult()).get("text", "")  # 获取最终剩余的识别结果
        if final_res:  # 如果有最终结果
            print(f"\n[ASR] ? {final_res}")  # 打印最终识别结果

async def main():
    """主函数，启动WebSocket服务器。"""
    # 创建并启动WebSocket服务器
    async with websockets.serve(
        ws_handler,           # 连接处理函数
        "0.0.0.0",           # 监听所有网络接口
        6006,                # 监听端口
        ping_interval=20,    # 心跳包间隔（秒），保持连接活跃
        max_size=None        # 不限制消息大小，允许接收大型二进制数据
    ):
        # 打印服务器启动信息
        print("[Srv] ? ws://0.0.0.0:6006/ws UP  (Vosk model loaded)")
        # 保持服务器运行，直到程序被手动终止
        await asyncio.Future()  # 创建一个永不完成的Future，使程序持续运行

# 程序入口点
if __name__ == "__main__":  # 当脚本直接运行时执行
    asyncio.run(main())  # 启动异步主函数
