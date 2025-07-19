# -*- coding: gbk -*-
"""
���� Tuya /v1.0/aispeech/voice/push���������·����豸����
ǰ�᣺����Ŀ�Ѷ��ġ�Text-to-Speech Service�����豸������֧�� TTS
"""
import time, hmac, hashlib, requests, json, os

CID  = "sgmpjw8pkg5kk7w7vhrg"                         # Access ID
CKEY = "e4e609f56a384c5cbd805a604c19a958"            # Access Secret
ENDP = "https://openapi.tuyacn.com"                  # ������Ŀ��������ѡ
DEVICE_ID = "xxxxxxxxxxxxxxxxxxxx"                  # ���Լ��ں�̨�鵽�� 20 λ ID

def _sign(msg: str) -> str:
    """HMAC-SHA256 & upper-hex"""
    return hmac.new(CKEY.encode(), msg.encode(),
                    hashlib.sha256).hexdigest().upper()

def get_token() -> str:
    t = str(int(time.time()*1000))
    headers = {
        "client_id": CID,
        "sign": _sign(CID + t),
        "t": t,
        "sign_method": "HMAC-SHA256"
    }
    r = requests.get(f"{ENDP}/v1.0/token?grant_type=1",
                     headers=headers, timeout=5)
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"Token error: {data}")
    return data["result"]["access_token"]

def push_tts(text: str):
    token = get_token()
    t = str(int(time.time()*1000))
    headers = {
        "client_id": CID,
        "access_token": token,
        "sign": _sign(CID + token + t),
        "t": t,
        "sign_method": "HMAC-SHA256",
        "Content-Type": "application/json"
    }
    body = {"tts": text, "deviceId": DEVICE_ID}
    r = requests.post(f"{ENDP}/v1.0/aispeech/voice/push",
                      headers=headers,
                      data=json.dumps(body, separators=(",", ":")),
                      timeout=5)
    print(r.json())

if __name__ == "__main__":
    push_tts("������磬����Ϳѻ")
