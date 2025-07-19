# baidu_tts.py
# coding=utf-8
import sys
import json

import time
import io
IS_PY3 = sys.version_info.major == 3
if IS_PY3:
    from urllib.request import urlopen
    from urllib.request import Request
    from urllib.error import URLError
    from urllib.parse import urlencode
    from urllib.parse import quote_plus
else:
    import urllib2
    from urllib import quote_plus
    from urllib2 import urlopen
    from urllib2 import Request
    from urllib2 import URLError
    from urllib import urlencode

API_KEY = '2DZd4jAKx2317SdMKH9GciM0'
SECRET_KEY = 'PKk9suxXbkTmKlhNPFY853hOCxBLiaZg'

# TEXT = "��ӭʹ�ðٶ������ϳɡ�"

# ������ѡ��, �������⣺0Ϊ��С����1Ϊ��С�3Ϊ����ң��4Ϊ��ѾѾ��
# ��Ʒ���⣺5Ϊ��С����103Ϊ���׶䣬106Ϊ�Ȳ��ģ�110Ϊ��Сͯ��111Ϊ��С�ȣ�Ĭ��Ϊ��С�� 
PER = 4
# ���٣�ȡֵ0-15��Ĭ��Ϊ5������
SPD = 5
# ������ȡֵ0-15��Ĭ��Ϊ5�����
PIT = 5
# ������ȡֵ0-9��Ĭ��Ϊ5������
VOL = 5
# ���ص��ļ���ʽ, 3��mp3(default) 4�� pcm-16k 5�� pcm-8k 6. wav
AUE = 3

FORMATS = {3: "mp3", 4: "pcm", 5: "pcm", 6: "wav"}
FORMAT = FORMATS[AUE]

CUID = "123456PYTHON"

TTS_URL = 'http://tsn.baidu.com/text2audio'


class DemoError(Exception):
    pass


"""  TOKEN start """

TOKEN_URL = 'http://aip.baidubce.com/oauth/2.0/token'
SCOPE = 'audio_tts_post'  # �д�scope��ʾ��tts������û��������ҳ�ﹴѡ


def fetch_token():
    print("fetch token begin")
    params = {'grant_type': 'client_credentials',
              'client_id': API_KEY,
              'client_secret': SECRET_KEY}
    post_data = urlencode(params)
    if (IS_PY3):
        post_data = post_data.encode('utf-8')
    req = Request(TOKEN_URL, post_data)
    try:
        f = urlopen(req, timeout=5)
        result_str = f.read()
    except URLError as err:
        # print('token http response http code : ' + str(err.code))
        result_str = err.read()
    if (IS_PY3):
        result_str = result_str.decode()

    # print(result_str)
    result = json.loads(result_str)
    # print(result)
    if ('access_token' in result.keys() and 'scope' in result.keys()):
        if not SCOPE in result['scope'].split(' '):
            raise DemoError('scope is not correct')
        print('SUCCESS WITH TOKEN: %s ; EXPIRES IN SECONDS: %s' % (result['access_token'], result['expires_in']))
        return result['access_token']
    else:
        raise DemoError('MAYBE API_KEY or SECRET_KEY not correct: access_token or scope not found in token response')


"""  TOKEN end """

def baidu_tts_test(text,PER=0,SPD=10,PIT=PIT,VOL=VOL):
    token = fetch_token()
    tex = quote_plus(text)  # �˴�TEXT��Ҫ����urlencode
    # print(tex)
    params = {'tok': token, 'tex': tex, 'per': PER, 'spd': SPD, 'pit': PIT, 'vol': VOL, 'aue': AUE, 'cuid': CUID,
              'lan': 'zh', 'ctp': 1}  # lan ctp �̶�����

    data = urlencode(params)
    # print('test on Web Browser' + TTS_URL + '?' + data)

    req = Request(TTS_URL, data.encode('utf-8'))
    has_error = False
    try:
        f = urlopen(req)
        result_str = f.read()

        headers = dict((name.lower(), value) for name, value in f.headers.items())

        has_error = ('content-type' not in headers.keys() or headers['content-type'].find('audio/') < 0)
    except  URLError as err:
        # print('asr http response http code : ' + str(err.code))
        result_str = err.read()
        has_error = True

    save_file = "error.txt" if has_error else 'result.' + FORMAT
    with open(save_file, 'wb') as of:
        of.write(result_str)

    if has_error:
        if (IS_PY3):
            result_str = str(result_str, 'utf-8')
        # print("tts api  error:" + result_str)
    else:
            pass  # ���Ƴ���Ƶ���Ź��ܣ��������ļ�
    print("result saved as :" + save_file)
