from PIL import Image
from predict import prediction
from preprocess_v3 import gen_images

### begin networking by xmcp

import getpass
import requests
import time
import io
import random

ELECTIVE_XH = input('学号：')
ELECTIVE_PW = getpass.getpass('密码：')
DELAY_S_MIN = 1.5
DELAY_S_DELTA = 1.5

adapter = requests.adapters.HTTPAdapter(pool_connections=3, pool_maxsize=3, pool_block=True, max_retries=3)
s = requests.Session()
s.mount('http://elective.pku.edu.cn', adapter)
s.mount('https://elective.pku.edu.cn', adapter)

def login():
    print('login')
    res = s.post(
        'https://iaaa.pku.edu.cn/iaaa/oauthlogin.do',
        data={
            'appid': 'syllabus',
            'userName': ELECTIVE_XH,
            'password': ELECTIVE_PW,
            'randCode': '',
            'smsCode': '',
            'otpCode': '',
            'redirUrl': 'http://elective.pku.edu.cn:80/elective2008/ssoLogin.do'
        },
    )
    res.raise_for_status()
    json = res.json()
    assert json['success'], json
    token = json['token']

    res = s.get(
        'https://elective.pku.edu.cn/elective2008/ssoLogin.do',
        params={
            'rand': '%.10f'%random.random(),
            'token': token,
        },
    )
    res.raise_for_status()

def get_captcha():
    res = s.get(
        'https://elective.pku.edu.cn/elective2008/DrawServlet?Rand=114514',
        headers={
            'referer': 'https://elective.pku.edu.cn/elective2008/edu/pku/stu/elective/controller/supplement/SupplyCancel.do',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36',
            #'cookie': ELECTIVE_COOKIE,
        },
        timeout=(3,3),
    )
    res.raise_for_status()
    rawim = res.content
    if not rawim.startswith(b'GIF89a'):
        print(res.text)
        raise RuntimeError('bad captcha')

    return rawim

def check_captcha(captcha):
    res = s.post(
        'https://elective.pku.edu.cn/elective2008/edu/pku/stu/elective/controller/supplement/validate.do',
        headers={
            'referer': 'https://elective.pku.edu.cn/elective2008/edu/pku/stu/elective/controller/supplement/SupplyCancel.do',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36',
            #'cookie': ELECTIVE_COOKIE,
        },
        data={
            'xh': ELECTIVE_XH,
            'validCode': captcha,
        },
        timeout=(3,3),
    )
    res.raise_for_status()
    try:
        json = res.json()
    except Exception as e:
        if '异常刷新' in res.text:
            login()
            return check_captcha(captcha)
        else:
            print(res.text)
            raise

    if json['valid']!='2':
        return False
    else:
        return True

### end networking

def step():
    rawim = get_captcha()
    im = Image.open(io.BytesIO(rawim))
    ans = prediction(gen_images(im))
    succ = check_captcha(ans)

    serial = '%d-%d'%(1000*time.time(), random.random()*1000)
    with open('bootstrap_img_%s/%s=%s.gif'%('succ' if succ else 'fail', ans, serial), 'wb') as f:
        f.write(rawim)
    
    return succ, ans

if __name__ == '__main__':
    tot = 0
    totsucc = 0
    login()
    while True:
        tot += 1
        succ, ans = step()
        if succ:
            totsucc += 1
        print('pred: %s\tacc'%(ans), totsucc, '/', tot, '=', '%.3f'%(totsucc/tot))
        time.sleep(DELAY_S_MIN + random.random()*DELAY_S_DELTA)