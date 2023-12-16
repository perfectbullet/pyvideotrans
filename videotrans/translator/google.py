# -*- coding: utf-8 -*-
import os
import re
import time
import urllib

import requests
from gradio_client import Client
from videotrans.configure import config
from videotrans.util import tools

gx_translate_url = "http://127.0.0.1:7860/"
client = Client(gx_translate_url)


def googletrans(text, src, dest, *, set_p=True):
    '''
    替换原来翻译功能为本地的
    gradio-3.50.2 gradio-client-0.6.1
    参考文档是：  https://www.gradio.app/3.50.2/guides/quickstart
    '''
    print('使用翻译gx_translate_url: ', gx_translate_url)
    try:
        result = client.predict(
            text,
            api_name="/translate"
        )
        print('使用翻译gx_translate: {} ------> {}'.format(text, result))
        return result
    except Exception as e:
        msg = f"error 翻译失败: 请确认 {gx_translate_url} 是否正常"
        raise e
        # return msg


def googletrans_bak(text, src, dest, *, set_p=True):
    """
    原来翻译功能备份
    """
    url = f"https://translate.google.com/m?sl={urllib.parse.quote(src)}&tl={urllib.parse.quote(dest)}&hl={urllib.parse.quote(dest)}&q={urllib.parse.quote(text)}"
    print('googletrans url {}'.format(url))
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    proxies = None
    serv = tools.set_proxy()
    if serv:
        proxies = {
            'http://': serv,
            'https://': serv
        }
    nums = 0
    msg = f"[error]google 翻译失败:{text=}"
    while nums < 2:
        nums += 1
        try:
            response = requests.get(url, proxies=proxies, headers=headers, timeout=40)
            print(f"google translate code={response.status_code}")
            if response.status_code != 200:
                msg = f"[error] google翻译失败 status_code={response.status_code}"
                time.sleep(3)
                continue

            re_result = re.findall(
                r'(?s)class="(?:t0|result-container)">(.*?)<', response.text)
            if len(re_result) < 1:
                msg = '[error]google翻译失败了'
                time.sleep(3)
                continue
            return re_result[0]
        except Exception as e:
            msg = f"[error]google 翻译失败{serv=}:请确认能连接到google" + str(e)
            time.sleep(3)
    return msg
