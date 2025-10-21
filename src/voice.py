import asyncio
import contextvars
import functools
import io
import os
import sys
import wave
import threading
from traceback import print_exc

import markdown
import numpy as np
import streamlit as st
# import speech_recognition as sr
from bs4 import BeautifulSoup
# import sounddevice as sd

from .utils import get_base_dir, print_info, split_str_length, tran_sync_to_async
from .config import *
import requests
import argparse
import time


def _get_baidu_access_token(token_name="token"):
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}

    # return "?cuid=TryLoess2624690754" + "&access_token=" + str(requests.post(url, params=params).json().get("access_token"))
    params = {
        "cuid": "TryLoess2624680754",
        token_name: str(requests.post(url, params=params).json().get("access_token"))
    }
    return params

def get_result_baidu(audio: bytes):
    """audio是raw格式"""
    access_token = _get_baidu_access_token()
    cloud_url = "http://vop.baidu.com/server_api"
    header = {"Content-Type": "audio/wav;rate=16000"}
    with open("test.wav", "wb") as f:
        f.write(audio)
    response = requests.post(cloud_url, headers=header, data=audio, params=access_token)
    return "".join(response.json()["result"])

async def m_receive(recognizer, source):
    if int(sys.version.split(".")[1]) < 9:
        loop = asyncio.get_running_loop()  # 先设置一个running_loop
        ctx = contextvars.copy_context()
        func_call = functools.partial(ctx.run, recognizer.listen, source, timeout=5, phrase_time_limit=20)
        task1 = loop.run_in_executor(None, func_call)  # 返回一个协程对象
    else:  # 3.9以上才有to_thread，3.8是没有这个玩意的
        task1 = asyncio.to_thread(recognizer.listen, source, timeout=5, phrase_time_limit=20)
    task2 = asyncio.create_task(async_show())
    res = await asyncio.gather(task1, task2)  # 这里统一await
    return res  # 这里是[task1返回值, task2返回值]

async def async_show():
    window = st.empty()
    window.warning("请稍后")
    await asyncio.sleep(0.3)
    window.empty()
    window.info("请开始讲话……")
    return window


# def voice_main_write():
#     """进行语音转写"""
#     recognizer = sr.Recognizer()
#     with sr.Microphone(sample_rate=16000) as source:
#         try:
#             audio_data, temp_window = asyncio.run(m_receive(recognizer, source))
#             audio_bytes = audio_data.get_wav_data()
#
#             # 使用wave模块读取音频数据
#             with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
#                 # 获取音频参数
#                 n_channels = wf.getnchannels()
#                 sample_width = wf.getsampwidth()
#                 framerate = wf.getframerate()
#                 n_frames = wf.getnframes()
#                 # 读取原始音频数据
#                 audio_data = wf.readframes(n_frames)
#
#             # 将音频数据转换为numpy数组进行处理
#             audio_array = np.frombuffer(audio_data, dtype=np.int16)
#             # 增加音量（乘以1.5相当于增加约3.5dB，可以根据需要调整）
#             louder_audio_array = np.clip(audio_array * 1.5, -32768, 32767).astype(np.int16)
#
#             # 将处理后的数据写入新的BytesIO对象
#             buffer = io.BytesIO()
#             with wave.open(buffer, 'wb') as wf:
#                 wf.setnchannels(n_channels)
#                 wf.setsampwidth(sample_width)
#                 wf.setframerate(framerate)
#                 wf.writeframes(louder_audio_array.tobytes())
#
#             buffer.seek(0)
#
#             audio_path = get_base_dir() + "/test.wav"
#             with open(audio_path, "wb") as f:
#                 f.write(audio_bytes)# buffer.read())
#             temp_window.empty()
#             temp_window.info("识别中，请稍候...")
#             command = get_result_baidu(buffer.read())
#             temp_window.empty()
#             return command
#
#         except sr.WaitTimeoutError:
#             st.error("录音超时，请重试。")
#         except sr.UnknownValueError:
#             st.error("无法识别语音，请重试。")
#         except sr.RequestError as e:
#             st.error(f"语音识别服务出错: {e}")
async def show_info(window, base_content, times=5):
    remain = times
    for _ in range(times):
        window.info(base_content + f"(剩余{remain}秒)")
        await asyncio.sleep(1)
        window.empty()
        remain -= 1

def _voice_write(duration=5):
    sample_rate = 16000
    print_info(duration)
    audio_array = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()  # 等待录音完成

    # 增加音量
    louder_audio_array = np.clip(audio_array * 1.5, -32768, 32767).astype(np.int16)

    # 将数据写入BytesIO对象以便发送到API
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16位 = 2字节
        wf.setframerate(sample_rate)
        wf.writeframes(louder_audio_array.tobytes())

    buffer.seek(0)

    # 保存测试用音频文件
    audio_path = get_base_dir() + "/test.wav"
    with open(audio_path, "wb") as f:
        f.write(buffer.getvalue())
    time.sleep(1.3)
    command = get_result_baidu(buffer.read())
    return command

async def voice_show(window, duration=5):
    tasks = [
        asyncio.create_task(tran_sync_to_async(_voice_write, duration)),
        asyncio.create_task(show_info(window, "请开始讲话", duration))
    ]
    # res = await asyncio.gather(*tasks)
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    first_task = list(done)[0]
    first_result = first_task.result()
    if first_result is None:
        window.empty()
        window.warning("正在语音识别，请稍后……")

    # 继续等待其他任务
    if pending:
        await asyncio.gather(*pending)

    # 收集所有结果
    all_results = [task.result() for task in tasks]
    return all_results[0]


def voice_main_write():
    """进行语音转写"""
    # 采样参数
    duration = 5  # 录音时长(秒)
    try:
        # 显示录音提示
        temp_window = st.empty()
        temp_window.warning("请稍后")
        time.sleep(0.5)
        temp_window.empty()
        res = asyncio.run(voice_show(temp_window, duration))
        print_info(res)
        temp_window.empty()

        return res
    except Exception as e:
        st.error(f"录音出错: {e}")
        print_exc()
        return None

# def _voice_main_return(tex, file_name=None):
#     """tex结果要小于60"""
#     access_token = _get_baidu_access_token()
#     cloud_url = "http://tsn.baidu.com/text2audio"
#     ex_params = {
#         "tex": tex,
#         "lan": "zh",
#         "cuid": access_token["cuid"],
#         "ctp": 1,
#         "tok": access_token["token"],
#         "per": 4132,
#         "audio_ctrl": {"sampling_rate": 16000},
#         "aue": 6,  # 返回wav
#         "vol": 6,  # 设置音量
#         "spd": 4,  # 语速
#
#     }
#     response = requests.post(cloud_url, params=ex_params)
#     print_info(response.status_code)
#     print_info(response.headers)
#     if file_name is not None:
#         if not os.path.exists(get_base_dir() + f"/data/voice"):
#             os.mkdir(get_base_dir() + f"/data/voice")
#         with open(get_base_dir() + f"/data/voice/{file_name}", "wb") as f:
#             f.write(response.content)
#         print_info(f"输出音频已保存至data/voice/{file_name}")
#     return response.content

def _voice_main_return(tex, type_):
    if type_ == "普通话":
        spkid = 0
        lanid = 0
    elif type_ == "闽南语":
        spkid = 10
        lanid = 3
    else:
        # 尽量以闽南语为默认，让闽南用户只用点击识别即可自动发音
        spkid = 10
        lanid = 3
    url = f"https://tc.talentedsoft.com:58120/ajax/proxy_tts?userid=yhy&spkid={spkid}&lanid={lanid}&token=1234&content={tex}&speed=1&volume=1"
    headers = {
    "accept": "*/*",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "connection": "keep-alive",
    "host": "tc.talentedsoft.com:58120",
    "referer": "https://tc.talentedsoft.com:58120/speech_synthesis",
    "sec-ch-ua": '"Microsoft Edge";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0"
}
    response = requests.get(url, headers=headers, timeout=300)
    print_info("访问返回代码", response.status_code)
    print_info("输入的原始内容为：", tex[:100] + ("..." if len(tex) > 100 else ""))
    return response.content

def _merge(voice_dir=get_base_dir() + "/data/voice", dst_path=get_base_dir() + "/data/voice/output_all.wav"):
    print_info("开始合并音频文件...")
    file_dir = [f for f in os.listdir(voice_dir) if f.endswith(".wav")]
    wav_files = sorted(file_dir, key=lambda x: int(x.split("_")[-1].split(".")[0]))  # 按照数字顺序排序,从小到大
    frames = []
    params = None
    for idx, file in enumerate(wav_files):
        with wave.open(os.path.join(voice_dir, file), "rb") as w:
            if idx == 0:
                params = w.getparams()
            frames.append(w.readframes(w.getnframes()))

    with wave.open(dst_path, "wb") as out_wav:
        out_wav.setparams(params)
        for f in frames:
            out_wav.writeframes(f)
    print_info(f"合并后的音频已保存至{dst_path}")

def voice_merge(wav_files):
    print_info("开始合并音频文件...")
    frames = []
    params = None
    for idx, file in enumerate(wav_files):
        with wave.open(io.BytesIO(file), "rb") as w:
            if idx == 0:
                params = w.getparams()
            frames.append(w.readframes(w.getnframes()))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out_wav:
        out_wav.setparams(params)
        for f in frames:
            out_wav.writeframes(f)
    buffer.seek(0)  # 重置指针到开始位置
    audio_data = buffer.read()  # 读取所有数据到变量中
    return audio_data


def get_all_voice(dst_path=get_base_dir() + "/data/voice/output_all.wav"):
    with open(dst_path, "rb") as f:
        return f.read()

def markdown_to_text(md_text):
    html = markdown.markdown(md_text)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator='\n')
    return text

# def voice_main_return(file_name, max_len=59):
#     """将文本按中文标点符号拆分，确保每段不超过最大长度限制，最后返回所有音频的二进制拼接结果"""
#     # 常用中文标点符号列表
#     with open(get_base_dir() + f"/data/voice/{file_name}", "r", encoding="utf-8") as f:
#         ori_text = f.read()
#     ori_text = markdown_to_text(ori_text)
#     segments = split_str_length(ori_text, max_len=max_len)
#     print_info("一共有：", len(segments))
#     # 为每个段落生成语音文件
#     for i, segment in enumerate(segments):
#         file_name = f"output_{i}.wav"
#         _voice_main_return(segment, file_name)
#     print_info("所有片段语音合成完成，开始合并...")
#     _merge()

def voice_main_return(ori_text, max_len=59, type_="闽南语"):
    """将文本按中文标点符号拆分，确保每段不超过最大长度限制，最后返回所有音频的二进制拼接结果"""
    # 常用中文标点符号列表
    ori_text = markdown_to_text(ori_text)
    segments = split_str_length(ori_text, max_len=max_len)
    print_info("一共有：", len(segments))
    wav_files = []
    # 为每个段落生成语音文件
    for i, segment in enumerate(segments):
        # file_name = f"output_{i}.wav"
        # TODO:这里统一使用厦大语音平台进行语音转文字了
        wav_files.append(_voice_main_return(segment, type_))

    print_info("所有片段语音合成完成，开始合并...")
    return voice_merge(wav_files)

# async def voice_main_return_async(file_name, max_len=59, callback=None):
#     """异步执行语音合成任务"""
#     try:
#         # 使用 asyncio.to_thread 将同步函数转为异步执行
#         if int(sys.version.split(".")[1]) >= 9:  # Python 3.9+支持to_thread
#             await asyncio.to_thread(voice_main_return, file_name, max_len)
#         else:  # 3.8及以下版本使用run_in_executor
#             loop = asyncio.get_running_loop()
#             ctx = contextvars.copy_context()
#             func_call = functools.partial(ctx.run, voice_main_return, file_name, max_len)
#             await loop.run_in_executor(None, func_call)
#
#         if callback:
#             await on_complete(True)
#             return True
#     except Exception as e:
#         print_info(f"语音合成失败: {e}")
#         if callback:
#             await on_complete(False, str(e))
#         return False
async def voice_main_return_async(ori_text, max_len=59, callback=None, type_="闽南语"):
    """异步执行语音合成任务"""
    try:
        # 使用 asyncio.to_thread 将同步函数转为异步执行
        if int(sys.version.split(".")[1]) >= 9:  # Python 3.9+支持to_thread
            audio_data = await asyncio.to_thread(voice_main_return, ori_text, max_len, type_)
        else:  # 3.8及以下版本使用run_in_executor
            loop = asyncio.get_running_loop()
            ctx = contextvars.copy_context()
            func_call = functools.partial(ctx.run, voice_main_return, ori_text, max_len, type_)
            audio_data = await loop.run_in_executor(None, func_call)

        if callback:
            await callback(True)
        return audio_data
    except Exception as e:
        print_info(f"语音合成失败: {e}")
        if callback:
            await callback(False, str(e))
        return None

async def on_complete(success, error=None):
    """异步回调函数"""
    if success:
        print_info("已完成")
    else:
        print_info(f"语音合成失败: {error}")



if __name__ == "__main__":
    # print(voice_main_return("这是测试的一句话，它不长也不短，只是用于测试闽南语音频合成的效果如何。这是第二句话，用于测试分段功能是否正常工作。如果一切顺利，这段话应该会被分成多个部分，每个部分都不会超过指定的长度限制。"))
    # parser = argparse.ArgumentParser(description='传递参数')
    # parser.add_argument("file_name", type=str)
    # parser.add_argument("--max_len", type=int, default=59)
    # args = parser.parse_args()
    # voice_main_return(args.file_name, args.max_len)  # 传递参数，因为这玩意不需要页面交互，需要多线程实行，所以直接命令行调用
    # print_info("运行结束")
    ...
