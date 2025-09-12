import inspect
import os
import base64
import asyncio
from PIL import Image
import io
import numpy as np
import sys

async def tran_sync_to_async(sync_func, *args, **kwargs):
    """将同步函数转换为异步函数"""
    version = sys.version_info.minor
    if version >= 9:
        return await asyncio.to_thread(sync_func, *args, **kwargs)
    else:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: sync_func(*args, **kwargs))

async def multiple_tasks(sync_func_list):
    tasks = []
    for sync_func, args in sync_func_list:
        task = tran_sync_to_async(sync_func, *args)
        tasks.append(task)

    # 并发执行所有任务
    results = await asyncio.gather(*tasks)
    return results



def split_str_length(ori_text, max_len=59):
    chinese_punctuations = [
        "。", "，", "；", "：", "？", "！",
        """, """, "「", "」", "《", "》",
        "、", "．", "…", "·", "—", "～",
        "（", "）", "【", "】", "［", "］"
    ]

    # 按标点符号拆分文本
    segments = []
    current_segment = ""

    for char in ori_text:
        current_segment += char

        # 如果当前字符是标点符号或当前段落即将超过长度限制
        if len(current_segment) >= max_len:
            for rechar in reversed(current_segment):
                if rechar in chinese_punctuations:
                    split_index = current_segment.rfind(rechar) + 1
                    segments.append(current_segment[:split_index])
                    current_segment = current_segment[split_index:]
                    break

    # 添加最后一个段落（如果有）
    if current_segment:
        segments.append(current_segment)
    return segments

def get_img_base64(file_name: str) -> str:
    """
    基于data/pic下的图片，进行base64编码
    """
    image = get_base_dir(True) + "/data/pic" + f"/{file_name}"
    # 读取图片
    with open(image, "rb") as f:
        buffer = io.BytesIO(f.read())
        base_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{base_str}"

def print_info(*args, sep=' ', end='\n', file=None, flush=False):
    caller_frame = inspect.currentframe().f_back
    filename = caller_frame.f_code.co_filename
    line_no = caller_frame.f_lineno
    prefix = f"[{filename}: {line_no}]: "
    message = sep.join(map(str, args))
    print(f"{prefix} {message}".encode('gbk', errors='replace').decode('gbk'), end=end, file=file, flush=flush)


def video_deal(video_path):
    # cap = cv2.VideoCapture(video_path)
    # res = []
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     res.append(frame)
    # cap.release()
    # return np.array(res)
    with open(video_path, 'rb') as file:
        return file.read()



def get_img_array(file_name: str) -> np.ndarray:
    try:
        image = get_base_dir(True) + "/data/pic" + f"/{file_name}"
        img = Image.open(image)
        img = img.convert('RGB')
        img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"报错: {e}")

def tran_jpg_binary(image_path):
    base_path = get_base_dir(True) + "/data/pic/" + image_path
    try:
        with Image.open(base_path) as img:
            output = io.BytesIO()
            img.convert("RGB").save(output, format="JPEG")
            jpg_binary = output.getvalue()
            output.close()
            return jpg_binary
    except Exception as e:
        print(f"转换图像时发生错误: {e}")
        return None

def get_base_dir(parent=True):
    """默认返回根目录"""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if parent:
        return os.path.join(src_dir, os.path.pardir)
    return src_dir