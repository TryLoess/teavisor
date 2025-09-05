import inspect
import os
import base64

from PIL import Image
import io
import numpy as np

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