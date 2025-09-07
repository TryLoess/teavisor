import asyncio
import copy
import json
import random
import time
from copy import deepcopy
from traceback import print_exc

import streamlit as st
import cv2

from .yolo_infer import InferYOLO, resize_image, predict_image, predict_image_use_resize
from .chat_ai import *
from .utils import *
from .chat_ai_openai import OpenaiResponse
from streamlit_modal import Modal

def hidden():
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                <![]()yle>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

def _disable_chat_input():
    st.session_state.disable_chat_input = True

def create_location(location_dict):
    if "location" not in location_dict:
        st.session_state.location = ""
    with st.sidebar:
        st.markdown("请在此输入茶园所在地")
        col1, col2, col3 = st.columns(3)
        sheng = col1.selectbox(
            "省份",
            key="sheng",
            options=["请选择"] + list(location_dict.keys())
        )
        # xian = col2.selectbox(
        #     "市区",
        #     key="xian",
        #     options=[] if sheng == "请选择" else location_dict[sheng].keys(),
        # )
        # xiang = col3.selectbox(
        #     "乡县",
        #     key="xiang",
        #     options= [] if (xian is None or sheng == "请选择") else (location_dict[sheng][xian]),
        # )
        # if xian is not None and sheng != "请选择":
        #     if xiang is not None:
        #         st.session_state.location = sheng + xian + xiang
        #     else:
        #         st.session_state.location = sheng + xian
        xian = col2.selectbox(
            "市区",
            key="xian",
            options=[] if sheng == "请选择" else ["请选择"] + list(location_dict[sheng].keys()),
        )
        xiang = col3.selectbox(
            "乡县",
            key="xiang",
            options= [] if (xian == "请选择" or sheng == "请选择") else (location_dict[sheng][xian]),
        )
        if xian != "请选择" and sheng != "请选择":
            if xiang is not None:
                st.session_state.location = sheng + xian + xiang
            else:
                st.session_state.location = sheng + xian
        st.markdown("---")


async def random_stream_text_asynic(ai, text, speed_range=(0.002, 0.06)):
    min_speed, max_speed = speed_range
    total = ""
    for char in text:
        total += char
        ai.write(total)  # 模拟流式输出
        await asyncio.sleep(random.uniform(min_speed, max_speed))  # 异步睡眠
    st.session_state.messages.append({"role": "assistant", "content": text})

def random_stream_text(ai, text, speed_range=(0.002, 0.06)):
    min_speed, max_speed = speed_range
    total = ""
    for char in text:
        total += char
        ai.write(total)  # 模拟流式输出
        time.sleep(random.uniform(min_speed, max_speed))  # 异步睡眠

def create_select_model():
    st.markdown("""
    <style>
    .main div[data-testid="stSelectbox"] {
    position: fixed;   /* 固定定位 */
    top: 4%;         /* 距顶部距离 */
    # left: 30%;
    z-index: 1000;     /* 确保在最上层 */
    width: 200px !important;  /* 设置宽度 */
    background: white;
    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
/* 调整输入框部分 */
# div.stElementContainer.st-key-select_model
    .main div[data-testid="stSelectbox"] > div[role="combobox"] {
    padding: 4px 8px;  /* 内边距 */
    min-height: 30px;  /* 最小高度 */
}

/* 调整字体大小 */
   .main div[data-testid="stSelectbox"] {
    font-size: 1em !important;
}

/* 调整下拉菜单位置 */
   .main  div[data-testid="stSelectbox"] [role="listbox"] {
    transform: translateY(38px) !important;  /* 下拉菜单位置修正 */
    width: 200px !important;  /* 下拉菜单宽度 */
}
    /* 让Modal贴近顶部 */
    div[data-modal-container='true'][key='camera_modal'] > div:first-child > div:first-child{
        margin-top: -7% !important;
          overflow: visible !important; /* 取消滚动，内容溢出时全部显示 */
      max-height: none !important;  /* 取消最大高度限制 */
      height: auto !important;      /* 高度自适应内容 */
    }
    /* 优化移动端显示 */
    @media (max-width: 768px) {
        div[data-modal-container='true'][key='camera_modal'] > div:first-child > div:first-child{
            margin-top: -10% !important;
            padding: 9% !important;
              overflow: visible !important; /* 取消滚动，内容溢出时全部显示 */
  max-height: none !important;  /* 取消最大高度限制 */
  height: auto !important;      /* 高度自适应内容 */
        }
    }

    /* 美化相机输入组件 */
    .stCamera > div {
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    </style>
    """, unsafe_allow_html=True)

    select_model = st.selectbox(
        "选择模型",
        ["联网Agent", "微调多模态"],
        key="select_model",
        label_visibility="collapsed"
    )
    if select_model != st.session_state.select_model:
        st.session_state.select_model = select_model
        st.rerun()

@st.cache_resource
def head_pic():
    return {"assistant": get_img_array("bot_head.png"),
            "user": get_img_array("user_head2.jpg")}

@st.cache_resource
def cache_model():
    class_names = ["back", '茶煤病', '茶饼病', '茶褐斑病', '遗传性病变', '茶灰斑病', '健康叶片', '地衣病', '缺镁症', '茶螨害',
                   '缺氮症', '缺钾症', '茶红锈病', '缺硫症', '日灼病']  # 目前是14种病症
    model_dir = get_base_dir() + "/yolo_model/cloud_yoloe"  # Directory containing model.pdmodel and model.pdiparams
    infer = InferYOLO(model_dir=model_dir, class_names=class_names, use_gpu=False)
    return infer, class_names

@st.cache_resource
def cache_openai():
    return OpenaiResponse()

def create_camera():
    with st.session_state.modal.container():
        print_info("已打开modal")
        st.session_state.camera_image = None
        st.markdown(
            "<h3 style='text-align: center; margin-top: 0;'>📸 点击相机下方的按钮（Take Photo）即可拍照</h3>"
            "<p style='text-align: right; margin-top: 0;'>点击这下面的相机按钮可切换前后摄像头</p>",
            unsafe_allow_html=True
        )
        print_info("存在更新相机组件")
        st.session_state.camera_image = st.camera_input("相机实时画面")
        if st.session_state.camera_image is not None:
            st.session_state.uploader_key += 1  # 这里将上传图片的key改变，达到重置上传图片的目的
            st.session_state.need_yolo = True
            st.success("已成功拍照")
            print_info("已成功拍照")
            time.sleep(1)
            st.session_state.modal.close()
            st.rerun()

# @st.cache_resource
# def loading_video():
#     video = video_deal(get_base_dir() + "/data/loading.mp4")
#     print(type(video))
#     return video
# async def get_response_async(prompt):
#     if st.session_state.upload_file is not None:
#         fileres = FileResponse(st.session_state.base_response, "test.jpg")
#         task = asyncio.create_task(fileres.get_response_async(prompt, st.session_state.location))
#     else:
#         task = asyncio.create_task(st.session_state.base_response.get_response_async(prompt, st.session_state.location))
#
#     st.image()
#
#     await task

def _create_button(msgs: list):
    def _activate():
        st.session_state.messages = msgs
    st.button(msgs[0]["content"], use_container_width=True, on_click=_activate)

# def _judge_upload_file():
    # if isinstance(st.session_state.upload_file, UploadedFile):
    #     return True
    # else:
    #     return False

def get_response(prompt, model, use_star=True):
    """使用星河社区的模型"""
    if model == "联网Agent":
        # time.sleep(20)
        # return "test的无意义内容啊啊啊"
        retry = 0
        while retry < 5:
            try:
                now_img = _get_ont_img()
                if now_img is not None:
                    if not use_star:
                        if st.session_state.need_yolo:
                            print_info("需要yolo")
                            fileres = FileResponse(st.session_state.base_response, file=now_img)
                        else:
                            fileres = FileResponse(st.session_state.base_response, "yolo_pic.jpg")
                        print_info("fileres正常运行")
                        response = fileres.get_response(prompt, st.session_state.location)
                    else:
                        if st.session_state.need_yolo:
                            print_info("需要yolo")
                            response = cache_openai().process_tea_disease_image(prompt, st.session_state.location, file=now_img)
                        else:
                            response = cache_openai().process_tea_disease_image(prompt, st.session_state.location, file_path=get_base_dir() + "/data/pic/yolo_pic.jpg")
                else:
                    if not use_star:
                        response = st.session_state.base_response.get_response(prompt, st.session_state.location)
                    else:
                        response = cache_openai().only_text(prompt, st.session_state.location)
                    # response = st.session_state.base_response.get_response(prompt, st.session_state.location)
                if use_star:
                    return response
                return response.json()["answer"]
            except Exception as e:
                print_exc()
                retry += 1
        else:
            return "疑似服务器出现问题，麻烦联系管理员"
    else:
        return "当前模型还在微调，未来将会接入该网站~请先调整回联网Agent模式进行使用"

def insert_video():
    file_path = get_base_dir() + "/data/loading.mp4"

    gif_html = f"""
    <div style="display: flex; justify-content: center; align-items: center;">
        <video autoplay loop muted playsinline style="pointer-events: none;">
            <source src="{file_path}"type="video/mp4">
            你的设备不支持该加载动画
        </video>
    </div>
    """

    # 通过 markdown 插入 HTML
    st.markdown(gif_html, unsafe_allow_html=True)
# def _save_upload_file(save_pic_name, resize=False):
#     assert st.session_state[f"upload_file_{st.session_state.uploader_key}"] is not None, "上传文件不能为空才对，有问题"
#     image = Image.open(st.session_state[f"upload_file_{st.session_state.uploader_key}"])
#     rgb_image = image.convert("RGB")
#     save_pic_path = get_base_dir() + "/data/pic/" + save_pic_name
#     rgb_image.save(save_pic_path, format="JPEG")
#     if resize:
#         resize_path = resize_image(save_pic_path)
#     else:
#         resize_path = None
#     return save_pic_path, resize_path
def _save_upload_file(save_pic_name, resize=False):
    now_img = _get_ont_img()
    assert now_img is not None, "上传文件不能为空才对，有问题"
    image = Image.open(now_img)
    rgb_image = image.convert("RGB")
    save_pic_path = get_base_dir() + "/data/pic/" + save_pic_name
    rgb_image.save(save_pic_path, format="JPEG")
    if resize:
        resize_path = resize_image(save_pic_path)
    else:
        resize_path = None
    return save_pic_path, resize_path

def _get_ont_img():
    if st.session_state[f"upload_file_{st.session_state.uploader_key}"] is not None or st.session_state.camera_image is not None:
        if st.session_state[f"upload_file_{st.session_state.uploader_key}"] is not None:
            print_info("使用上传图片")
            return st.session_state[f"upload_file_{st.session_state.uploader_key}"]
        else:
            print_info("使用相机拍照的图片")
            return st.session_state.camera_image
    return None

def _reset_ss():
    st.session_state.uploader_key += 1
    st.session_state.need_yolo = True
    st.session_state.camera_image = None
    print_info("已清空")
    st.rerun()

def _read_pic(pic_name):
    img_path = get_base_dir() + "/data/pic/" + pic_name
    img = cv2.imread(img_path)
    return deepcopy(img)

def main_chat_dialog():
    st.markdown(
        """
        <style>
        .loading_gif{
          display: flex;
          justify-content: center; /* 水平居中 */
          align-items: center;     /* 垂直居中（如果需要） */
          margin-top: 16px;        /* 内容与图片间距 */
        }
        .loading_gif img{
          max-width: 80%;
          height: auto;
          display: block;
        }
        /* 用户消息样式 - 让消息容器成为 flex 容器 */
        .stChatMessage:has([aria-label="Chat message from user"]) {
          display: flex;
          flex-direction: row-reverse; /* 反转排列，将头像放在右侧 */
          align-items: flex-start; /* 顶部对齐 */
          justify-content: flex-end; /* 内容靠右 */
          gap: 3px; /* 头像与消息之间的间距 */
        }
        
        /* 头像图片样式 */
        .stChatMessage:has([aria-label="Chat message from user"]) > img[alt="user avater"] {
          width: 40px; /* 头像固定宽度 */
          height: 40px; /* 头像固定高度 */
          order: 1; /* 控制 flex 布局中的顺序 */
        }
        
        .stChatMessage:has([aria-label="Chat message from user"]) div {
          flex: 1; /* 占满剩余空间 */
          display: flex;
          justify-content: flex-end; /* 内容靠右 */
        }
        
        /* 保证消息中的图片大小合适 */
        .stChatMessage:has([aria-label="Chat message from user"]) img {
          max-width: 40% !important; /* 限制图片最大宽度 */
          display: block;
            margin-left: auto;   /* 让图片靠右 */
            margin-right: 0;     /* 保证右边没有多余间距 */
        }
            
        /* Optional: Style the video elements */
        [data-testid="stVideo"] {
          pointer-events: none;
          overflow: hidden;
        }
        # .stMain div[data-testid="stHorizontalBlock"] {
        #     position: fixed; /* 或 absolute，根据需求选择 */
        #     bottom: 5%;
        #     left: calc(415px * var(--sidebar-width-state, 1));
        #     right: 0;
        #     width: calc(100% - 415px * var(--sidebar-width-state, 1));
        #     z-index: 1000; /* 保证元素在页面最顶层 */
        # }
        </style>
        """,
        unsafe_allow_html=True
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "base_response" not in st.session_state:
        st.session_state.base_response = AllResponse()
    if "disable_text_input" not in st.session_state:
        st.session_state.disable_text_input = True
    if "loading" not in st.session_state:
        st.session_state.loading = False
    if "history_conversations" not in st.session_state:
        st.session_state.history_conversations = []
    if "in_process" not in st.session_state:
        st.session_state.in_process = False
    if "prompt" not in st.session_state:
        st.session_state.prompt = ""
    if "need_yolo" not in st.session_state:
        st.session_state.need_yolo = True
    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = True
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "camera_image" not in st.session_state:
        st.session_state.camera_image = None
    if "modal" not in st.session_state:
        st.session_state.modal = Modal(
        "相机",
        key="camera_modal",
        max_width=600
    )
    if "yolo_pic" not in st.session_state:
        st.session_state.yolo_pic = None

    with st.sidebar:
        uploader_container = st.empty()
        col1, col2 = uploader_container.columns([1, 1])
        if st.session_state.show_uploader:
            with col1:
                st.file_uploader(
                    "请在这里上传茶叶图片",
                    type=["png", "jpg", "jpeg"],
                    key=f"upload_file_{st.session_state.uploader_key}"
                )
                if st.session_state[f"upload_file_{st.session_state.uploader_key}"] is not None:
                    st.session_state.camera_image = None  # 重置拍照，二者只能存在一个
            with col2:
                if st.button("使用相机拍照", use_container_width=True):
                    st.session_state.modal.open()
                if st.button("清空图片", use_container_width=True):
                    _reset_ss()

                now_img = _get_ont_img()
                if now_img is not None:
                    st.image(now_img, caption="当前照片", use_column_width=True)
                else:
                    st.info("当前无图片")

        st.markdown("---")
        col1, col2 = st.columns([1, 1])

        if col1.button("开启新对话", use_container_width=True):
            if "messages" in st.session_state and len(st.session_state.messages) > 0:
                if st.session_state.messages not in st.session_state.history_conversations:
                    st.session_state.history_conversations.insert(0, st.session_state.messages)
                st.session_state.messages = []
            # 通过改变 key 来重置文件上传器
            _reset_ss()

        if col2.button("🗑️清空当前对话", use_container_width=True):
            st.session_state.messages = []
            # 通过改变 key 来重置文件上传器
            _reset_ss()
        if st.session_state.history_conversations:
            with st.expander("历史对话列表如下"):
                for i in st.session_state.history_conversations:
                    _create_button(i)

    if st.session_state.modal.is_open():
        print_info("现在需要使用相机")
        create_camera()


    for msg in st.session_state.messages:
        role = st.chat_message(msg["role"], avatar=head_pic()[msg["role"]])
        if "pic" in msg.keys():
            role.image(msg["pic"])
        role.write(msg["content"])

    st.session_state.disable_text_input = False if st.session_state.location != "" else True
    if st.session_state.location != "" :
        disable_text = "请输入问题~还可以上传图片哦"
    else:
        disable_text = "请先在侧边栏配置位置，再来输入问题可以获得更加准确的结果哦~"
    if st.session_state.in_process:
        disable_text = "现在正在运行，请等待运行结束再进行输入~"
 # , accept_file=True, type=["png", "jpg", "jpeg"])


    # 获取用户输入
    if (prompt := st.chat_input(disable_text,
                           disabled=st.session_state.disable_text_input or st.session_state.in_process,
                           on_submit=_disable_chat_input)) or st.session_state.in_process:
        if not st.session_state.in_process:

            st.session_state.messages.append({"role": "user", "content": prompt})
            # user = st.chat_message("user", avatar=head_pic()["user"])
            # with user:
            #     if st.session_state.upload_file is not None:
            #         st.write("已上传:" + st.session_state.upload_file.name)
            #         st.image(st.session_state.upload_file)
            #
            #     st.write(prompt)
            now_img = _get_ont_img()
            if now_img is not None:
                st.session_state.messages[-1]["pic"] = copy.deepcopy(now_img)

            st.session_state.prompt = prompt
            st.session_state.in_process = True
            st.rerun()

        assistant_message = st.chat_message("assistant", avatar=head_pic()["assistant"])
        now_img = _get_ont_img()
        if now_img is not None:
            # if st.session_state.need_yolo:
            infer = cache_model()[0]
            user_pic_path, resize_path = _save_upload_file("user_upload.jpg", True)
            # predict_image(infer, "user_upload.jpg", "yolo_pic.jpg", conf_threshold=0.45)
            st.session_state.yolo_pic = predict_image_use_resize(infer,
                                                                 user_pic_path,
                                                                 resize_path,
                                                                 "yolo_pic.jpg",   # 暂时弃用的参数
                                                                 class_names=cache_model()[1],
                                                                 conf_threshold=0.7)
            st.session_state.need_yolo = False
            assistant_message.image(st.session_state.yolo_pic.copy())# yolo_pic_path)
            # st.session_state.upload_file = upload_file_path  # assistant_message.image(st.session_state.upload_file)

        #     st.session_state.upload_file = None
        assistant_message_placeholder = assistant_message.empty()
        with assistant_message_placeholder.container():
            show1, show2 = st.columns([1, 8])
            # show1.video(get_base_dir() + "/data/loading.mp4", loop=True)
            with show1:
                # st.markdown('<div class="loading_gif">', unsafe_allow_html=True)
                show1.image(get_base_dir() + "/data/loading_gif.gif")
                # st.markdown('</div>', unsafe_allow_html=True)
            # with show1:
            #     insert_video()
            show2.markdown("<h2>大模型正在思考……</h2>", unsafe_allow_html=True)
            print_info("等待输出中……")
            full_reply = get_response(st.session_state.prompt, st.session_state.select_model)
        assistant_message_placeholder.empty()  # 清空内容

        random_stream_text(assistant_message_placeholder, full_reply)
        st.session_state.messages.append({"role": "assistant", "content": full_reply})
        if now_img is not None:
            st.session_state.messages[-1]["pic"] = st.session_state.yolo_pic
        st.session_state.disable_text_input = False
        st.session_state.in_process = False
        st.rerun()
        # st.session_state.messages.append({"role": "assistant", "content": full_reply})
    # pointer - events: none;

    if not st.session_state.messages:
        st.markdown("""
    # 🌱 欢迎使用 TeaVisor

    ✨ **智能助力茶叶种植**
    - **多模态智能分析**：结合图片与基于种植地气候与天气预报，深入诊断茶叶生长状况与病虫害问题
    - **智能搜索引擎**：结合百度智能搜索，获得与时俱进的解决方案
    - **简单易用**：手机即开即用，无使用门槛
    ---
    📌 **核心功能，一目了然**！
    - **配置位置**：**请先在侧边栏设置种植地位置**，系统会获取当地气候信息。
    - **上传图片**：**上传茶叶图片**识别，诊断生长状况和病虫害。
    - **输入问题**：在**下方输入栏输入**您的问题，之后模型就会回答您~
            """)