import asyncio
import copy
import json
import random
import time
from traceback import print_exc

import streamlit as st
import cv2
from .yolo_infer import InferYOLO, resize_image
from .chat_ai import *
from .utils import *

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

def predict_image(infer, file_name, dst_name, conf_threshold=0.45):
    """file_path只需要输入文件名即可，所有pic都需要存入pic文件夹下统一转换读取，file"""
    print_info("直接预测")
    class_names = cache_model()[1]
    result_img, _, boxes, scores, class_ids = infer.predict(get_base_dir() + "/data/pic/" + file_name, conf_threshold=conf_threshold)

    # Save the annotated image
    cv2.imwrite(get_base_dir() + "/data/pic/" + dst_name, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    for i, box in enumerate(boxes):
        class_id = class_ids[i]
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        print(f"检测到{class_name}置信度为{scores[i]:.2f}在{box}")
    return get_base_dir() + "/data/pic/" + dst_name

def predict_image_use_resize(infer, ori_file, resize_file, dst_name, conf_threshold=0.45):
    print_info("使用resize预测")
    class_names = cache_model()[1]
    result_img, boxes, scores, class_ids = infer.base_resize_predict(ori_file, resize_file, conf_threshold=conf_threshold)
    # cv2.imwrite(get_base_dir() + "/data/pic/" + dst_name, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    for i, box in enumerate(boxes):
        class_id = class_ids[i]
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        print(f"检测到{class_name}置信度为{scores[i]:.2f}在{box}")
    # return get_base_dir() + "/data/pic/" + dst_name
    return result_img # cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
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
    # def _acti
    def _activate():
        st.session_state.messages = msgs
    st.button(msgs[0]["content"], use_container_width=True, on_click=_activate)

# def _judge_upload_file():
    # if isinstance(st.session_state.upload_file, UploadedFile):
    #     return True
    # else:
    #     return False


def get_response(prompt, model):
    if model == "联网Agent":
        time.sleep(20)
        return "test的无意义内容啊啊啊"
        retry = 0
        while retry < 5:
            try:
                if st.session_state[f"upload_file_{st.session_state.uploader_key}"] is not None:
                    if st.session_state.need_yolo:
                        print_info("需要yolo")
                        fileres = FileResponse(st.session_state.base_response, file=st.session_state[f"upload_file_{st.session_state.uploader_key}"])
                    else:
                        fileres = FileResponse(st.session_state.base_response, "yolo_pic.jpg")
                    print_info("fileres正常运行")
                    response = fileres.get_response(prompt, st.session_state.location)
                else:
                    response = st.session_state.base_response.get_response(prompt, st.session_state.location)
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
def _save_upload_file(save_pic_name, resize=False):
    assert st.session_state[f"upload_file_{st.session_state.uploader_key}"] is not None, "上传文件不能为空才对，有问题"
    image = Image.open(st.session_state[f"upload_file_{st.session_state.uploader_key}"])
    rgb_image = image.convert("RGB")
    save_pic_path = get_base_dir() + "/data/pic/" + save_pic_name
    rgb_image.save(save_pic_path, format="JPEG")
    if resize:
        resize_path = resize_image(save_pic_path)
    else:
        resize_path = None
    return save_pic_path, resize_path

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

    with st.sidebar:
        uploader_container = st.empty()

        if st.session_state.show_uploader:
            with uploader_container:
                st.file_uploader(
                    "请在这里上传茶叶图片",
                    type=["png", "jpg", "jpeg"],
                    key=f"upload_file_{st.session_state.uploader_key}"
                )

        st.markdown("---")
        col1, col2 = st.columns([1, 1])

        if col1.button("开启新对话", use_container_width=True):
            if "messages" in st.session_state and len(st.session_state.messages) > 0:
                if st.session_state.messages not in st.session_state.history_conversations:
                    st.session_state.history_conversations.insert(0, st.session_state.messages)
                st.session_state.messages = []
            # 通过改变 key 来重置文件上传器
            st.session_state.uploader_key += 1
            st.session_state.need_yolo = True
            st.rerun()

        if col2.button("🗑️清空当前对话", use_container_width=True):
            st.session_state.messages = []
            # 通过改变 key 来重置文件上传器
            st.session_state.uploader_key += 1
            st.session_state.need_yolo = True
            st.rerun()
        if st.session_state.history_conversations:
            with st.expander("历史对话列表如下"):
                for i in st.session_state.history_conversations:
                    _create_button(i)


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
            if st.session_state[f"upload_file_{st.session_state.uploader_key}"] is not None:
                st.session_state.messages[-1]["pic"] = copy.deepcopy(st.session_state[f"upload_file_{st.session_state.uploader_key}"])

            st.session_state.prompt = prompt
            st.session_state.in_process = True
            st.rerun()

        yolo_pic_path = get_base_dir() + "/data/pic/" + "yolo_pic.jpg"
        assistant_message = st.chat_message("assistant", avatar=head_pic()["assistant"])
        if st.session_state[f"upload_file_{st.session_state.uploader_key}"] is not None:
            # if st.session_state.need_yolo:
            infer = cache_model()[0]
            user_pic_path, resize_path = _save_upload_file("user_upload.jpg", True)
            # predict_image(infer, "user_upload.jpg", "yolo_pic.jpg", conf_threshold=0.45)
            img = predict_image_use_resize(infer, user_pic_path, resize_path, "yolo_pic.jpg", conf_threshold=0.5)
            # st.session_state.need_yolo = False
            assistant_message.image(img.copy())# yolo_pic_path)
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

            full_reply = get_response(st.session_state.prompt, st.session_state.select_model)
        assistant_message_placeholder.empty()  # 清空内容

        random_stream_text(assistant_message_placeholder, full_reply)
        st.session_state.messages.append({"role": "assistant", "content": full_reply})
        if st.session_state[f"upload_file_{st.session_state.uploader_key}"] is not None:
            st.session_state.messages[-1]["pic"] = img.copy()
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