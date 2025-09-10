import streamlit as st
from src.utils import *
from src.dialogue import *

@st.cache_data()
def load_location_data():
    with open(get_base_dir() + "/data/location_dict.json", "r", encoding="utf-8") as f:
        location_dict = json.load(f)
    return location_dict

def init():
    if "location" not in st.session_state:
        st.session_state.location = ""

def main():
    st.set_page_config(
        "TeaVisor",
        page_icon="🌱",
        initial_sidebar_state="expanded",
        layout="centered",
    )

    st.markdown(
        """
        <style>
        [data-testid="stSidebarUserContent"] {
            padding-top: 20px;
        }
        # [data-testid="stSidebar"] {
        #     max-width: 600px;
        #     resize: horizontal;
        #     overflow: auto;
        #     margin-bottom: 0px;
        # }
        [data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 29% !important;
        }

        /* 侧边栏收回状态不设置最小宽度限制 */
        [data-testid="stSidebar"][aria-expanded="false"] {
            min-width: initial !important;
        }
        .block-container {
            padding-top: 25px;
        }
        [data-testid="stBottomBlockContainer"] {
            padding-bottom: 20px;
        }
        [data-testid="stSidebarUserContent"] img {
            position: relative;
            top: 0% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    with st.sidebar:
        st.image(
            get_img_base64("sider_head.jpg"),
            # use_container_width=True,
        )
        # st.caption(
        #     unsafe_allow_html=True,
        # )

        # selected_page = sac.menu(
        #     [
        #         sac.MenuItem("多功能对话", icon="chat"),
        #         # sac.MenuItem("实时目标检测", icon="hdd-stack"),
        #     ],
        #     key="selected_page",
        # )
        st.markdown("---")
    hidden()
    # if selected_page == "实时目标检测":
    #     st.markdown("# PP-YOLOv2还在数据收集与训练当中，之后就会开放~")
    create_location(load_location_data())
    create_select_model()
    main_chat_dialog()
        # dialogue_page(api=api, is_lite=is_lite)
if __name__ == "__main__":
    main()