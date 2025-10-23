#该应用创建工具共包含三个区域，顶部工具栏，左侧代码区，右侧交互效果区，其中右侧交互效果是通过左侧代码生成的，存在对照关系。
#顶部工具栏：运行、保存、新开浏览器打开、实时预览开关，针对运行和在浏览器打开选项进行重要说明：
#[运行]：交互效果并非实时更新，代码变更后，需点击运行按钮获得最新交互效果。
#[在浏览器打开]：新建页面查看交互效果。
#以下为应用创建工具的示例代码

import streamlit as st
st.header("应用创建工具入门指南")
"您可以查看[官方文档](https://ai.baidu.com/ai-doc/AISTUDIO/Gktuwqf1x#%E5%BA%94%E7%94%A8%E5%88%9B%E5%BB%BA%E5%B7%A5%E5%85%B7)了解详情，或阅读以下入门指南！"
st.markdown("<hr />",unsafe_allow_html=True)
st.write("**1、输入组件**")
st.write("Streamlit支持多种输入组件，包括按钮、滑块、复选框、选择框、文本输入框等，以下为常用的输入组件样式：")
col1, col2 = st.columns(2)
with col1:
    st.button("常规按钮")
    st.text_input('文本输入组件样式：','默认输入内容')
    st.selectbox('单选择组件样式：',
                          ('默认选项1', '默认选项2', '默认选项3'))
with col2:
    st.download_button("下载按钮",'欢迎使用应用创建工具')
    st.number_input('数字输入组件样式：', 1, 20, 10)
    st.multiselect(
        '多选框组件样式：',
        ['默认选项1', '默认选项2', '默认选项3'],
        ['默认选项1'])
st.markdown("<hr />",unsafe_allow_html=True)
st.write("**2、媒体组件**")
st.write("Streamlit支持多种媒体组件，包括图片、视频、语音等，以下为常用的输入组件样式：")
tab1, tab2, tab3 = st.tabs(["照片", "视频", "语音"])
with tab1:
   st.image("https://codelab-public.bj.bcebos.com/image2.jpg")
with tab2:
   st.video("https://codelab-public.bj.bcebos.com/video.mp4")
with tab3:
    st.audio("https://codelab-public.bj.bcebos.com/audio.oga")
st.markdown("<hr />",unsafe_allow_html=True)
st.write("**3、风格迁移交互示例教程**")
st.write("利用PaddleHub中AnimeGAN模型将输入图片转换成新海诚动漫风格的交互效果，可以通过以下方式展现：")
per_image = st.file_uploader("上传图片", type=['png', 'jpg'], label_visibility='hidden')
col3, col4 = st.columns(2)
with col3:
    if per_image:
        st.image(per_image)
    else:
        st.image("https://codelab-public.bj.bcebos.com/base.jpeg")
    test=st.button("提交图片")
with col4:
    if test:
        st.image("https://codelab-public.bj.bcebos.com/test.jpeg")
        test=False
    else:
        st.write("暂无预测结果,请点击提交图片")
st.markdown("<hr />",unsafe_allow_html=True)
st.write("**以上是推荐常用交互样式哦，还在等什么，赶快行动起来，结合飞桨模型打造你的专属应用吧！**")
