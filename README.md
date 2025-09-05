# 第一步：克隆应用空间

```linux
git lfs install
git clone http://git.aistudio.baidu.com/testlyh/teavisor.git
```

[获取access token](https://aistudio.baidu.com/account/accessToken)

# 第二步：创建Streamlit.app.py文件

```python
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

```

# 第三步：上传&提交文件

```
git add Streamlit.app.py
git commit -m "Add application file"
git push
```

# 从项目中部署

除了在本地完成代码开发，上传至应用空间之外，我们也允许大家在[项目](https://aistudio.baidu.com/my/project)中编辑Streamlit前端代码，编辑完成后点击部署即可自动创建应用空间并完成发布。



Streamlit 版本支持：1.33.0、1.30.0

完整文档请见：https://ai.baidu.com/ai-doc/AISTUDIO/Plu48z144