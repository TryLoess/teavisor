MAPPING = {
    "algal leaf": "地衣病",
    "Anthracnose": "茶饼病",
    "bird eye spot": "茶煤病",
    "brown blight": "茶褐斑病",
    "gray light": "茶灰斑病",
    "healthy": "健康叶片",
    "red leaf spot": "茶红锈病",
    "white spot": "white spot"  # 未找到对应中文病名，输出原始病名
}
CSS = """
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
        .stMain div[data-testid="stHorizontalBlock"]:has(> div:nth-child(3):nth-last-child(1)),
        .main div[data-testid="stHorizontalBlock"]:has(> div:nth-child(3):nth-last-child(1)) {  /* 使用has进行条件判断，>div代表其下的所有div元素，既是第3个child，也是最后一个child说明只有三个元素 */
            position: fixed;
            bottom: 0;
            width: calc(80% - var(--sidebar-width-state, 20rem) * var(--sidebar-width-state, 1.1));
            background-color: rgba(255, 255, 255, 1);
            z-index: 9999;
            padding: 10px 0;
            /* box-shadow: 0 -2px 8px rgba(0,0,0,0.1); */
        }
        /* 让按钮在水平Block中保持水平排列 */
        .stMain div[data-testid="stHorizontalBlock"] div[data-testid="column"],
        .main div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
            position: static !important;
            box-shadow: none;
        }
        
        /* 防止内容被底部工具栏遮挡 */
        .block-container {
            padding-bottom: 80px !important;
        }
        
        /* 在移动设备上的适配 */
        @media (max-width: 768px) {
        .main div[data-testid="stHorizontalBlock"]:has(> div:nth-child(3):nth-last-child(1)) {
                width: 80%;
            }
        }
        </style>
        """

API_KEY = "aDtioH6XEYlaAovoITWq4nrL"  # TODO:需要隐藏
SECRET_KEY = "GHX3VwImLrBKpHsapNqLVsFfmqJzUqTI"

