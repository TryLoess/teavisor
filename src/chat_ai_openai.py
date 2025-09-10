import base64
import os
from openai import OpenAI
from streamlit import image
from streamlit.runtime.uploaded_file_manager import UploadedFile
from .utils import print_info
from .chat_ai import get_weather, get_search


# from .config import MAPPING

def base64_encode(file_path=None, file=None):
    """返回base64编码下的图片字符串"""
    if file_path is not None:
        try:
            if not os.path.exists(file_path):
                return "错误: 无法找到指定图片文件"

            file_ext = file_path.split('.')[-1].lower()
            if file_ext not in ['jpg', 'jpeg', 'png']:
                return "错误: 文件格式必须是JPG或PNG"

            with open(file_path, "rb") as f:
                base64_img = base64.b64encode(f.read()).decode("utf-8")

        except Exception as e:
            return f"错误: 读取图片文件失败 - {str(e)}"

    elif file is not None:
        if isinstance(file, UploadedFile):
            file_bytes = file.read()
        elif isinstance(file, bytes):
            file_bytes = file
        else:
            print_info(type(file))
            raise ValueError("file参数必须是UploadedFile或bytes类型")
        base64_img = base64.b64encode(file_bytes).decode("utf-8")
        file_ext = "jpg"
    else:
        print_info(type(file))
        raise ValueError("必须提供file_path或file参数")

    if not base64_img or base64_img.strip() == "":
        return "错误: 图像数据为空"

    return base64_img, file_ext
class OpenaiResponse:
    def __init__(self):
        self.client = OpenAI(
            api_key="2a89c789ff3dd03f4fe53df5bca90858a26e1857",
            base_url="https://aistudio.baidu.com/llm/lmapi/v3",
        )
        self.disease_prompt = """### 病虫害特征

        病害/失调名称 (Disease/Disorder)	主要危害部位	关键特征 (Key Characteristics for Identification)
        茶煤病 	叶片、枝干	叶片表面覆盖一层黑色烟煤状霉层，霉层厚度和紧密度因病原种类而异。严重时可蔓延至小枝及茎上。常伴随黑刺粉虱、蚧类或蚜虫及其分泌的蜜露存在。
        茶饼病 	嫩叶、嫩茎	叶片正面出现淡黄色或红棕色半透明凹陷病斑，对应背面凸起呈馒头状（饼状），并生有灰白色或粉红色粉状霉层（担子层）。嫩茎和叶柄受害后肿胀扭曲。
        茶褐斑病 (茶褐色叶斑病) 	老叶为主	多从叶缘开始产生褐色小斑，扩大后呈圆形或半圆形褐色病斑，病斑中央色略浅，边缘紫褐色，病健部分界不明显。后期病斑上产生灰黑色霉层（分生孢子梗和分生孢子）。
        茶灰斑病 (山茶灰斑病) 	成叶、老叶	病斑较大（直径可达10-20mm或更大），不规则形，初期褐色，后期病斑中央变为灰白色，边缘褐色并隆起。病斑上散生较大的黑色小点粒（分生孢子盘），潮湿时溢出黑色粘孢子团。
        日灼病 (非侵染性)	叶片	通常发生在叶片中部或边缘，尤其是向阳面。受害处褪绿变白或呈黄褐色焦枯状，质地变脆，病健部分界常较清晰。
        缺氮症 (非侵染性)	全株老叶	老叶均匀失绿变黄（黄化从老叶开始），植株生长不良，芽叶瘦小，叶片变薄。严重时整株叶片焦黄脱落。
        缺钾症 (非侵染性)	老叶	老叶叶尖和叶缘先变黄，进而出现焦枯坏死斑（灼烧状），叶片向下卷曲，根系发育不良。
        缺镁症 (非侵染性)	老叶	老叶叶脉间失绿黄化，但叶脉仍保持绿色，形成清晰的网状花纹。严重时整叶发黄脱落。
        缺硫症 (非侵染性)	新叶	新叶均匀褪绿黄化（与缺氮相似，但症状先出现在新叶），叶片变薄，节间缩短，植株矮小。
        茶螨害 (虫害)	嫩叶、成叶	受害叶片失去光泽，质地变硬、变脆。叶背常可见锈色或铜褐色斑块，严重时叶片脱落。肉眼需仔细查看是否有极小的螨虫活动。
        地衣病 [注]	枝干	在茶树枝条上附着灰绿色、叶状、壳状或枝状的共生体（藻类和真菌共生）。常指示树势衰弱。
        茶红锈病 [注]	枝叶	叶片背面出现橙黄色至红锈色的蜡质状小斑点（锈孢子器），对应叶正面褪绿。嫩枝受害部位肿胀。
        遗传性病变 [注]	不确定	此术语在茶树中不常见。可能指植物本身遗传变异引起的非正常生长，如图斑、白化等，但需与病毒病等区分。
        """
    def only_text(self, user_ask, city, input_pic=""):
        """input_pic决定了是否需要传入图片照片"""
        sys_prompt = """# 身份介绍
你是一位经验丰富的茶叶种植专家，掌握茶叶种植管理、病虫害防治和气候对作物影响的专业知识。你的任务是根据用户提供的种植地气候数据，结合茶叶种植的最佳实践，科学分析用户的问题并提供权威建议。
无论是种植技术、病虫害防控还是气候影响分析，你需要基于搜索引擎的结果进行回答，并确保建议具有可操作性，帮助用户实现高效、健康的茶叶种植。
"""
        weather = get_weather(city)
        content = f"""
# 任务
农民的问题为：{user_ask}
农民所在地的天气为:{weather}
"""  + ("请结合图片内容进行回答。" + input_pic if input_pic != "" else "")
        # 带网络搜索的请求
        chat_completion = self.client.chat.completions.create(
            model="ernie-4.5-turbo-128k-preview",  # 使用支持搜索的模型
            messages=[
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': [
                    {"type": "text", "text": content}
                ]}
            ],
            extra_body={
                "web_search": {
                    "enable": True,
                    "enable_trace": True
                }
            },
        )

        # 提取搜索结果
        search_results = []
        if hasattr(chat_completion, 'search_results'):
            search_results = chat_completion.search_results

        # 打印回复内容
        answer = chat_completion.choices[0].message.content

        # 打印参考资料
        if search_results:
            print("\n参考资料：")
            unique_dict = {}
            for item in search_results:
                unique_dict[item["index"]] = item

            for result in list(unique_dict.values()):
                print(f"{result['index']}. {result['title']}. {result['url']}")

        return answer

    def process_tea_disease_image(self, user_ask, city, file_path=None, file=None, sys_prompt=None):
        """
        输入文件位置，转换陈base64并进行图文理解
        """
        if sys_prompt is None:
            sys_prompt = self.disease_prompt + " 请查看图片，识别这是哪种茶树病害，并结合天气信息提供防治建议，需要指明使用的方法等。请使用简单明了，通俗易懂的中文回答,使得文化程度不高的人也能看懂。"
        # 创建基本的openai，TODO：apikay需要修改

        base64_img, file_ext = base64_encode(file_path=file_path, file=file)
        image_url = f"data:image/{file_ext};base64,{base64_img}"

        weather = get_weather(city)
        try:
            # Make API call
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": sys_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text":
                                f"""## 天气信息:
        {weather}
        
        ## 用户输入:
        {user_ask}"""},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                model="ernie-4.5-turbo-vl",
            )

            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"API调用错误: {str(e)}"


    def cloud_process_big_small(self, user_ask, city, file_path=None, file=None, yolo_text_file=None, yolo_text=None):
        """
        输入文件位置，转换base64并进行图文理解
        """
        if yolo_text_file is not None and yolo_text is None:
            with open(yolo_text_file, "r", encoding="utf-8") as f:
                yolo_text = f.read().strip()

        print_info(user_ask, city, yolo_text)
        vl_response = self.process_tea_disease_image("""请根据系统提示词的病情表格，尽可能详细描述图片中的内容，以帮助文本大模型更好的理解当前发生的病虫害种类与状况。
                                                     你需要将输出的内容分为两行，第一行指明当前病虫害是什么，其他内容均不要输出，第二行开始阐述当前病虫害发生的状况。无关内容均不要输出
                                                     """ + (f"yolo标注结果为:{yolo_text}，如果你觉得结果不对，请进行反驳的证明" if yolo_text is not None else ""),
                                                     city,
                                                     file_path=file_path,
                                                     file=file,
                                                     )
        # print_info("vl_response:", vl_response)
        # disease = vl_response.split("\n")[0].strip()
        # search_info = get_search(f"茶树{disease}的防治方法")
        res = self.only_text(user_ask, city, vl_response)
        return res

# Example usage
if __name__ == "__main__":
    # result = process_tea_disease_image(r"E:\python\create_model\archive\tea sickness dataset\algal leaf\UNADJUSTEDNONRAW_thumb_1.jpg", "这是什么病？", "福建泉州安溪")
    # print("模型回复:", result)
    all_response = OpenaiResponse()
    all_response.cloud_process_big_small("这是什么病？", "福建泉州安溪", file_path=r"E:\python\create_model\archive\tea sickness dataset\algal leaf\UNADJUSTEDNONRAW_thumb_1.jpg")
