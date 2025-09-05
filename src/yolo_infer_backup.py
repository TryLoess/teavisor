import paddle
from paddle.inference import Config, create_predictor
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
from .utils import get_base_dir, print_info


def preprocess(image_path):
    """
    Preprocess the image for YOLO model inference, matched with training preprocessing
    """
    # 读取照片，不能有中文名
    img = cv2.imread(image_path)
    assert img is not None, f"Image not found: {image_path}"

    # 取出原始shape
    origin_shape = img.shape[:2]

    # Convert to RGB (YOLO models typically expect RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 修改图片大小，统一使用训练时的目标尺寸 640x640
    img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

    # 使用与训练一致的均值和标准差进行标准化
    # NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # is_scale=True 表示先除以255再标准化
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - mean) / std

    # Permute: 转换维度 HWC -> CHW，与训练时一致
    img_transposed = np.transpose(img_normalized, (2, 0, 1))

    # 加入batch维度
    img_input = np.expand_dims(img_transposed, axis=0)

    # 计算缩放因子，用于后处理时将检测框映射回原图
    scale_factor = np.array([640.0 / origin_shape[1], 640.0 / origin_shape[0]]).astype(np.float32)
    scale_factor = np.expand_dims(scale_factor, axis=0)

    return img_input, scale_factor, img


def visualize_detection(image, boxes, scores, class_ids, class_names, scale_factor, threshold=0.5,
                                font_path=None):
    """
    支持中文类别名称的可视化函数

    Args:
        image: 原始图片 (未缩放的)
        boxes: 检测框坐标 (来自640x640模型输出)
        scores: 置信度分数
        class_ids: 类别ID
        class_names: 类别名称列表 (支持中文)
        scale_factor: 从preprocess函数返回的缩放因子 [scale_x, scale_y]
        threshold: 置信度阈值
        font_path: 中文字体文件路径
    """
    img_copy = image.copy()
    h, w = img_copy.shape[:2]

    # 提取缩放因子
    scale_x, scale_y = scale_factor[0]

    # 转换为PIL图像用于绘制中文
    pil_img = Image.fromarray(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 加载中文字体
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, 20)
        else:
            # 尝试系统默认中文字体
            font = ImageFont.truetype("simhei.ttf", 20)  # Windows
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 20)  # macOS
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)  # Linux
            except:
                font = ImageFont.load_default()  # 默认字体

    # Draw each detection above threshold
    for box, score, class_id in zip(boxes, scores, class_ids):
        if score < threshold:
            continue

        # Convert class_id to int and get class name
        class_id = int(class_id)
        class_name = class_names[class_id] if class_id <= len(class_names) else f"Unknown-{class_id}"  # 这里可能还有一个15

        # 将坐标从640x640缩放回原始图片尺寸
        x1, y1, x2, y2 = box
        print_info(box)
        x1 = int(x1)# / scale_x)
        y1 = int(y1)# / scale_y)
        x2 = int(x2)# / scale_x)
        y2 = int(y2)# / scale_y)
        print_info(x1, y1, x2, y2)

        # 确保坐标在图片范围内
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # Generate color based on class id
        color_bgr = (
            (class_id * 100) % 255,
            (class_id * 50) % 255,
            (class_id * 150) % 255
        )
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # BGR to RGB for PIL

        # 先转回OpenCV绘制矩形框（因为PIL绘制矩形比较麻烦）
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.rectangle(cv2_img, (x1, y1), (x2, y2), color_bgr, 4)
        pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Create label with class name and score
        label = f"{class_name}: {score:.2f}"

        # 获取文本尺寸
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 确保标签位置在图片内
        label_y = y1 - text_height - 10
        if label_y < 0:  # 如果标签会超出图片顶部
            label_y = y2 + 10  # 将标签放到检测框下方

        # 确保标签不会超出图片右边界
        label_x = x1
        if label_x + text_width > w:
            label_x = w - text_width

        # 确保标签不会超出图片底部
        if label_y + text_height > h:
            label_y = h - text_height - 10

        # Draw text background
        bg_x1 = label_x - 5
        bg_y1 = label_y - 5
        bg_x2 = label_x + text_width + 5
        bg_y2 = label_y + text_height + 5

        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color_rgb)

        # Draw text
        draw.text((label_x, label_y), label, fill=(255, 255, 255), font=font)
    result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result_img
class InferYOLO:
    def __init__(self, model_dir, class_names, use_gpu=False):
        """
        Initialize YOLO inference model
        Args:
            model_dir: Directory containing model.pdmodel and model.pdiparams
            class_names: List of class names for visualization
            use_gpu: Whether to use GPU for inference
        """

        # Load model files
        model_file = os.path.join(model_dir, "model.pdmodel")
        params_file = os.path.join(model_dir, "model.pdiparams")

        assert os.path.exists(model_file), f"没有这个pdmodel{model_file}"
        assert os.path.exists(params_file), f"没有这个pdiparams{params_file}"

        self.config = Config(model_file, params_file)

        if use_gpu:
            self.config.enable_use_gpu(1000, 0)  # 显存1000MB, GPU id 0
        else:
            self.config.disable_gpu()
            print_info("使用cpu预测")

        self.config.switch_ir_optim(True)
        self.config.enable_memory_optim()

        print_info("加载模型")
        self.predictor = create_predictor(self.config)
        print_info("模型成功加载")

        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()
        print_info("Inputs:", self.input_names)
        print_info("Outputs:", self.output_names)

        # Default class names if not provided
        self.class_names = class_names

    def predict(self, image_path, conf_threshold=0.5):
        """
        进行推断
        Returns:
            Annotated image, boxes, scores, class_ids
        """
        # Preprocess image
        img_input, scale_factor, original_img = preprocess(image_path)
        print_info("Input shape:", img_input.shape, "dtype:", img_input.dtype)

        # Set input tensor for image
        input_tensor_img = self.predictor.get_input_handle(self.input_names[0])
        input_tensor_img.copy_from_cpu(img_input)

        # Set input tensor for scale_factor if it exists in input_names
        if len(self.input_names) > 1 and 'scale_factor' in self.input_names:  #
            input_tensor_scale = self.predictor.get_input_handle('scale_factor')
            input_tensor_scale.copy_from_cpu(scale_factor)

        # Run inference
        self.predictor.run()

        # Get output - boxes, scores, num_boxes
        boxes_tensor = self.predictor.get_output_handle(self.output_names[0])
        boxes_data = boxes_tensor.copy_to_cpu()
        print_info("Boxes output shape:", boxes_data.shape)  # 这里有特别多框，且框的值不按照比例缩放，没有按照置信度排序

        # 对于pp_yoloe，输出格式有两种
        # [N, 6], where each row is [class_id, score, x1, y1, x2, y2],这个是对的
        # For models with separate outputs for boxes, scores, num_boxes
        scores_tensor = self.predictor.get_output_handle(self.output_names[1])  # 这里取出了'multiclass_nms3_0.tmp_2'
        scores_data = scores_tensor.copy_to_cpu()
        print("Scores output shape:", scores_data.shape)  # 这里只有一个值，就是第一个维度的数量

        # Extract valid detections
        boxes = []
        scores = []
        class_ids = []

        # Process based on output format (adjust as needed for your model)
        # print_info(boxes_data)
        for i in range(len(boxes_data)):
            box = boxes_data[i][-4:]  # x1, y1, x2, y2
            score = boxes_data[i][1]
            class_id = int(boxes_data[i][0])  # If class_id is in first position

            if score >= conf_threshold:
                boxes.append(box)
                scores.append(score)
                class_ids.append(class_id)

        # Visualize detections
        result_img = visualize_detection(
            original_img,
            boxes,
            scores,
            class_ids,
            self.class_names,
            scale_factor,
            conf_threshold,
            get_base_dir() + "/data/chinese.otf"  # 中文字体路径
        )

        return result_img, boxes, scores, class_ids

# import paddle
# from paddle.vision.models import ppyoloe
#
# # 导入 PPYOLOE 模型
# # 您可以选择 PPYOLOE_S, PPYOLOE_M, PPYOLOE_L 或 PPYOLOE_X
# model = ppyoloe.ppyoloe_s(pretrained=False, num_classes=80)
#
# # 加载模型参数
# params = paddle.load('model.pdparams')
# model.set_state_dict(params)
#
# # 设置为评估模式
# model.eval()

# 使用模型进行预测
# x = paddle.randn([1, 3, 640, 640])  # 示例输入
# output = model(x)