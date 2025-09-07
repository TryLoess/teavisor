import paddle
from paddle.inference import Config, create_predictor
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
from .utils import get_base_dir, print_info

def predict_image(infer, file_name, dst_name, class_names, conf_threshold=0.45):
    """file_path只需要输入文件名即可，所有pic都需要存入pic文件夹下统一转换读取，file"""
    print_info("直接预测")
    result_img, _, boxes, scores, class_ids = infer.predict(get_base_dir() + "/data/pic/" + file_name, conf_threshold=conf_threshold)

    # Save the annotated image
    cv2.imwrite(get_base_dir() + "/data/pic/" + dst_name, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    for i, box in enumerate(boxes):
        class_id = class_ids[i]
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        print(f"检测到{class_name}置信度为{scores[i]:.2f}在{box}")
    return get_base_dir() + "/data/pic/" + dst_name

def predict_image_use_resize(infer, ori_file, resize_file, dst_name, class_names, conf_threshold=0.45):
    print_info("使用resize预测")
    result_img, boxes, scores, class_ids = infer.base_resize_predict(ori_file, resize_file, conf_threshold=conf_threshold)
    # cv2.imwrite(get_base_dir() + "/data/pic/" + dst_name, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    for i, box in enumerate(boxes):
        class_id = class_ids[i]
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        print(f"检测到{class_name}置信度为{scores[i]:.2f}在{box}")
    # return get_base_dir() + "/data/pic/" + dst_name
    return result_img # cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

def preprocess(image_path):
    """
    使用letterbox方法进行预处理，保持宽高比
    """
    # 读取照片
    img = cv2.imread(image_path)
    assert img is not None, f"Image not found: {image_path}"

    # 取出原始shape
    original_h, original_w = img.shape[:2]
    original_size = (original_h, original_w)

    # 转为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 计算letterbox参数
    ratio = min(640 / original_w, 640 / original_h)
    new_w, new_h = int(original_w * ratio), int(original_h * ratio)

    # 计算填充
    pad_x, pad_y = (640 - new_w) // 2, (640 - new_h) // 2

    # 创建letterbox画布
    letterbox_img = np.full((640, 640, 3), 114, dtype=np.uint8)  # 使用灰色填充

    # 调整图像尺寸
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 将调整大小的图像放到画布中央
    letterbox_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_img

    # 标准化处理
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_normalized = letterbox_img.astype(np.float32) / 255.0
    img_normalized = (img_normalized - mean) / std

    # 转换维度 HWC -> CHW
    img_transposed = np.transpose(img_normalized, (2, 0, 1))

    # 加入batch维度
    img_input = np.expand_dims(img_transposed, axis=0)

    # 保存预处理信息
    preprocess_info = {
        "original_size": original_size,
        "new_size": (new_h, new_w),
        "pad": (pad_y, pad_x),
        "ratio": ratio
    }

    # 对于兼容性，仍然返回scale_factor，但实际上我们会使用preprocess_info
    scale_factor = np.array([640.0 / original_w, 640.0 / original_h]).astype(np.float32)
    scale_factor = np.expand_dims(scale_factor, axis=0)

    return img_input, scale_factor, img, preprocess_info

def visualize_detection(image, boxes, scores, class_ids, class_names, scale_factor, threshold=0.5,
                                  font_path=None):
    """
    优化后的目标检测可视化函数

    Args:
        image: 原始图像
        boxes: 模型输出的边界框 (在640x640尺度上)
        scores: 置信度分数
        class_ids: 类别ID
        class_names: 类别名称列表
        scale_factor: 缩放因子 [scale_x, scale_y]
        threshold: 显示阈值
        font_path: 字体路径
    """
    img_copy = image.copy()
    h, w = img_copy.shape[:2]

    # 提取缩放因子
    scale_x, scale_y = scale_factor[0]

    # 计算letterbox参数 - 确定填充区域
    # 在原始图像中宽高的实际比例
    original_aspect_ratio = float(w) / float(h)

    # 计算在保持宽高比的情况下，图像在640x640中的实际尺寸
    if original_aspect_ratio > 1:  # 宽大于高 (横向图像)
        # 宽度会被缩放到640，高度会更小
        new_w = 640
        new_h = int(640 / original_aspect_ratio)
        pad_x = 0
        pad_y = (640 - new_h) // 2
    else:  # 高大于宽 (纵向图像)
        # 高度会被缩放到640，宽度会更小
        new_h = 640
        new_w = int(640 * original_aspect_ratio)
        pad_x = (640 - new_w) // 2
        pad_y = 0

    # 实际缩放比例 (从原始尺寸到letterbox尺寸的比例)
    ratio_w = new_w / w
    ratio_h = new_h / h

    # 转换为PIL图像
    pil_img = Image.fromarray(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 绘制每个检测结果
    for box, score, class_id in zip(boxes, scores, class_ids):
        if score < threshold:
            continue

        class_id = int(class_id)
        class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown-{class_id}"

        # 将坐标从640x640映射回原始图片尺寸
        x1, y1, x2, y2 = box

        # 1. 首先移除padding (考虑letterbox填充)
        x1 = x1 - pad_x
        y1 = y1 - pad_y
        x2 = x2 - pad_x
        y2 = y2 - pad_y

        # 2. 将无填充的坐标映射回原始图像尺寸
        x1 = max(0, int(x1 / ratio_w))
        y1 = max(0, int(y1 / ratio_h))
        x2 = min(w, int(x2 / ratio_w))
        y2 = min(h, int(y2 / ratio_h))

        # 确保坐标在图片范围内
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # 如果框太小，可能是误检，跳过
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue

        # Generate color based on class id
        color_bgr = (
            (class_id * 100) % 255,
            (class_id * 50) % 255,
            (class_id * 150) % 255
        )
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # BGR to RGB for PIL

        # 绘制矩形框
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.rectangle(cv2_img, (x1, y1), (x2, y2), color_bgr, 4)
        pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Create label with class name and score
        label = f"{class_name}: {score:.2f}"

        # 获取文本尺寸
        font = ImageFont.truetype(font_path, 20) if font_path else ImageFont.load_default()
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


def visualize_middle(input_image, boxes, scores, class_ids, class_names, threshold=0.5,
                                font_path=None):
    if len(input_image.shape) == 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # 创建一个可视化用的副本
    img_viz = input_image.copy()

    # 转换为PIL图像用于绘制
    pil_img = Image.fromarray(img_viz)
    draw = ImageDraw.Draw(pil_img)


    font = ImageFont.truetype(font_path, 15)
    # 在图像上绘制每个检测框
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        if score < threshold:
            continue

        # 确保class_id是整数
        class_id = int(class_id)

        # 获取类别名称
        class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown-{class_id}"

        # 获取边界框坐标
        x1, y1, x2, y2 = box

        # 确保坐标是整数
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 检查坐标是否超出640×640范围
        is_out_of_bounds = (x1 < 0 or y1 < 0 or x2 > 640 or y2 > 640)

        # 生成颜色 (不同类别不同颜色)
        color_rgb = (
            (class_id * 100) % 255,
            (class_id * 50) % 255,
            (class_id * 150) % 255
        )

        # 如果超出边界，使用红色
        if is_out_of_bounds:
            color_rgb = (255, 0, 0)  # 红色表示超出范围的框

        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=2)

        # 创建标签
        label = f"{class_name}: {score:.2f} [Raw]"
        if is_out_of_bounds:
            label += " (Out of bounds!)"

        # 获取文本尺寸
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 确保标签位置在图片内
        label_y = y1 - text_height - 5
        if label_y < 0:
            label_y = y2 + 5

        label_x = x1
        if label_x + text_width > 640:
            label_x = 640 - text_width

        # 绘制标签背景
        draw.rectangle(
            [label_x, label_y, label_x + text_width, label_y + text_height],
            fill=color_rgb
        )

        # 绘制标签文本
        draw.text((label_x, label_y), label, fill=(255, 255, 255), font=font)

    # 绘制640×640的边界
    draw.rectangle([0, 0, 639, 639], outline=(0, 255, 0), width=2)

    # 添加说明文本
    info_text = "Model Input Space (640x640)"
    bbox = draw.textbbox((0, 0), info_text, font=font)
    text_width = bbox[2] - bbox[0]

    draw.rectangle(
        [10, 10, 10 + text_width, 10 + 20],
        fill=(0, 0, 0)
    )
    draw.text((10, 10), info_text, fill=(0, 255, 0), font=font)

    # 将PIL图像转回OpenCV格式
    result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # # 添加边界框统计信息
    # total_boxes = len([s for s in scores if s >= threshold])
    # out_of_bounds = sum(1 for box in boxes if (box[0] < 0 or box[1] < 0 or box[2] > 640 or box[3] > 640) and scores[
    #     list(boxes).index(box)] >= threshold)

    # cv2.putText(
    #     result_img,
    #     f"Total: {total_boxes}, Out of bounds: {out_of_bounds}",
    #     (10, 640 - 10),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5,
    #     (0, 255, 0),
    #     1
    # )

    return result_img

def resize_image(image_path):
    """固定写入debug文件，返回debug图片路径"""
    img = cv2.imread(image_path)
    assert img is not None, f"Image not found: {image_path}"

    # 取出原始shape
    original_h, original_w = img.shape[:2]

    # 转为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(get_base_dir() + "/debug_original.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    ratio = min(640 / original_w, 640 / original_h)
    new_w, new_h = int(original_w * ratio), int(original_h * ratio)

    # 创建640x640的画布并填充
    canvas = np.ones((640, 640, 3), dtype=np.uint8) * 114  # 使用灰色填充

    # 将调整大小的图像放在画布中央
    offset_x, offset_y = (640 - new_w) // 2, (640 - new_h) // 2
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_img
    cv2.imwrite(get_base_dir() + "/data/pic/debug_resized.jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return get_base_dir() + "/data/pic/debug_resized.jpg"

def box_iou(box1, box2, score1, score2, thereshould=0.5):
    """
    计算两个方框的 IoU（交并比）
    box1, box2: [x1, y1, x2, y2]
    thereshould: IoU 阈值,大于阈值将会被淘汰
    返回: iou 值，范围 0~1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 重叠区域
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # 各自面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # 并集面积
    union_area = area1 + area2 - inter_area  # 正常情况下都要大于0的，不大报错也正常

    iou = inter_area / union_area  # 可能会报错除以0，但是报错了也说明要检查了、
    print_info("当前方框的iou为", iou)
    if iou > thereshould:
        if score1 > score2:
            return box2  # 代表了列表中的box会被删掉
        return False  # 代表当前的box会被删掉
    return True  # 代表save
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
        img_input, scale_factor, original_img, _ = preprocess(image_path)
        img_to_save = img_input[0]  # shape: (3, 640, 640)
        img_to_save = np.transpose(img_to_save, (1, 2, 0))  # shape: (640, 640, 3)
        img_to_save = (img_to_save * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(get_base_dir() + "/debug_input.jpg", cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(get_base_dir() + "/debug_input.jpg", cv2.cvtColor(np.transpose(img_input[0], (1, 2, 0)), cv2.COLOR_RGB2BGR))
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
        middle_img = visualize_middle(
            img_to_save,
            boxes,
            scores,
            class_ids,
            self.class_names,
            conf_threshold,
            get_base_dir() + "/data/chinese.otf"  # 中文字体路径
        )
        return result_img, middle_img, boxes, scores, class_ids

    def base_resize_predict(self, original_img_path, resize_img_path, conf_threshold=0.5):
        """
        进行推断
        Returns:
            Annotated image, boxes, scores, class_ids
        """
        original_img = cv2.imread(original_img_path)  # 加载原始图片
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        # Preprocess image
        img_input, scale_factor, _, _ = preprocess(resize_img_path)  # 对resize的图片进行处理

        img_to_save = img_input[0]  # shape: (3, 640, 640)
        img_to_save = np.transpose(img_to_save, (1, 2, 0))  # shape: (640, 640, 3)
        img_to_save = (img_to_save * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(get_base_dir() + "/debug_input.jpg", cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(get_base_dir() + "/debug_input.jpg", cv2.cvtColor(np.transpose(img_input[0], (1, 2, 0)), cv2.COLOR_RGB2BGR))
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
        print_info("Scores output shape:", scores_data.shape)  # 这里只有一个值，就是第一个维度的数量

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
                print_info("当前的box为", box, "score为", score, "class_id为", class_id)
                # 判断iou
                res_box = True  # 三种情况，True保留所有box，False删除当前box，box删除列表中的某个box
                for j in range(len(boxes)):
                    res_box = box_iou(box, boxes[j], score, scores[j], thereshould=0.5)
                    if not isinstance(res_box, bool):
                        print_info("这个效果不好的框框被删除")
                        boxes.remove(boxes[j])  # 相当于把这个框框删除
                        scores.remove(scores[j])
                        class_ids.remove(class_ids[j])
                        break
                print_info(boxes)
                if res_box is not False:  # 只要不是False就说明要加入
                    print_info("当前的box为", box, "score为", score, "class_id为", class_id)
                    boxes.append(box)
                    scores.append(score)
                    class_ids.append(class_id)
                print_info(boxes)



        # Visualize detections
        print_info("将会打框的图片为", original_img.shape)
        result_img = visualize_detection(
            original_img,  # 这里输入原始图像进行打标注
            boxes,
            scores,
            class_ids,
            self.class_names,
            scale_factor,
            conf_threshold,
            get_base_dir() + "/data/chinese.otf"  # 中文字体路径
        )
        print_info("最后图片的shape为", result_img.shape)
        return result_img, boxes, scores, class_ids
