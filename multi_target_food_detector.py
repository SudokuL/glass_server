# -*- coding: gbk -*-
"""
多目标中国菜检测 + 手势识别系统
结合YOLO目标检测、现有中国菜分类模型和MediaPipe手势识别
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import platform

# 添加food_cv目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'food_cv'))

@dataclass
class FoodDetection:
    """食物检测结果"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    dish_name: str
    nutrients: Dict
    crop_image: np.ndarray
    yolo_class: str

@dataclass
class HandGesture:
    """手势识别结果"""
    landmark_points: List[Tuple[int, int]]
    pointing_position: Tuple[int, int]
    confidence: float

class MultiTargetFoodDetector:
    """多目标中国菜检测器"""
    
    def __init__(self, 
                 yolo_model_path: str = "yolov8n.pt",
                 chinese_food_model_path: str = None,
                 silent_mode: bool = False):
        """
        初始化检测器
        
        Args:
            yolo_model_path: YOLO模型路径
            chinese_food_model_path: 中国菜分类模型路径
            silent_mode: 静默模式，True时不输出日志
        """
        self.silent_mode = silent_mode
        
        if not self.silent_mode:
            print("[MultiTarget] 正在初始化多目标食物检测器...")
        
        # 1. 加载YOLO模型（用于食物目标检测）
        if not self.silent_mode:
            print("[MultiTarget] 正在加载YOLO模型...")
        try:
            self.yolo_model = YOLO(yolo_model_path)
            if not self.silent_mode:
                print(f"[MultiTarget] YOLO模型加载成功: {yolo_model_path}")
        except Exception as e:
            if not self.silent_mode:
                print(f"[MultiTarget] YOLO模型加载失败: {e}")
                print("[MultiTarget] 尝试下载默认模型...")
            self.yolo_model = YOLO("yolov8n.pt")
        
        # 2. 加载您现有的中国菜识别模型
        if chinese_food_model_path:
            try:
                from GatedRegNet_Food import FoodRecognizer
                self.chinese_food_model = FoodRecognizer()
                if not self.silent_mode:
                    print("[MultiTarget] 中国菜识别模型加载成功")
            except Exception as e:
                if not self.silent_mode:
                    print(f"[MultiTarget] 中国菜识别模型加载失败: {e}")
                self.chinese_food_model = None
        else:
            try:
                from GatedRegNet_Food import FoodRecognizer
                self.chinese_food_model = FoodRecognizer()
                if not self.silent_mode:
                    print("[MultiTarget] 中国菜识别模型加载成功")
            except Exception as e:
                if not self.silent_mode:
                    print(f"[MultiTarget] 中国菜识别模型加载失败: {e}")
                self.chinese_food_model = None
        
        # 3. 初始化MediaPipe手势识别
        if not self.silent_mode:
            print("[MultiTarget] 正在初始化手势识别...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,  # 静态图像模式
            max_num_hands=2,  # 最多检测2只手
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 4. 食物类别映射（YOLO COCO数据集中的真实食物类别）
        # 参考: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
        self.food_classes = {
            45: "orange",      # 橙子
            46: "banana",      # 香蕉
            47: "apple",       # 苹果
            48: "sandwich",    # 三明治
            49: "orange",      # 橙子（重复）
            50: "broccoli",    # 西兰花
            51: "carrot",      # 胡萝卜
            52: "hot dog",     # 热狗
            53: "pizza",       # 披萨
            54: "donut",       # 甜甜圈
            55: "cake",        # 蛋糕
            56: "chair",       # 椅子（非食物，但可能与餐桌场景相关）
            60: "dining table", # 餐桌
            61: "toilet",      # 马桶（非食物）
            62: "tv",          # 电视（非食物）
            63: "laptop",      # 笔记本电脑（非食物）
            64: "mouse",       # 鼠标（非食物）
            65: "remote",      # 遥控器（非食物）
            66: "keyboard",    # 键盘（非食物）
            67: "cell phone"   # 手机（非食物）
        }
        
        # 只保留真正的食物类别
        self.food_classes = {
            45: "orange",
            46: "banana", 
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake"
        }
        
        if not self.silent_mode:
            print("[MultiTarget] 多目标食物检测器初始化完成！")
    
    def detect_food_objects(self, image: np.ndarray) -> List[Dict]:
        """
        使用YOLO检测图像中的食物目标
        
        Args:
            image: 输入图像
            
        Returns:
            检测到的食物目标列表
        """
        if not self.silent_mode:
            print("[MultiTarget] 正在进行YOLO食物目标检测...")
        
        # YOLO推理
        results = self.yolo_model(image, verbose=False)
        
        food_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取类别ID和置信度
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # 只保留食物相关的检测结果，置信度阈值设为0.3
                    if class_id in self.food_classes and confidence > 0.3:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # 确保边界框在图像范围内
                        h, w = image.shape[:2]
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)
                        
                        # 计算边界框面积和图像面积
                        bbox_area = (x2 - x1) * (y2 - y1)
                        image_area = h * w
                        area_ratio = bbox_area / image_area
                        
                        # 过滤条件：有效边界框 + 面积不能超过图像的80%
                        if x2 > x1 and y2 > y1 and area_ratio < 0.8:
                            food_detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'yolo_class': self.food_classes[class_id],
                                'class_id': class_id
                            })
        
        if not self.silent_mode:
            print(f"[MultiTarget] 检测到 {len(food_detections)} 个食物目标")
        return food_detections
    
    def classify_chinese_food(self, crop_image: np.ndarray) -> Dict:
        """
        使用您现有的中国菜分类模型进行精细分类
        
        Args:
            crop_image: 裁剪的食物图像
            
        Returns:
            中国菜分类结果
        """
        if self.chinese_food_model is None:
            return {
                'dish_name': '未知食品',
                'confidence': 0.0,
                'nutrients': {},
                'top3_predictions': []
            }
        
        try:
            # 将numpy数组转换为PIL图像
            from PIL import Image
            pil_image = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
            
            # 保存临时文件
            temp_path = "/tmp/temp_food_crop.jpg"
            pil_image.save(temp_path)
            
            # 调用您现有的中国菜识别模型
            result = self.chinese_food_model.recognize_food(temp_path)
            
            # 删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if result:
                return result
            else:
                return {
                    'dish_name': '未知食品',
                    'confidence': 0.0,
                    'nutrients': {},
                    'top3_predictions': []
                }
                
        except Exception as e:
            if not self.silent_mode:
                print(f"[MultiTarget] 中国菜分类失败: {e}")
            return {
                'dish_name': '未知食品',
                'confidence': 0.0,
                'nutrients': {},
                'top3_predictions': []
            }
    
    def detect_hand_gesture(self, image: np.ndarray) -> Optional[HandGesture]:
        """
        检测手势并获取指向位置
        
        Args:
            image: 输入图像
            
        Returns:
            手势检测结果
        """
        if not self.silent_mode:
            print("[MultiTarget] 正在进行手势识别...")
        
        # 转换为RGB格式（MediaPipe需要RGB）
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            if not self.silent_mode:
                print(f"[MultiTarget] 检测到 {len(results.multi_hand_landmarks)} 只手")
            
            # 选择第一只手进行分析
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 获取关键点坐标
            h, w = image.shape[:2]
            landmark_points = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmark_points.append((x, y))
            
            # 计算指向位置（使用食指尖端，索引8）
            if len(landmark_points) > 8:
                pointing_position = landmark_points[8]
                
                # 计算手势置信度（基于手部关键点的稳定性）
                confidence = 0.8
                
                if not self.silent_mode:
                    print(f"[MultiTarget] 手势指向位置: {pointing_position}")
                
                return HandGesture(
                    landmark_points=landmark_points,
                    pointing_position=pointing_position,
                    confidence=confidence
                )
        else:
            if not self.silent_mode:
                print("[MultiTarget] 未检测到手势")
        
        return None
    
    def match_gesture_to_food(self, 
                             gesture: HandGesture, 
                             food_detections: List[Dict]) -> Optional[Dict]:
        """
        将手势指向位置与检测到的食物进行匹配
        
        Args:
            gesture: 手势检测结果
            food_detections: 食物检测结果列表
            
        Returns:
            匹配的食物检测结果
        """
        if not gesture or not food_detections:
            return None
        
        pointing_x, pointing_y = gesture.pointing_position
        if not self.silent_mode:
            print(f"[MultiTarget] 正在匹配手势指向位置 ({pointing_x}, {pointing_y}) 与食物...")
        
        # 找到包含指向点的食物边界框
        for food_detection in food_detections:
            x1, y1, x2, y2 = food_detection['bbox']
            
            # 检查指向点是否在边界框内
            if x1 <= pointing_x <= x2 and y1 <= pointing_y <= y2:
                dish_name = food_detection.get('chinese_food_result', {}).get('dish_name', '未知')
                if not self.silent_mode:
                    print(f"[MultiTarget] 手势直接指向食物: {dish_name}")
                return food_detection
        
        # 如果没有直接匹配，找最近的食物
        min_distance = float('inf')
        closest_food = None
        
        for food_detection in food_detections:
            x1, y1, x2, y2 = food_detection['bbox']
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            distance = np.sqrt((pointing_x - center_x)**2 + (pointing_y - center_y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_food = food_detection
        
        # 如果最近的食物在合理距离内，返回它
        if closest_food and min_distance < 150:  # 可调整的阈值
            dish_name = closest_food.get('chinese_food_result', {}).get('dish_name', '未知')
            if not self.silent_mode:
                print(f"[MultiTarget] 手势指向最近的食物: {dish_name} (距离: {min_distance:.1f}px)")
            return closest_food
        
        if not self.silent_mode:
            print(f"[MultiTarget] 手势未指向任何食物 (最近距离: {min_distance:.1f}px)")
        return None
    
    def process_frame(self, image: np.ndarray) -> Dict:
        """
        处理单帧图像，完整的检测流程
        
        Args:
            image: 输入图像
            
        Returns:
            完整的检测结果
        """
        if not self.silent_mode:
            print("[MultiTarget] 开始处理图像...")
        
        # 1. 食物目标检测
        food_detections = self.detect_food_objects(image)
        
        # 2. 手势识别
        gesture = self.detect_hand_gesture(image)
        
        # 3. 对检测到的食物进行中国菜分类
        enhanced_food_detections = []
        for i, food_detection in enumerate(food_detections):
            if not self.silent_mode:
                print(f"[MultiTarget] 正在分析第 {i+1}/{len(food_detections)} 个食物目标...")
            
            x1, y1, x2, y2 = food_detection['bbox']
            
            # 裁剪食物区域
            crop_image = image[y1:y2, x1:x2]
            
            if crop_image.size > 0:
                # 中国菜精细分类
                chinese_food_result = self.classify_chinese_food(crop_image)
                
                # 合并结果
                enhanced_detection = {
                    **food_detection,
                    'chinese_food_result': chinese_food_result,
                    'crop_image': crop_image
                }
                enhanced_food_detections.append(enhanced_detection)
                
                # 使用UTF-8编码打印，避免终端乱码
                if not self.silent_mode:
                    try:
                        print(f"[MultiTarget] 食物 {i+1}: {chinese_food_result['dish_name']} (置信度: {chinese_food_result['confidence']:.3f})")
                    except UnicodeEncodeError:
                        dish_name_safe = chinese_food_result['dish_name'].encode('utf-8', 'ignore').decode('utf-8')
                        print(f"[MultiTarget] 食物 {i+1}: {dish_name_safe} (置信度: {chinese_food_result['confidence']:.3f})")
        
        # 4. 手势与食物匹配
        selected_food = None
        if gesture:
            selected_food = self.match_gesture_to_food(gesture, enhanced_food_detections)
        
        result = {
            'food_detections': enhanced_food_detections,
            'gesture': gesture,
            'selected_food': selected_food,
            'total_detected': len(enhanced_food_detections)
        }
        
        if not self.silent_mode:
            print(f"[MultiTarget] 处理完成！检测到 {len(enhanced_food_detections)} 个食物目标")
        if selected_food:
            if not self.silent_mode:
                try:
                    print(f"[MultiTarget] 手势选中的食物: {selected_food['chinese_food_result']['dish_name']}")
                except UnicodeEncodeError:
                    dish_name_safe = selected_food['chinese_food_result']['dish_name'].encode('utf-8', 'ignore').decode('utf-8')
                    print(f"[MultiTarget] 手势选中的食物: {dish_name_safe}")
        
        return result
    
    def _get_chinese_font(self, size=20):
        """
        获取中文字体，解决中文显示乱码问题
        """
        try:
            # 尝试不同的中文字体路径，优先使用CJK字体
            font_paths = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux CJK
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",     # Linux CJK Bold
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/System/Library/Fonts/Arial.ttf",  # macOS
                "C:/Windows/Fonts/simhei.ttf",  # Windows
                "C:/Windows/Fonts/msyh.ttf",   # Windows微软雅黑
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    if not self.silent_mode:
                        print(f"[MultiTarget] 使用字体: {font_path}")
                    return ImageFont.truetype(font_path, size)
            
            # 如果都找不到，使用默认字体
            if not self.silent_mode:
                print("[MultiTarget] 警告: 未找到中文字体，使用默认字体")
            return ImageFont.load_default()
        except Exception as e:
            if not self.silent_mode:
                print(f"[MultiTarget] 字体加载错误: {e}")
            return ImageFont.load_default()
    
    def _draw_chinese_text(self, image, text, position, font_size=20, color=(255, 255, 255)):
        """
        在图像上绘制中文文字
        """
        # 转换为PIL图像
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # 获取字体
        font = self._get_chinese_font(font_size)
        
        # 绘制文字
        draw.text(position, text, font=font, fill=color)
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def visualize_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            results: 检测结果
            
        Returns:
            标注后的图像
        """
        if not self.silent_mode:
            print("[MultiTarget] 正在生成可视化结果...")
        
        annotated_image = image.copy()
        
        # 1. 绘制食物检测框
        for i, food_detection in enumerate(results['food_detections']):
            x1, y1, x2, y2 = food_detection['bbox']
            confidence = food_detection['confidence']
            chinese_result = food_detection['chinese_food_result']
            yolo_class = food_detection['yolo_class']
            
            # 检测框颜色
            if food_detection == results['selected_food']:
                color = (0, 0, 255)  # 选中的食物用红色
                thickness = 3
            else:
                color = (0, 255, 0)  # 其他食物用绿色
                thickness = 2
            
            # 绘制边界框
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制标签背景矩形
            label = f"{i+1}. {chinese_result['dish_name']} ({confidence:.2f})"
            # 估算文字大小（中文字符按2个字符宽度计算）
            text_width = len(label.encode('gbk')) * 8
            text_height = 25
            cv2.rectangle(annotated_image, (x1, y1-text_height-5), 
                         (x1+text_width, y1), color, -1)
            
            # 使用PIL绘制中文标签（黑色字体）
            text_color = (0, 0, 0)  # 黑色字体
            annotated_image = self._draw_chinese_text(annotated_image, label, 
                                                    (x1, y1-text_height), 16, text_color)
            
            # 绘制YOLO类别信息（英文，可以用OpenCV）
            yolo_label = f"YOLO: {yolo_class}"
            cv2.putText(annotated_image, yolo_label, (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # 2. 绘制手势
        if results['gesture']:
            # 定义手部连接关系（MediaPipe手部21个关键点的连接）
            hand_connections = [
                # 拇指
                (0, 1), (1, 2), (2, 3), (3, 4),
                # 食指
                (0, 5), (5, 6), (6, 7), (7, 8),
                # 中指
                (0, 9), (9, 10), (10, 11), (11, 12),
                # 无名指
                (0, 13), (13, 14), (14, 15), (15, 16),
                # 小指
                (0, 17), (17, 18), (18, 19), (19, 20),
                # 手掌连接
                (5, 9), (9, 13), (13, 17)
            ]
            
            landmark_points = results['gesture'].landmark_points
            
            # 绘制手部骨架连线
            for connection in hand_connections:
                if connection[0] < len(landmark_points) and connection[1] < len(landmark_points):
                    pt1 = landmark_points[connection[0]]
                    pt2 = landmark_points[connection[1]]
                    cv2.line(annotated_image, pt1, pt2, (0, 255, 255), 2)  # 青色连线
            
            # 绘制手部关键点
            for i, point in enumerate(landmark_points):
                if i == 8:  # 食指尖特殊标记
                    cv2.circle(annotated_image, point, 5, (0, 0, 255), -1)  # 红色
                else:
                    cv2.circle(annotated_image, point, 3, (255, 0, 0), -1)  # 蓝色
            
            # 突出显示指向点
            pointing_pos = results['gesture'].pointing_position
            cv2.circle(annotated_image, pointing_pos, 8, (255, 255, 0), 2)
            
            # 使用PIL绘制中文"指向"文字
            annotated_image = self._draw_chinese_text(annotated_image, "指向", 
                                                    (pointing_pos[0]+10, pointing_pos[1]-10), 
                                                    16, (255, 255, 0))
            
            # 绘制指向线（从食指到选中的食物）
            if results['selected_food']:
                selected = results['selected_food']
                x1, y1, x2, y2 = selected['bbox']
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.line(annotated_image, pointing_pos, (center_x, center_y), 
                         (255, 255, 0), 3)  # 加粗连线
                
                # 在连线中点添加箭头效果
                mid_x = (pointing_pos[0] + center_x) // 2
                mid_y = (pointing_pos[1] + center_y) // 2
                cv2.circle(annotated_image, (mid_x, mid_y), 5, (255, 255, 0), -1)
        
        # 3. 绘制统计信息
        stats_text = f"检测到 {results['total_detected']} 个食物目标"
        annotated_image = self._draw_chinese_text(annotated_image, stats_text, 
                                                (10, 10), 20, (255, 255, 255))
        
        if results['gesture']:
            gesture_text = "检测到手势"
            annotated_image = self._draw_chinese_text(annotated_image, gesture_text, 
                                                    (10, 40), 20, (255, 255, 0))
        
        return annotated_image

def main():
    """主函数 - 处理测试图片"""
    print("=== 多目标中国菜检测 + 手势识别系统 ===")
    
    # 初始化检测器
    detector = MultiTargetFoodDetector()
    
    # 处理测试图片
    test_image_path = "test2.jpg"
    if not os.path.exists(test_image_path):
        print(f"错误：找不到测试图片 {test_image_path}")
        return
    
    print(f"正在处理测试图片: {test_image_path}")
    image = cv2.imread(test_image_path)
    
    if image is None:
        print("错误：无法读取图片")
        return
    
    print(f"图片尺寸: {image.shape}")
    
    # 进行检测
    results = detector.process_frame(image)
    
    # 可视化结果
    annotated_image = detector.visualize_results(image, results)
    
    # 保存结果
    output_path = "result_multi_detection.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"结果已保存到: {output_path}")
    
    # 输出详细信息
    print("\n=== 检测结果详情 ===")
    print(f"总共检测到 {results['total_detected']} 个食物目标")
    
    for i, food in enumerate(results['food_detections']):
        print(f"\n食物 {i+1}:")
        print(f"  位置: {food['bbox']}")
        print(f"  YOLO类别: {food['yolo_class']}")
        try:
            print(f"  中国菜名称: {food['chinese_food_result']['dish_name']}")
        except UnicodeEncodeError:
            dish_name_safe = food['chinese_food_result']['dish_name'].encode('utf-8', 'ignore').decode('utf-8')
            print(f"  中国菜名称: {dish_name_safe}")
        print(f"  置信度: {food['chinese_food_result']['confidence']:.3f}")
        
        if food['chinese_food_result']['nutrients']:
            print(f"  主要营养信息:")
            nutrients = food['chinese_food_result']['nutrients']
            for key, value in list(nutrients.items())[:5]:  # 只显示前5个
                print(f"    {key}: {value:.2f}")
    
    if results['gesture']:
        print(f"\n手势检测:")
        print(f"  指向位置: {results['gesture'].pointing_position}")
        print(f"  置信度: {results['gesture'].confidence:.3f}")
    
    if results['selected_food']:
        selected = results['selected_food']
        print(f"\n手势选中的食物:")
        try:
            print(f"  名称: {selected['chinese_food_result']['dish_name']}")
        except UnicodeEncodeError:
            dish_name_safe = selected['chinese_food_result']['dish_name'].encode('utf-8', 'ignore').decode('utf-8')
            print(f"  名称: {dish_name_safe}")
        print(f"  位置: {selected['bbox']}")
        print(f"  置信度: {selected['chinese_food_result']['confidence']:.3f}")
    
    print(f"\n可视化结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
