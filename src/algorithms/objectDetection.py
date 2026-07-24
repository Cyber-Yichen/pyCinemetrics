"""
物体检测模块
使用 VGG19 识别视频帧中的物体
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter

import csv
import numpy as np
from PIL import Image
import torch
import torch.nn
import torchvision.models as models
from torchvision import transforms

from src.constants import (
    IMAGE_RESIZE_SIZE,
    IMAGE_CROP_SIZE,
    IMAGE_NORMALIZE_MEAN,
    IMAGE_NORMALIZE_STD,
    OUTPUT_CSV_OBJECTS,
    FRAME_SUBDIR,
)
from src.logger import get_logger
from src.algorithms.wordcloud2frame import WordCloud2Frame

logger = get_logger(__name__)


class ObjectDetection:
    """物体检测类"""

    # ImageNet 类别文件路径
    CLASSES_FILE = Path(__file__).parent / "imagenet_classes.txt"

    def __init__(self, image_path: str):
        """
        初始化物体检测

        Args:
            image_path: 图像目录路径
        """
        self.image_path = Path(image_path)
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_RESIZE_SIZE),
            transforms.CenterCrop(IMAGE_CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGE_NORMALIZE_MEAN,
                std=IMAGE_NORMALIZE_STD
            )
        ])

    def make_model(self) -> torch.nn.Module:
        """
        创建 VGG19 模型

        Returns:
            VGG19 模型实例
        """
        logger.info("Loading VGG19 model")
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        model = model.eval()

        if torch.cuda.is_available():
            model.cuda()
            logger.info("Using CUDA for inference")

        return model

    def object_detection(self) -> None:
        """执行物体检测"""
        model = self.make_model()

        if not self.image_path:
            logger.error("Image path is empty")
            return

        frame_dir = self.image_path / FRAME_SUBDIR
        if not frame_dir.exists():
            logger.error("Frame directory not found: %s", frame_dir)
            return

        # 加载类别标签
        if not self.CLASSES_FILE.exists():
            logger.error("Classes file not found: %s", self.CLASSES_FILE)
            return

        with open(self.CLASSES_FILE, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines()]

        framelist: List[Tuple[str, str]] = []
        image_extensions = {'.jpg', '.png', '.bmp'}

        # 遍历所有帧
        for file_name in sorted(frame_dir.iterdir()):
            if file_name.suffix.lower() not in image_extensions:
                continue

            try:
                img_t = self.transform(Image.open(file_name))

                if torch.cuda.is_available():
                    batch_t = torch.unsqueeze(img_t, 0).cuda()
                else:
                    batch_t = torch.unsqueeze(img_t, 0)

                with torch.no_grad():
                    out = model(batch_t)

                _, indices = torch.sort(out, descending=True)
                percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

                # 获取 Top1 结果
                top_idx = indices[0][0].item()
                frame_id = file_name.stem.replace(FRAME_PREFIX, "")
                framelist.append((frame_id, classes[top_idx]))

            except Exception as e:
                logger.error("Failed to process %s: %s", file_name, e)
                continue

        logger.info("Detected objects in %d frames", len(framelist))

        # 保存结果
        self._save_csv(framelist)

    def _save_csv(self, framelist: List[Tuple[str, str]]) -> None:
        """
        保存检测结果为 CSV

        Args:
            framelist: 帧列表 [(frame_id, object_name), ...]
        """
        csv_path = self.image_path / OUTPUT_CSV_OBJECTS

        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['FrameId', 'Top1-Objects'])

                for frame_id, obj_name in framelist:
                    writer.writerow([frame_id, obj_name])

            logger.info("Saved objects CSV to %s", csv_path)

        except Exception as e:
            logger.error("Failed to save CSV: %s", e)
            return

        # 生成词云
        try:
            wc2f = WordCloud2Frame()
            tf = wc2f.wordfrequency(str(csv_path))
            wc2f.plotwordcloud(tf, str(self.image_path), "/objects")
            logger.info("Generated word cloud")
        except Exception as e:
            logger.error("Failed to generate word cloud: %s", e)
