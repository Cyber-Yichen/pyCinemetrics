"""
色彩分析模块
使用 K-Means 提取视频帧的主要颜色
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.cluster.vq import vq, kmeans
import math

from src.constants import (
    COLOR_MAX_POINTS,
    COLOR_DEFAULT_CLUSTERS,
    OUTPUT_CSV_COLORS,
    FRAME_SUBDIR,
)
from src.logger import get_logger
from src.algorithms.resultsave import resultsave

logger = get_logger(__name__)


class ColorAnalysis:
    """色彩分析类"""

    def __init__(self, filename: str):
        """
        初始化色彩分析

        Args:
            filename: 图像文件路径
        """
        self.filename = Path(filename)

    def load_image(self) -> List[Tuple[int, ...]]:
        """
        加载图像并提取颜色点

        Returns:
            颜色点列表
        """
        try:
            img = Image.open(self.filename)
            img = img.rotate(-90)
            img.thumbnail((COLOR_MAX_POINTS, COLOR_MAX_POINTS))
            w, h = img.size

            points = []
            for count, color in img.getcolors(w * h):
                points.append(color)

            return points

        except Exception as e:
            logger.error("Failed to load image %s: %s", self.filename, e)
            return []

    def kmeans(self, imgdata: List[Tuple[int, ...]], n: int) -> np.ndarray:
        """
        K-Means 聚类提取主要颜色

        Args:
            imgdata: 颜色数据
            n: 聚类数量

        Returns:
            聚类中心
        """
        data = np.array(imgdata, dtype=float)
        centers, _ = kmeans(data, n)
        return np.array(centers, dtype=int)

    def calculate_distances(self, centers: np.ndarray) -> List[List[int]]:
        """
        计算最近的真实颜色

        Args:
            centers: 聚类中心

        Returns:
            最近的真实颜色列表
        """
        imgdata = self.load_image()
        result = []

        for center in centers:
            min_dist = float('inf')
            closest_color = None

            for color in imgdata:
                dist = math.sqrt(
                    (color[0] - center[0]) ** 2 +
                    (color[1] - center[1]) ** 2 +
                    (color[2] - center[2]) ** 2
                )
                if dist < min_dist:
                    min_dist = dist
                    closest_color = list(color)

            if closest_color:
                result.append(closest_color)

        return result

    @staticmethod
    def rgb_to_hex(colors: List[List[int]]) -> List[str]:
        """
        RGB 转 HEX

        Args:
            colors: RGB 颜色列表

        Returns:
            HEX 颜色列表
        """
        hex_colors = []
        for color in colors:
            hex_color = '#'
            for value in color:
                hex_color += f"{value:02X}"
            hex_colors.append(hex_color)
        return hex_colors

    def img_colors(self, imgpath: str, colors_c: int = COLOR_DEFAULT_CLUSTERS) -> None:
        """
        批量分析图像目录的色彩

        Args:
            imgpath: 图像目录路径
            colors_c: 聚类数量
        """
        frame_dir = Path("img") / imgpath / FRAME_SUBDIR
        if not frame_dir.exists():
            logger.error("Frame directory not found: %s", frame_dir)
            return

        colorlist = []
        allrealcolors = []
        allcolors = []

        image_extensions = {'.jpg', '.png', '.bmp'}

        for img_file in sorted(frame_dir.iterdir()):
            if img_file.suffix.lower() not in image_extensions:
                continue

            try:
                self.filename = img_file
                imgdata = self.load_image()

                if len(imgdata) < 6:
                    realcolor = [list(imgdata[0])] * COLOR_DEFAULT_CLUSTERS
                else:
                    colors = self.kmeans(imgdata, colors_c)
                    realcolor = self.calculate_distances(colors)

                allrealcolors += realcolor
                allcolors.append(list(realcolor))

                colors_array = np.array(realcolor).reshape(1, 3 * colors_c)
                colorlist.append([img_file.name, colors_array[0]])

            except Exception as e:
                logger.error("Failed to process %s: %s", img_file, e)
                continue

        # 保存结果
        save_path = Path("img") / imgpath
        rs = resultsave(str(save_path))
        rs.color_csv(colorlist)
        rs.plot_scatter_3d(allcolors)

        logger.info("Color analysis completed for %d frames", len(colorlist))

    def analysis_single(self, imgpath: str, color_c: int = COLOR_DEFAULT_CLUSTERS) -> None:
        """
        分析单张图像的色彩

        Args:
            imgpath: 图像路径
            color_c: 聚类数量
        """
        self.filename = Path(imgpath)
        imgdata = self.load_image()

        if not imgdata:
            logger.error("No color data extracted from %s", imgpath)
            return

        if len(imgdata) < 6:
            realcolor = [list(imgdata[0])] * COLOR_DEFAULT_CLUSTERS
        else:
            colors = self.kmeans(imgdata, color_c)
            realcolor = self.calculate_distances(colors)

        color_hex = self.rgb_to_hex(realcolor)
        self._draw_pie(imgdata, realcolor, color_hex)

    def _draw_pie(
        self,
        imgdata: List[Tuple[int, ...]],
        colors: List[List[int]],
        colors_hex: List[str],
    ) -> None:
        """
        绘制饼图

        Args:
            imgdata: 原始颜色数据
            colors: 聚类颜色
            colors_hex: HEX 颜色
        """
        try:
            cluster, _ = vq(imgdata, colors)
            result = Counter(cluster.tolist())

            plt.style.use("dark_background")
            plt.pie(
                x=[result.get(i, 0) for i in range(len(colors))],
                colors=colors_hex,
                wedgeprops=dict(width=0.2, edgecolor='w'),
                labels=colors_hex,
                autopct='%1.2f%%',
            )

            output_path = self.filename.parent / 'colortmp.png'
            plt.savefig(str(output_path))
            plt.close()

            logger.info("Saved pie chart to %s", output_path)

        except Exception as e:
            logger.error("Failed to draw pie chart: %s", e)
