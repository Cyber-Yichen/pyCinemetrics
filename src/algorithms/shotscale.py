"""
景别识别模块
使用 OpenPose 检测骨骼点，判断拍摄景别
"""

import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter

import cv2
import time
import math
import numpy as np
from matplotlib import pyplot as plt

from src.constants import (
    OPENPOSE_NET_HEIGHT,
    OPENPOSE_THRESHOLD,
    OPENPOSE_PAF_SCORE_THRESHOLD,
    OPENPOSE_CONFIDENCE_THRESHOLD,
    OPENPOSE_INTERP_SAMPLES,
    HEAD_PARTS,
    CHEST_BELOW_PARTS,
    FEET_PARTS,
    SHOT_SCALE_TYPES,
    OUTPUT_CSV_SHOTSCALE,
    OUTPUT_PNG_SHOTSCALE,
)
from src.logger import get_logger
from src.algorithms.shotscaleconfig import *

logger = get_logger(__name__)


class ShotScale:
    """景别识别类"""

    def __init__(self, keypoint_num: int = 25):
        """
        初始化景别识别

        Args:
            keypoint_num: 关键点数量（25 或 18）
        """
        if keypoint_num not in (25, 18):
            raise ValueError("keypoint_num must be 25 or 18")

        self.keypoint_num = keypoint_num

        if keypoint_num == 25:
            self.point_names = point_name_25
            self.point_pairs = point_pairs_25
            self.map_idx = map_idx_25
            self.colors = colors_25
            self.prototxt = prototxt_25
            self.caffemodel = caffemodel_25
        else:
            self.point_names = point_names_18
            self.point_pairs = point_pairs_18
            self.map_idx = map_idx_18
            self.colors = colors_18
            self.prototxt = prototxt_18
            self.caffemodel = caffemodel_18

        self.num_points = keypoint_num
        self.pose_net = self._load_model()

    def _load_model(self) -> cv2.dnn.Net:
        """
        加载 OpenPose 模型

        Returns:
            OpenCV DNN 网络
        """
        logger.info("Loading OpenPose model (%d keypoints)", self.keypoint_num)
        net = cv2.dnn.readNetFromCaffe(self.prototxt, self.caffemodel)
        return net

    def predict(self, imgfile: str) -> Tuple[np.ndarray, str, int]:
        """
        预测图像的景别

        Args:
            imgfile: 图像文件路径

        Returns:
            (annotated_image, shot_type, person_count) 元组
        """
        img = cv2.imread(imgfile)
        if img is None:
            raise IOError(f"Failed to read image: {imgfile}")

        height, width, _ = img.shape
        net_width = int((OPENPOSE_NET_HEIGHT / height) * width)
        start_time = time.time()

        # 前向传播
        in_blob = cv2.dnn.blobFromImage(
            img, 1.0 / 255, (net_width, OPENPOSE_NET_HEIGHT),
            (0, 0, 0), swapRB=False, crop=False
        )
        self.pose_net.setInput(in_blob)
        output = self.pose_net.forward()

        # 检测关键点
        detected_keypoints = []
        points_table = []
        keypoints_list = np.zeros((0, 3))
        keypoint_id = 0

        for part in range(self.num_points):
            prob_map = output[0, part, :, :]
            prob_map = cv2.resize(prob_map, (width, height))

            keypoints = self._get_keypoints(prob_map, OPENPOSE_THRESHOLD)

            for kp in keypoints:
                points_table.append(kp + (keypoint_id,) + (self.point_names[part],))
                keypoints_list = np.vstack([keypoints_list, kp])
                keypoint_id += 1

            detected_keypoints.append([kp + (i,) for i, kp in enumerate(keypoints)])

        # 获取有效配对
        valid_pairs, invalid_pairs = self._get_valid_pairs(
            output, detected_keypoints, width, height
        )

        # 构建人体姿态
        personwise_keypoints = self._get_personwise_keypoints(
            valid_pairs, invalid_pairs, keypoints_list
        )

        # 可视化
        img = self._visualize_pose(imgfile, personwise_keypoints, keypoints_list)

        # 检测关键人物
        key_parts, min_y, max_y = self._detect_key_person(
            personwise_keypoints, points_table
        )

        # 判断景别
        shot_type = self._classify_shot_scale(key_parts, min_y, max_y, height)

        # 添加信息到图像
        fps = math.ceil(1 / (time.time() - start_time))
        img = cv2.putText(img, f"FPS:{fps}", (25, 50),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        img = cv2.putText(img, f"ShotSize:{shot_type}", (25, 100),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img, shot_type, len(personwise_keypoints)

    def _get_keypoints(
        self, prob_map: np.ndarray, threshold: float
    ) -> List[Tuple[int, int, float]]:
        """
        从概率图中提取关键点

        Args:
            prob_map: 概率图
            threshold: 阈值

        Returns:
            关键点列表 [(x, y, confidence), ...]
        """
        map_smooth = cv2.GaussianBlur(prob_map, (3, 3), 0, 0)
        map_mask = np.uint8(map_smooth > threshold)
        keypoints = []

        contours, _ = cv2.findContours(
            map_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            blob_mask = np.zeros(map_mask.shape)
            blob_mask = cv2.fillConvexPoly(blob_mask, cnt, 1)
            masked_prob = map_smooth * blob_mask
            _, max_val, _, max_loc = cv2.minMaxLoc(masked_prob)
            keypoints.append(max_loc + (prob_map[max_loc[1], max_loc[0]],))

        return keypoints

    def _get_valid_pairs(
        self,
        output: np.ndarray,
        detected_keypoints: List[List[Tuple]],
        width: int,
        height: int,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """获取有效关键点配对"""
        valid_pairs = []
        invalid_pairs = []

        for k in range(len(self.map_idx)):
            paf_a = cv2.resize(output[0, self.map_idx[k][0], :, :], (width, height))
            paf_b = cv2.resize(output[0, self.map_idx[k][1], :, :], (width, height))

            cand_a = detected_keypoints[self.point_pairs[k][0]]
            cand_b = detected_keypoints[self.point_pairs[k][1]]

            n_a, n_b = len(cand_a), len(cand_b)

            if n_a == 0 or n_b == 0:
                invalid_pairs.append(k)
                valid_pairs.append(np.array([]))
                continue

            valid_pair = np.zeros((0, 3))

            for i in range(n_a):
                max_j = -1
                max_score = -1

                for j in range(n_b):
                    d_ij = np.subtract(cand_b[j][:2], cand_a[i][:2])
                    norm = np.linalg.norm(d_ij)

                    if norm == 0:
                        continue

                    d_ij = d_ij / norm

                    # 插值
                    interp_coords = list(zip(
                        np.linspace(cand_a[i][0], cand_b[j][0], num=OPENPOSE_INTERP_SAMPLES),
                        np.linspace(cand_a[i][1], cand_b[j][1], num=OPENPOSE_INTERP_SAMPLES)
                    ))

                    # 获取 PAF 分数
                    paf_scores = []
                    for coord in interp_coords:
                        x, y = int(round(coord[0])), int(round(coord[1]))
                        paf_scores.append([
                            paf_a[y, x],
                            paf_b[y, x]
                        ])

                    paf_dot = np.dot(paf_scores, d_ij)
                    avg_score = np.mean(paf_dot)

                    if (len(np.where(paf_dot > OPENPOSE_PAF_SCORE_THRESHOLD)[0]) /
                            OPENPOSE_INTERP_SAMPLES) > OPENPOSE_CONFIDENCE_THRESHOLD:
                        if avg_score > max_score:
                            max_j = j
                            max_score = avg_score

                if max_j >= 0:
                    valid_pair = np.append(
                        valid_pair,
                        [[cand_a[i][3], cand_b[max_j][3], max_score]],
                        axis=0
                    )

            valid_pairs.append(valid_pair)

        return valid_pairs, invalid_pairs

    def _get_personwise_keypoints(
        self,
        valid_pairs: List[np.ndarray],
        invalid_pairs: List[int],
        keypoints_list: np.ndarray,
    ) -> np.ndarray:
        """构建人体姿态关键点"""
        personwise = -1 * np.ones((0, self.num_points + 1))

        for k in range(len(self.map_idx)):
            if k in invalid_pairs:
                continue

            if len(valid_pairs[k]) == 0:
                continue

            part_as = valid_pairs[k][:, 0]
            part_bs = valid_pairs[k][:, 1]
            index_a, index_b = np.array(self.point_pairs[k])

            for i in range(len(valid_pairs[k])):
                found = False
                person_idx = -1

                for j in range(len(personwise)):
                    if personwise[j][index_a] == part_as[i]:
                        person_idx = j
                        found = True
                        break

                if found:
                    personwise[person_idx][index_b] = part_bs[i]
                    personwise[person_idx][-1] += (
                        keypoints_list[int(part_bs[i]), 2] +
                        valid_pairs[k][i][2]
                    )
                elif k < self.num_points - 1:
                    row = -1 * np.ones(self.num_points + 1)
                    row[index_a] = part_as[i]
                    row[index_b] = part_bs[i]
                    row[-1] = (
                        sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) +
                        valid_pairs[k][i][2]
                    )
                    personwise = np.vstack([personwise, row])

        return personwise

    def _visualize_pose(
        self,
        img_file: str,
        personwise: np.ndarray,
        keypoints_list: np.ndarray,
    ) -> np.ndarray:
        """可视化骨骼连接"""
        img = cv2.imread(img_file)

        for i in range(self.num_points - 1):
            for n in range(len(personwise)):
                index = personwise[n][np.array(self.point_pairs[i])]
                if -1 in index:
                    continue

                x = np.int32(keypoints_list[index.astype(int), 0])
                y = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(img, (x[0], y[0]), (x[1], y[1]), self.colors[i], 3, cv2.LINE_AA)

        return img

    def _detect_key_person(
        self,
        personwise: np.ndarray,
        points_table: List[Tuple],
    ) -> Tuple[Optional[List[str]], Optional[int], Optional[int]]:
        """检测关键人物（面积最大）"""
        max_area = 0
        key_person_index = -1
        key_parts = []
        min_y, max_y = 0, 0

        for i, person_kp in enumerate(personwise):
            x_coords, y_coords, parts = [], [], []

            for j in range(self.num_points):
                value = int(person_kp[j])
                if value == -1:
                    continue

                for point in points_table:
                    if point[3] == value:
                        x_coords.append(point[0])
                        y_coords.append(point[1])
                        parts.append(point[4])

            if not x_coords:
                continue

            area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))

            if area > max_area:
                max_area = area
                key_person_index = i
                key_parts = parts
                min_y = min(y_coords)
                max_y = max(y_coords)

        if key_person_index == -1:
            return None, None, None

        return key_parts, min_y, max_y

    def _classify_shot_scale(
        self,
        key_parts: Optional[List[str]],
        min_y: Optional[int],
        max_y: Optional[int],
        height: int,
    ) -> str:
        """
        分类景别

        Args:
            key_parts: 检测到的身体部位
            min_y: 最小 y 坐标
            max_y: 最大 y 坐标
            height: 图像高度

        Returns:
            景别类型
        """
        if key_parts is None:
            return SHOT_SCALE_TYPES["EMPTY"]

        body_parts = set(key_parts)

        # 检测头部
        has_head = bool(body_parts.intersection(HEAD_PARTS))
        # 检测胸部以下
        has_chest_below = bool(body_parts.intersection(CHEST_BELOW_PARTS))
        # 检测脚部
        has_feet = bool(body_parts.intersection(FEET_PARTS))
        # 检测颈部
        has_neck = 'Neck' in body_parts

        if has_head and has_feet:
            if (max_y - min_y) <= height / 2:
                return SHOT_SCALE_TYPES["LONG"]
            else:
                return SHOT_SCALE_TYPES["FULL"]
        elif has_head and not has_feet and has_chest_below:
            return SHOT_SCALE_TYPES["MEDIUM"]
        elif has_head and not has_chest_below and has_neck:
            return SHOT_SCALE_TYPES["MEDIUM_CLOSE"]
        elif has_head and not has_neck:
            return SHOT_SCALE_TYPES["CLOSE"]
        else:
            return SHOT_SCALE_TYPES["CLOSE"]

    def save_csv(
        self,
        detect_info: List[Tuple[str, str, int]],
        save_path: str,
    ) -> None:
        """
        保存检测结果为 CSV

        Args:
            detect_info: 检测结果 [(frame_id, shot_type, person_count), ...]
            save_path: 保存路径
        """
        csv_path = Path(save_path) / OUTPUT_CSV_SHOTSCALE

        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['FrameId', 'ShotScale', 'Detect_Person_Num'])

                for frame_id, shot_type, person_count in detect_info:
                    writer.writerow([frame_id, shot_type, person_count])

            logger.info("Saved shot scale CSV to %s", csv_path)

        except Exception as e:
            logger.error("Failed to save CSV: %s", e)

    def save_plot(
        self,
        detect_info: List[Tuple[str, str, int]],
        image_save: str,
    ) -> None:
        """
        保存景别分布图

        Args:
            detect_info: 检测结果
            image_save: 保存路径
        """
        categories = [item[1] for item in detect_info]
        category_counts = Counter(categories)
        total = len(categories)

        sizes = [(count / total) * 100 for count in category_counts.values()]
        labels = list(category_counts.keys())

        try:
            plt.clf()
            plt.style.use('dark_background')

            inner_radius = 0.5
            width = 0.3

            plt.pie(
                sizes,
                labels=labels,
                autopct='%0.1f%%',
                shadow=True,
                pctdistance=0.5,
                wedgeprops=dict(width=width, edgecolor='w')
            )

            centre_circle = plt.Circle((0, 0), inner_radius, fc='black')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)

            plt.title('Shot Scale')
            plt.axis('equal')

            output_path = Path(image_save) / OUTPUT_PNG_SHOTSCALE
            plt.savefig(str(output_path))
            plt.close()

            logger.info("Saved shot scale plot to %s", output_path)

        except Exception as e:
            logger.error("Failed to save plot: %s", e)
