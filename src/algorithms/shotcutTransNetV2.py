"""
TransNetV2 分镜切割模块
使用深度学习模型将视频分割成镜头帧
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import tensorflow as tf
import cv2

from src.constants import (
    TRANSNET_INPUT_SIZE,
    TRANSNET_SCENE_THRESHOLD,
    FRAME_SUBDIR,
    FRAME_PREFIX,
)
from src.logger import get_logger

logger = get_logger(__name__)


class TransNetV2:
    """TransNetV2 视频分镜模型"""

    def __init__(self, model_dir: Optional[str] = None):
        """
        初始化 TransNetV2 模型

        Args:
            model_dir: 模型目录路径，默认使用项目内置模型

        Raises:
            FileNotFoundError: 模型目录不存在
            IOError: 模型加载失败
        """
        if model_dir is None:
            model_dir = str(
                Path(__file__).parent.parent.parent / "models" / "transnetv2-weights"
            )

        if not Path(model_dir).is_dir():
            raise FileNotFoundError(
                f"[TransNetV2] Model directory not found: {model_dir}"
            )

        logger.info("Loading TransNetV2 model from %s", model_dir)

        try:
            self._model = tf.saved_model.load(model_dir)
        except OSError as e:
            raise IOError(
                f"[TransNetV2] Failed to load model from {model_dir}. "
                "Files may be corrupted. Please re-download."
            ) from e

        logger.info("TransNetV2 model loaded successfully")

    def predict_raw(self, frames: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        原始预测（批量）

        Args:
            frames: 输入帧，形状 [batch, frames, height, width, 3]

        Returns:
            (single_frame_pred, all_frames_pred) 元组

        Raises:
            ValueError: 输入形状不正确
        """
        if len(frames.shape) != 5 or frames.shape[2:] != TRANSNET_INPUT_SIZE:
            raise ValueError(
                f"Input shape must be [batch, frames, {TRANSNET_INPUT_SIZE[0]}, "
                f"{TRANSNET_INPUT_SIZE[1]}, 3], got {frames.shape}"
            )

        frames = tf.cast(frames, tf.float32)
        logits, dict_ = self._model(frames)

        single_frame_pred = tf.sigmoid(logits)
        all_frames_pred = tf.sigmoid(dict_["many_hot"])

        return single_frame_pred, all_frames_pred

    def predict_frames(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测帧序列

        Args:
            frames: 输入帧，形状 [frames, height, width, 3]

        Returns:
            (single_frame_pred, all_frames_pred) 元组

        Raises:
            ValueError: 输入形状不正确
        """
        if len(frames.shape) != 4 or frames.shape[1:] != TRANSNET_INPUT_SIZE:
            raise ValueError(
                f"Input shape must be [frames, {TRANSNET_INPUT_SIZE[0]}, "
                f"{TRANSNET_INPUT_SIZE[1]}, 3], got {frames.shape}"
            )

        def input_iterator():
            """生成滑动窗口批次"""
            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start +
                [frames] +
                [end_frame] * no_padded_frames_end,
                0
            )

            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                yield padded_inputs[ptr:ptr + 100][np.newaxis]
                ptr += 50

        predictions = []
        total_frames = len(frames)

        for i, inp in enumerate(input_iterator()):
            single_frame_pred, all_frames_pred = self.predict_raw(inp)
            predictions.append((
                single_frame_pred.numpy()[0, 25:75, 0],
                all_frames_pred.numpy()[0, 25:75, 0]
            ))
            processed = min((i + 1) * 50, total_frames)
            logger.info("Processing frames: %d/%d", processed, total_frames)

        single_frame_pred = np.concatenate([s for s, _ in predictions])
        all_frames_pred = np.concatenate([a for _, a in predictions])

        return single_frame_pred[:total_frames], all_frames_pred[:total_frames]

    def predict_video(self, video_fn: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测视频文件

        Args:
            video_fn: 视频文件路径

        Returns:
            (video_frames, single_frame_predictions, all_frame_predictions) 元组

        Raises:
            ModuleNotFoundError: ffmpeg-python 未安装
            FileNotFoundError: 视频文件不存在
        """
        try:
            import ffmpeg
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "ffmpeg-python is required for predict_video(). "
                "Install with: pip install ffmpeg-python"
            )

        if not Path(video_fn).exists():
            raise FileNotFoundError(f"Video file not found: {video_fn}")

        logger.info("Extracting frames from %s", video_fn)

        video_stream, err = ffmpeg.input(video_fn).output(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
        ).run(capture_stdout=True, capture_stderr=True)

        video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
        single_pred, all_pred = self.predict_frames(video)

        return video, single_pred, all_pred

    @staticmethod
    def predictions_to_scenes(
        predictions: np.ndarray,
        threshold: float = TRANSNET_SCENE_THRESHOLD,
    ) -> np.ndarray:
        """
        将预测结果转换为场景列表

        Args:
            predictions: 预测结果
            threshold: 阈值

        Returns:
            场景列表，每个场景为 [start, end]
        """
        predictions = (predictions > threshold).astype(np.uint8)

        scenes: List[List[int]] = []
        t_prev = 0
        start = 0

        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t

        if t == 0:
            scenes.append([start, i])

        if not scenes:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

    @staticmethod
    def visualize_predictions(frames: np.ndarray, predictions):
        """可视化预测结果"""
        from PIL import Image, ImageDraw

        if isinstance(predictions, np.ndarray):
            predictions = [predictions]

        ih, iw, ic = frames.shape[1:]
        width = 25

        pad_with = width - len(frames) % width if len(frames) % width != 0 else 0
        frames = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions)), (0, 0)])
        predictions = [np.pad(x, (0, pad_with)) for x in predictions]
        height = len(frames) // width

        img = frames.reshape([height, width, ih + 1, iw + len(predictions), ic])
        img = np.concatenate(np.split(
            np.concatenate(np.split(img, height), axis=2)[0], width
        ), axis=2)[0, :-1]

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        for i, pred in enumerate(zip(*predictions)):
            x, y = i % width, i // width
            x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1

            for j, p in enumerate(pred):
                color = [0, 0, 0]
                color[(j + 1) % 3] = 255

                value = round(p * (ih - 1))
                if value != 0:
                    draw.line((x + j, y, x + j, y - value), fill=tuple(color), width=1)

        return img


def get_frame_number(file_path: str) -> List[int]:
    """
    从文本文件读取帧号

    Args:
        file_path: 文件路径

    Returns:
        帧号列表
    """
    frame_numbers = []

    with open(file_path, 'r') as f:
        for line in f:
            numbers = [int(n) for n in line.split()]
            if len(numbers) >= 2:
                frame_numbers.append(numbers[1])

    logger.info("Read %d frame numbers from %s", len(frame_numbers), file_path)
    return frame_numbers


def transNetV2_run(
    v_path: str,
    image_save: str,
    th: float = TRANSNET_SCENE_THRESHOLD,
) -> List[List[int]]:
    """
    运行 TransNetV2 分镜切割

    Args:
        v_path: 视频路径
        image_save: 图像保存目录
        th: 场景阈值

    Returns:
        镜头长度列表 [[start, end, length], ...]
    """
    # 加载模型
    model = TransNetV2()

    # 检查是否已处理
    if Path(f"{v_path}.predictions.txt").exists() or Path(f"{v_path}.scenes.txt").exists():
        logger.warning("Predictions already exist for %s, skipping", v_path)

    # 预测
    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(v_path)
    scenes = model.predictions_to_scenes(single_frame_predictions, threshold=th)

    # 保存场景信息
    scene_file = Path(image_save) / "scenes.txt"
    np.savetxt(str(scene_file), scenes, fmt="%d")

    # 读取帧号
    frame_numbers = get_frame_number(str(scene_file))
    if frame_numbers:
        frame_numbers.pop()  # 移除最后一个

    # 创建输出目录
    frame_save = Path(image_save) / FRAME_SUBDIR
    frame_save.mkdir(parents=True, exist_ok=True)

    # 清理旧帧
    for old_frame in frame_save.glob("*.png"):
        old_frame.unlink()

    # 提取关键帧
    cap = cv2.VideoCapture(v_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {v_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_len = len(str(frame_count))
    shot_len: List[List[int]] = []

    # 保存第一帧
    ret, first_frame = cap.read()
    if ret:
        frame_id = f"{0:0{frame_len}d}"
        cv2.imwrite(str(frame_save / f"{FRAME_PREFIX}{frame_id}.png"), first_frame)

    # 保存分镜帧
    start = 0
    for frame_num in frame_numbers:
        frame_num += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame_id = f"{frame_num:0{frame_len}d}"
            cv2.imwrite(str(frame_save / f"{FRAME_PREFIX}{frame_id}.png"), frame)
            shot_len.append([start, frame_num, frame_num - start])
            start = frame_num

    cap.release()
    logger.info("TransNetV2 completed: %d shots extracted", len(shot_len))
    return shot_len
