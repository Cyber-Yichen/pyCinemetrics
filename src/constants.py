"""
pyCinemetrics 常量配置
集中管理所有魔法数字和配置参数
"""

# ── OpenPose 景别识别 ─────────────────────────────────────────────────────
OPENPOSE_NET_HEIGHT = 368
OPENPOSE_THRESHOLD = 0.1
OPENPOSE_PAF_SCORE_THRESHOLD = 0.1
OPENPOSE_CONFIDENCE_THRESHOLD = 0.7
OPENPOSE_INTERP_SAMPLES = 15

# ── TransNetV2 分镜切割 ─────────────────────────────────────────────────
TRANSNET_INPUT_SIZE = (27, 48, 3)
TRANSNET_WINDOW_SIZE = 100
TRANSNET_STRIDE = 50
TRANSNET_SCENE_THRESHOLD = 0.5

# ── 色彩分析 ──────────────────────────────────────────────────────────────
COLOR_MAX_POINTS = 200
COLOR_DEFAULT_CLUSTERS = 5

# ── 文件路径 ──────────────────────────────────────────────────────────────
FRAME_SUBDIR = "frame"
FRAME_PREFIX = "frame"
OUTPUT_CSV_SHOTSCALE = "shotscale.csv"
OUTPUT_CSV_OBJECTS = "objects.csv"
OUTPUT_CSV_COLORS = "colors.csv"
OUTPUT_PNG_SHOTSCALE = "shotscale.png"
OUTPUT_PNG_OBJECTS = "objects.png"
OUTPUT_PNG_COLORS = "colors.png"

# ── 图像预处理 ─────────────────────────────────────────────────────────────
IMAGE_RESIZE_SIZE = 256
IMAGE_CROP_SIZE = 224
IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZE_STD = [0.229, 0.224, 0.225]

# ── 景别分类 ──────────────────────────────────────────────────────────────
SHOT_SCALE_TYPES = {
    "EMPTY": "Empty Shot",
    "LONG": "Long Shot",
    "FULL": "Full Shot",
    "MEDIUM": "Medium Shot",
    "MEDIUM_CLOSE": "Medium Close-Up",
    "CLOSE": "Close-Up",
}

HEAD_PARTS = {'Nose', 'REye', 'LEye', 'REar', 'LEar'}
CHEST_BELOW_PARTS = {'MidHip', 'RHip', 'LHip', 'RKnee', 'LKnee'}
FEET_PARTS = {'RAnkle', 'LAnkle', 'RHeel', 'LHeel', 'RBigToe', 'LBigToe', 'RSmallToe', 'LSmallToe'}

# ── 版本信息 ──────────────────────────────────────────────────────────────
VERSION = "0.2.0"
APP_NAME = "CCKS Cinemetrics"
