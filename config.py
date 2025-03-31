from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Source
SOURCES_LIST = ["图片", "视频", "摄像头"]

# DL model config
DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
INSTANCE_SEGMENTATION_MODEL_DIR = ROOT / 'weights' / 'instance_segmentation'
CLASSIFICATION_MODEL_DIR = ROOT / 'weights' / 'classification'
YOLOv8n = DETECTION_MODEL_DIR / "yolov8n.pt"
YOLOv8s = DETECTION_MODEL_DIR / "yolov8s.pt"
YOLOv8m = DETECTION_MODEL_DIR / "yolov8m.pt"
YOLOv8l = DETECTION_MODEL_DIR / "yolov8l.pt"
YOLOv8x = DETECTION_MODEL_DIR / "yolov8x.pt"

DETECTION_MODEL_LIST = [
    "目标检测.pt","实例分割.pt","图像分类.pt"]

INSTANCE_SEGMENTATION_MODEL_LIST = [
     "目标检测.pt","实例分割.pt","图像分类.pt"]
DEFAULT_INSTANCE_SEGMENTATION_MODEL = "实例分割.pt"

CLASSIFICATION_MODEL_LIST = [
    "目标检测.pt","实例分割.pt","图像分类.pt"]
DEFAULT_CLASSIFICATION_MODEL = "图像分类.pt"