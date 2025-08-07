from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "model"

YOLO_MODEL_PATH: Path = MODEL_DIR / "best.pt"
RESNET_MODEL_PATH: Path = MODEL_DIR / "resnet50v2_finetuned_tingkat_ketertarikan.h5"

TMP_DIR: Path = BASE_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO_DIR: Path = BASE_DIR / "output_video"
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

FACE_CONFIDENCE_THRESHOLD: float = 0.75
