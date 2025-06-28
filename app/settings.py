from pathlib import Path

# Direktori dasar dari proyek (biasanya /backend)
BASE_DIR = Path(__file__).resolve().parent.parent

# Direktori model
MODEL_DIR = BASE_DIR / "model"

# Path model deteksi wajah dan klasifikasi ketertarikan
YOLO_MODEL_PATH: Path = MODEL_DIR / "best.pt"
RESNET_MODEL_PATH: Path = MODEL_DIR / "resnet50v2_finetuned_tingkat_ketertarikan.h5"

# Direktori sementara untuk penyimpanan file input dan hasil gambar dari analisis image
TMP_DIR: Path = BASE_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Direktori output video (untuk hasil frame-frame video)
OUTPUT_VIDEO_DIR: Path = BASE_DIR / "output_video"
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Threshold confidence untuk deteksi wajah dengan YOLO
FACE_CONFIDENCE_THRESHOLD: float = 0.75
