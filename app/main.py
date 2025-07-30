import  uuid, shutil
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .analyzer import FaceInterestAnalyzer
from .settings import YOLO_MODEL_PATH, RESNET_MODEL_PATH, TMP_DIR, OUTPUT_VIDEO_DIR

app = FastAPI(
    title="Deteksi Wajah dan Pengenalan Ekspresi Siswa SD untuk Mengukur Ketertarikan Terhadap Mata Pelajaran",
    description="Model YOLOV8 dan ResNet50",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/output", StaticFiles(directory=str(TMP_DIR)), name="output")
app.mount("/output_video", StaticFiles(directory=str(OUTPUT_VIDEO_DIR)), name="output_video")

analyzer: FaceInterestAnalyzer | None = None

@app.on_event("startup")
def load_models():
    global analyzer
    analyzer = FaceInterestAnalyzer(
        yolo_model_path=YOLO_MODEL_PATH,
        resnet_model_path=RESNET_MODEL_PATH
    )

@app.get("/", tags=["health"])
def health():
    return {"status": "ok"}

@app.post("/analyze/image", tags=["analyze"])
async def analyze_image(
    file: UploadFile = File(...),
    save_crops: bool = False
):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(415, detail="File must be JPEG or PNG")

    raw_bytes = await file.read()
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")

    tmp_name = f"{uuid.uuid4().hex}.jpg"
    tmp_path: Path = TMP_DIR / tmp_name
    cv2.imwrite(str(tmp_path), img)

    try:
        result = analyzer.analyze_image(
            image_path=str(tmp_path),
            save_crops=save_crops,
            output_dir=str(TMP_DIR)
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    result["analyzed_filename"] = f"{Path(result['image_path']).stem}_analyzed.jpg"
    result["analyzed_url"] = f"/output/{result['analyzed_filename']}"

    return JSONResponse(result)

@app.post("/analyze/video", tags=["analyze"])
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    interval_sec: int = 3,
    save_crops: bool = False
):
    if file.content_type not in ["video/mp4", "video/avi"]:
        raise HTTPException(415, detail="Unsupported video format")

    tmp_video: Path = TMP_DIR / f"{uuid.uuid4()}_{file.filename}"
    with tmp_video.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        frames_result = analyzer.analyze_video(
            video_path=str(tmp_video),
            interval_sec=interval_sec,
            save_crops=save_crops
        )
    finally:
        tmp_video.unlink(missing_ok=True)

    return JSONResponse(frames_result)
