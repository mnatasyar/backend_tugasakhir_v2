from ultralytics import YOLO
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from datetime import datetime

from .settings import FACE_CONFIDENCE_THRESHOLD, OUTPUT_VIDEO_DIR

class FaceInterestAnalyzer:
    def __init__(self, yolo_model_path, resnet_model_path):
        self.yolo_model = YOLO(yolo_model_path)
        self.resnet_model = load_model(resnet_model_path)

        self.interest_labels = ['netral', 'tertarik', 'tidak_tertarik']
        self.colors = {
            'netral': (255, 255, 0),
            'tertarik': (0, 255, 0),
            'tidak_tertarik': (0, 0, 255)
        }
        self.face_confidence_threshold = FACE_CONFIDENCE_THRESHOLD

    def preprocess_face_for_resnet(self, face_crop):
        face_resized = cv2.resize(face_crop, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype('float32') / 255.0
        face_batch = np.expand_dims(face_normalized, axis=0)
        return face_batch

    def classify_interest(self, face_crop):
        try:
            processed_face = self.preprocess_face_for_resnet(face_crop)
            predictions = self.resnet_model.predict(processed_face, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_label = self.interest_labels[predicted_class_idx]
            return predicted_label, confidence, predictions[0]
        except Exception as e:
            print(f"Error dalam klasifikasi: {e}")
            return "error", 0.0, None

    def analyze_image(self, image_path, save_crops=True, output_dir=None, frame_subfolder=None):
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        subfolder_path = os.path.join(output_dir, frame_subfolder) if output_dir and frame_subfolder else output_dir
        output_path = os.path.join(subfolder_path, f"{name}_analyzed{ext}")

        crops_folder = os.path.join(subfolder_path, "face_crops") if save_crops else None
        if save_crops and not os.path.exists(crops_folder):
            os.makedirs(crops_folder)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Tidak dapat membaca gambar {image_path}")
            return None

        results = self.yolo_model(image, conf=self.face_confidence_threshold)
        analysis_results = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'total_faces': 0,
            'faces': []
        }

        face_count = 0
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_confidence = float(box.conf[0])
                face_crop = image[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                face_count += 1
                interest_label, interest_confidence, all_predictions = self.classify_interest(face_crop)

                if save_crops:
                    crop_filename = os.path.join(crops_folder, f"face_{face_count}_{interest_label}_{interest_confidence:.2f}.jpg")
                    cv2.imwrite(crop_filename, face_crop)

                color = self.colors.get(interest_label, (255, 255, 255))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

                label = f"Face {face_count}: {interest_label}"
                confidence_text = f"Interest: {interest_confidence:.2f}"
                face_conf_text = f"Face: {face_confidence:.2f}"

                texts = [label, confidence_text, face_conf_text]
                max_width = max([cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for text in texts])
                cv2.rectangle(image, (x1, y1 - 80), (x1 + max_width + 10, y1), color, -1)

                y_offset = y1 - 60
                for text in texts:
                    cv2.putText(image, text, (x1 + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    y_offset += 20

                face_result = {
                    'face_id': face_count,
                    'bbox': [x1, y1, x2, y2],
                    'face_confidence': face_confidence,
                    'interest_label': interest_label,
                    'interest_confidence': interest_confidence,
                    'all_predictions': {
                        'netral': float(all_predictions[0]) if all_predictions is not None else 0,
                        'tertarik': float(all_predictions[1]) if all_predictions is not None else 0,
                        'tidak_tertarik': float(all_predictions[2]) if all_predictions is not None else 0
                    }
                }
                analysis_results['faces'].append(face_result)

        analysis_results['total_faces'] = face_count
        cv2.imwrite(output_path, image)
        analysis_results['analyzed_filename'] = os.path.basename(output_path)
        return analysis_results

    def analyze_video(self, video_path, interval_sec=3, save_crops=True):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        target_times = np.arange(0, duration, interval_sec)

        folder_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = os.path.join(str(OUTPUT_VIDEO_DIR), folder_name)
        os.makedirs(output_dir, exist_ok=True)

        analysis_summary = []

        for idx, t in enumerate(target_times):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                print(f"[WARNING] Tidak bisa membaca frame pada detik ke-{round(t, 2)}.")
                continue

            frame_folder = f"frame_{idx + 1}"
            frame_dir = os.path.join(output_dir, frame_folder)
            os.makedirs(frame_dir, exist_ok=True)

            frame_name = f"frame_{idx:03d}.jpg"
            temp_path = os.path.join(frame_dir, frame_name)
            cv2.imwrite(temp_path, frame)

            print(f"[INFO] Menganalisis frame pada detik ke-{round(t, 2)}...")

            result = self.analyze_image(
                image_path=temp_path,
                save_crops=save_crops,
                output_dir=output_dir,
                frame_subfolder=frame_folder
            )

            if result:
                result['frame_index'] = int(t * fps)
                result['frame_time_sec'] = round(t, 2)
                result['total_faces'] = result.get('total_faces', 0)

                frame_summary = {}
                for face in result['faces']:
                    label = face['interest_label']
                    frame_summary[label] = frame_summary.get(label, 0) + 1
                result['frame_summary'] = frame_summary

                analyzed_filename = f"{os.path.splitext(frame_name)[0]}_analyzed.jpg"
                result['output_image'] = f"{folder_name}/{frame_folder}/{analyzed_filename}"

                analysis_summary.append(result)

        cap.release()
        return analysis_summary
