import os
import sys
import cv2
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

# -------------------------------------------------
# 0. Make sure we are using the THU-MIG yolov10 repo
# -------------------------------------------------
repo_dir = os.path.dirname(os.path.abspath(__file__))
yolo_dir = os.path.join(repo_dir, "yolov10")
if yolo_dir not in sys.path:
    sys.path.insert(0, yolo_dir)

from sort import Sort
from ultralytics import YOLO  # this is the ultralytics package inside yolov10/

# -----------------------------
# 1. Config
# -----------------------------
CONFIG = {
    "model": {
        "weights": "weights/yolov10n.pt",
        "device": "cpu",            # "cpu" only for your M3 setup
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
    },
    "input": {
        "videos_dir": "videos",
        "output_dir": "outputs",
        "crop_objects": True,
        "crop_size_limit": 1024,
    },
    "tracker": {
        "max_age": 30,
        "min_hits": 3,
        "iou_threshold": 0.3,
    },
    "save": {
        "detections_csv": True,
        "tracks_csv": True,
        "tracks_json": True,
    },
}

# -----------------------------
# 2. Prepare folders and device
# -----------------------------
device = CONFIG["model"]["device"]
VIDEOS_DIR = CONFIG["input"]["videos_dir"]
OUTPUTS_DIR = CONFIG["input"]["output_dir"]
os.makedirs(OUTPUTS_DIR, exist_ok=True)
CROP_DIR = os.path.join(OUTPUTS_DIR, "crops")
os.makedirs(CROP_DIR, exist_ok=True)

# -----------------------------
# 3. Ensure model weights exist
# -----------------------------
weights = CONFIG["model"]["weights"]
os.makedirs(os.path.dirname(weights), exist_ok=True)
if not os.path.exists(weights):
    import urllib.request

    print(f"Downloading {weights} ...")
    url = "https://github.com/THU-MIG/yolov10/releases/download/v1.0/yolov10n.pt"
    urllib.request.urlretrieve(url, weights)
    print("Download complete.")

# -----------------------------
# 4. Load YOLOv10 model (THU-MIG)
# -----------------------------
print("Loading YOLOv10 model...")
model = YOLO(weights)          # this uses the THU-MIG ultralytics fork
model.to(device)

# -----------------------------
# 5. Initialize SORT tracker
# -----------------------------
tracker = Sort(
    max_age=CONFIG["tracker"]["max_age"],
    min_hits=CONFIG["tracker"]["min_hits"],
    iou_threshold=CONFIG["tracker"]["iou_threshold"],
)

# -----------------------------
# 6. Process videos
# -----------------------------
summary_rows = []

for file in os.listdir(VIDEOS_DIR):
    if not file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        continue

    video_path = os.path.join(VIDEOS_DIR, file)
    print(f"\nProcessing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Could not open video: {video_path}")
        continue

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_path = os.path.join(OUTPUTS_DIR, f"{os.path.splitext(file)[0]}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    pbar = tqdm(total=frame_count, desc=f"Processing {file}")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -----------------------------
        # YOLOv10 inference via YOLO wrapper
        # -----------------------------
        # model(...) returns a list of Results objects
        results_list = model(
            frame,
            conf=CONFIG["model"]["conf_threshold"],
            iou=CONFIG["model"]["iou_threshold"],
            verbose=False,
        )
        results = results_list[0]

        dets = []
        if results.boxes is not None and len(results.boxes) > 0:
            # results.boxes.xyxy: (N,4), results.boxes.conf: (N,)
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                dets.append([float(x1), float(y1), float(x2), float(y2), float(conf)])

        dets = np.array(dets, dtype=float) if dets else np.empty((0, 5), dtype=float)

        # -----------------------------
        # Update SORT tracker
        # -----------------------------
        tracks = tracker.update(dets)

        n_objects = len(tracks)
        mean_area = (
            float(
                np.mean((tracks[:, 2] - tracks[:, 0]) * (tracks[:, 3] - tracks[:, 1]))
            )
            if n_objects
            else 0.0
        )
        mean_conf = float(np.mean(dets[:, 4])) if len(dets) else 0.0

        # -----------------------------
        # Draw boxes & save crops
        # -----------------------------
        for trk in tracks:
            x1, y1, x2, y2, track_id = trk
            track_id = int(track_id)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )
            label = f"ID {track_id}"
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Save crop if enabled
            if CONFIG["input"]["crop_objects"]:
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                x1i = max(0, min(x1i, width - 1))
                x2i = max(0, min(x2i, width - 1))
                y1i = max(0, min(y1i, height - 1))
                y2i = max(0, min(y2i, height - 1))

                if x2i > x1i and y2i > y1i:
                    crop = frame[y1i:y2i, x1i:x2i]
                    h, w = crop.shape[:2]
                    if max(h, w) > CONFIG["input"]["crop_size_limit"]:
                        scale = CONFIG["input"]["crop_size_limit"] / max(h, w)
                        crop = cv2.resize(
                            crop,
                            (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                    crop_path = os.path.join(
                        CROP_DIR,
                        f"{os.path.splitext(file)[0]}_id{track_id}_frame{frame_idx:06d}.jpg",
                    )
                    cv2.imwrite(crop_path, crop)

        # Write frame to output video
        out.write(frame)

        # Save per-frame summary
        summary_rows.append(
            {
                "video": file,
                "frame": frame_idx,
                "n_objects": n_objects,
                "mean_area": mean_area,
                "mean_confidence": mean_conf,
            }
        )

        frame_idx += 1
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()

# -----------------------------
# 7. Save summary stats
# -----------------------------
df = pd.DataFrame(summary_rows)
df.to_csv(os.path.join(OUTPUTS_DIR, "detections_summary.csv"), index=False)
print(f"\n Done! Results saved to: {OUTPUTS_DIR}")
