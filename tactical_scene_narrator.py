"""Tactical Scene Narrator – real‑time SALUTE generator

Prerequisites (tested 2025‑07, Python 3.10):
    pip install ultralytics opencv-python requests python-dotenv

Environment variables:
    HF_TOKEN  – Hugging Face Inference API token (scope: inference)

Usage:
    python tactical_scene_narrator.py --source demo.mp4 --lat 37.44 --lon 126.95
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import sys
import threading
import time
import uuid

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
# from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import requests
HEADERS = {"User-Agent": "TacticalSceneNarrator/0.1"}

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

# -----------------------------------------
# constants & utils
# -----------------------------------------

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

# 전역 선언
processor = LlavaNextProcessor.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf"
)
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

PERSON_IDX = COCO_CLASSES.index("person")
VEHICLE_IDXS = [COCO_CLASSES.index(x) for x in ("car", "truck", "bus", "motorcycle", "train", "airplane")]
WEAPON_KEYWORDS = {"rifle", "gun", "weapon"}  # naive textual filter for caption step

# -----------------------------------------
# data structures
# -----------------------------------------

@dataclass
class SaluteReport:
    report_id: str
    size: int
    activity: str
    location: str
    unit: str
    time: str
    equipment: str
    severity: str # 중요도 필드 추가

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, separators=(",", ":"))

# -----------------------------------------
# core functions
# -----------------------------------------


def detect_objects(model: YOLO, frame: np.ndarray, conf: float = 0.3):
    """Run YOLO detection and return ndarray of shape [N, 6] (x1,y1,x2,y2,conf,class)."""
    results = model(frame)  # list with length 1
    boxes = results[0].boxes
    if boxes is None:
        return np.empty((0, 6))
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy().reshape(-1, 1)
    clss = boxes.cls.cpu().numpy().reshape(-1, 1)
    detections = np.hstack([xyxy, confs, clss])
    detections = detections[detections[:, 4] >= conf]
    return detections


def summarize_counts(dets: np.ndarray) -> Tuple[int, int, int]:
    persons = (dets[:, 5] == PERSON_IDX).sum()
    vehicles = np.isin(dets[:, 5], VEHICLE_IDXS).sum()
    # weapons via detection rare; rely on caption keywords later
    return int(persons), int(vehicles), 0


def query_llava(frame: np.ndarray) -> str:
    """LLaVA Next 로컬 모델을 사용해 이미지 설명 생성 (오류 수정된 버전)."""
    try:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # LLaVA 1.6 모델에 맞는 정확한 프롬프트 형식으로 수정
        prompt_template = "[INST] <image>\nDescribe what you see in this scene concisely. [/INST]"

        # Processor를 통해 입력 생성 (text와 images 인자를 명시적으로 사용)
        inputs = processor(text=prompt_template, images=image, return_tensors="pt").to(model.device)

        # 추론
        with torch.no_grad():
            # pad_token_id를 명시적으로 설정하여 경고 메시지 방지
            output = model.generate(**inputs, max_new_tokens=64, pad_token_id=processor.tokenizer.eos_token_id)

        # 결과 디코딩 및 프롬프트 부분 제거
        full_caption = processor.decode(output[0], skip_special_tokens=True)
        # 생성된 텍스트만 깔끔하게 추출
        caption = full_caption.split("[/INST]")[-1].strip()

        return caption or "… No caption generated …"

    except Exception as e:
        print(f"[LLaVA Error] {e}") # 터미널에 오류를 출력하기 위해 유지
        return "… Caption generation error …"

SALUTE_REGEX = re.compile(
    r"(?P<size>\d+).*?(?P<activity>[\w\s]+?)(?:at|near|in) (?P<location>.*?)(?:,|\.|$).*?(?P<time>\d{1,2}:\d{2})?.*?(?P<equipment>[\w\s]+)?",
    re.IGNORECASE,
)


def extract_salute(caption: str, persons: int, vehicles: int, weapons_flag: bool, default_addr: str) -> SaluteReport:
    report_id = f"REP-{str(uuid.uuid4())[:8].upper()}" # 고유 리포트 ID 생성

    m = SALUTE_REGEX.search(caption)
    if m:
        size = int(m.group("size"))
        activity = (m.group("activity") or "unknown").strip()
        location = (m.group("location") or default_addr).strip()
        report_time = m.group("time") or time.strftime("%H:%M")
        equipment = (m.group("equipment") or ("weapons" if weapons_flag else "none")).strip()
    else:
        size = max(1, persons + vehicles)
        activity = "moving" if vehicles else "standing"
        location = default_addr
        report_time = time.strftime("%H:%M")
        equipment = "weapons" if weapons_flag else "none"
    
    unit = "unknown"
    
    # 중요도 자동 판단 로직
    if weapons_flag:
        severity = "CRITICAL"
    elif vehicles > 5 or persons > 10:
        severity = "HIGH"
    elif vehicles > 0 or persons > 0:
        severity = "MEDIUM"
    else:
        severity = "LOW"
        
    return SaluteReport(report_id, size, activity, location, unit, report_time, equipment, severity)


def reverse_geocode(lat: float, lon: float) -> str:
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "jsonv2"},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("display_name", "")
    except Exception as e:
        print(f"[Nominatim] error: {e}")
        return "coordinates {:.5f},{:.5f}".format(lat, lon)

# -----------------------------------------
# worker thread for caption queue (decouple from main thread)
# -----------------------------------------

def caption_worker(q_in: "queue.Queue[np.ndarray]", q_out: "queue.Queue[str]", stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            frame = q_in.get(timeout=0.1)
        except queue.Empty:
            continue
        caption = query_llava(frame)
        q_out.put(caption)
        q_in.task_done()

# -----------------------------------------
# main
# -----------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Tactical Scene Narrator – real‑time SALUTE generator")
    p.add_argument("--source", required=True, help="RTSP url or video file path")
    p.add_argument("--lat", type=float, default=None, help="latitude (if known)")
    p.add_argument("--lon", type=float, default=None, help="longitude (if known)")
    p.add_argument("--sample-rate", type=int, default=30, help="frames interval for caption (≈fps)")
    p.add_argument("--out", type=str, default=None, help="optional path to save annotated video")
    p.add_argument("--show", action="store_true", help="show window")
    return p.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Error: cannot open source")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_interval = max(1, int(args.sample_rate))

    out_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    yolo = YOLO("yolov8s.pt")

    addr_name = "unknown location"
    if args.lat is not None and args.lon is not None:
        addr_name = reverse_geocode(args.lat, args.lon)

    q_in: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
    q_out: "queue.Queue[str]" = queue.Queue()
    stop_event = threading.Event()
    worker = threading.Thread(target=caption_worker, args=(q_in, q_out, stop_event), daemon=True)
    worker.start()

    frame_id = 0
    last_caption = ""
    last_report_json = "{}"

    def handle_sigint(sig, frame):
        stop_event.set()
        cap.release()
        if out_writer:
            out_writer.release()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = detect_objects(yolo, frame)
        n_person, n_vehicle, _ = summarize_counts(dets)

        # mark bounding boxes
        for x1, y1, x2, y2, conf, cls in dets:
            cls = int(cls)
            label = COCO_CLASSES[cls]
            color = (0, 255, 0) if cls == PERSON_IDX else (255, 0, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            cv2.putText(frame, label, (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # enqueue frame for caption every sample_interval frames
        if frame_id % sample_interval == 0 and not q_in.full():
            q_in.put(frame.copy())

        # fetch caption if available
        try:
            last_caption = q_out.get_nowait()
        except queue.Empty:
            pass

        # weapons keyword detection on caption
        weapons_flag = any(k in last_caption.lower() for k in WEAPON_KEYWORDS)

        # generate SALUTE
        report = extract_salute(last_caption, n_person, n_vehicle, weapons_flag, addr_name)
        last_report_json = report.to_json()

        # overlay HUD
        hud_lines = [f"CAPTION: {last_caption[:60]}…" if last_caption else "CAPTION: …pending…",
                     f"SALUTE: {last_report_json}"]
        y0 = 20
        for line in hud_lines:
            cv2.putText(frame, line, (10, y0), cv2.FONT_HERSHEY_PLAIN, 0.45, (0, 255, 255), 1)
            y0 += 18

        # write / show
        if out_writer:
            out_writer.write(frame)
        if args.show:
            cv2.imshow("Tactical Scene Narrator", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_id % int(fps) == 0:  # every second log to console
            print(last_report_json)

        frame_id += 1

    stop_event.set()
    worker.join()
    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
