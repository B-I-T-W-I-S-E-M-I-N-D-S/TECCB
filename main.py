import os
import json
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import opts_thumos as opts_thumos  # Import thumos options
import opts_egtea as opts_egtea    # Import egtea options
import time
import h5py
from tqdm import tqdm
from iou_utils import *
from eval import evaluation_detection
from dataset import VideoDataSet, calc_iou
from models import MYNET, SuppressNet
from loss_func import cls_loss_func, regress_loss_func
from loss_func import MultiCrossEntropyLoss
from functools import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import warnings
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel
import nest_asyncio
from pyngrok import ngrok
import uuid
import re

# Visualization Configuration
VIS_CONFIG = {
    'frame_interval': 1.0,
    'max_frames': 20,
    'save_dir': os.path.join('output', 'visualizations'),
    'video_save_dir': os.path.join('output', 'videos'),
    'gt_color': '#1f77b4',  # Blue for ground truth
    'pred_color': '#ff7f0e',  # Orange for predictions
    'fontsize_label': 10,
    'fontsize_title': 14,
    'frame_highlight_both': 'green',
    'frame_highlight_gt': 'red',
    'frame_highlight_pred': 'black',
    'iou_threshold': 0.3,
    'frame_scale_factor': 0.8,
    'video_text_scale': 0.5,
    'video_gt_text_color': (180, 119, 31),  # BGR for OpenCV
    'video_pred_text_color': (14, 127, 255),  # BGR for OpenCV
    'video_text_thickness': 1,
    'video_font_path': os.path.join('data', 'Poppins ExtraBold Italic 800.ttf'),
    'video_font_fallback': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    'video_pred_text_y': 0.45,
    'video_gt_text_y': 0.55,
    'video_footer_height': 150,
    'video_gt_bar_y': 0.2,
    'video_pred_bar_y': 0.5,
    'video_bar_height': 0.15,
    'video_bar_text_scale': 0.7,
    'min_segment_duration': 1.0,
    'video_frame_text_y': 0.05,
    'video_bar_label_x': 10,
    'video_bar_label_scale': 0.5,
    'scroll_window_duration': 20.0,
    'scroll_speed': 0.2,
}

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# FastAPI Models
class VideoInput(BaseModel):
    video_path: str
    video_name: Optional[str] = None

class Segment(BaseModel):
    label: str
    start: float
    end: float
    duration: float
    score: Optional[float] = None

class VideoPrediction(BaseModel):
    video_name: str
    pred_segments: List[Segment]
    gt_segments: List[Segment]
    summary: Dict
    mAP: float
    visualization_path: Optional[str] = None
    video_output_path: Optional[str] = None
    video_task_id: Optional[str] = None

class TaskStatus(BaseModel):
    visualization_path: Optional[str] = None
    video_output_path: Optional[str] = None
    visualization_stream_url: Optional[str] = None
    video_stream_url: Optional[str] = None
    status: str
    error: Optional[str] = None

# HTML template for video player
VIDEO_PLAYER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Action Detection Video Player</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .video-container {
            max-width: 800px;
            width: 100%;
        }
        video {
            width: 100%;
            border: 2px solid #333;
            border-radius: 8px;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        .controls {
            margin-top: 10px;
            text-align: center;
        }
        button {
            padding: 10px 20px;
            margin: 0 5px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <h1>Action Detection Video: {video_name}</h1>
    <div class="video-container">
        <video controls>
            <source src="{stream_url}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <div class="controls">
            <button onclick="seek(-10)">Rewind 10s</button>
            <button onclick="seek(10)">Forward 10s</button>
        </div>
    </div>
    <script>
        const video = document.querySelector('video');
        function seek(seconds) {{
            video.currentTime += seconds;
        }}
    </script>
</body>
</html>
"""

# Action Detection Model Class
class ActionDetectionModel:
    def __init__(self):
        self.thumos_opt = vars(opts_thumos.parse_opt())
        self.egtea_opt = vars(opts_egtea.parse_opt())
        self.model = None
        self.suppress_model = None
        self.task_status = {}
        self.base_url = None  # Will be set to ngrok public URL

    def set_base_url(self, base_url: str):
        """Set the base URL for streaming endpoints"""
        self.base_url = base_url

    def load_model(self, video_name: str):
        """Loads the MYNET and SuppressNet models based on video name"""
        # Determine which options to use based on video name
        opt = self.thumos_opt if video_name.startswith("video") else self.egtea_opt
        
        # Save options to checkpoint path
        os.makedirs(opt["checkpoint_path"], exist_ok=True)
        opt_file_path = os.path.join(opt["checkpoint_path"], f"{opt['exp']}_opts.json")
        with open(opt_file_path, "w") as f:
            json.dump(opt, f)
        
        # Set seed if specified
        if opt['seed'] >= 0:
            seed = opt['seed']
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Parse anchors
        opt['anchors'] = [int(item) for item in opt['anchors'].split(',')]

        self.model = MYNET(opt).to(device)
        if video_name.startswith("video"):
            checkpoint_path = os.path.join(opt["checkpoint_path"], f"{opt['exp']}ckp_best.pth.tar")
        else:
            checkpoint_path = os.path.join(opt["checkpoint_path"], f"{opt['exp']}_ckp_best.pth.tar")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        base_dict = checkpoint['state_dict']
        self.model.load_state_dict(base_dict)
        self.model.eval()

        if opt["pptype"] == "net":
            self.suppress_model = SuppressNet(opt).to(device)
            if video_name.startswith("video"):
                suppress_checkpoint_path = os.path.join(opt["checkpoint_path"], "_ckp_best_suppress.pth.tar")
            else:
                suppress_checkpoint_path = os.path.join(opt["checkpoint_path"], "ckp_best_suppress.pth.tar")
            suppress_checkpoint = torch.load(suppress_checkpoint_path, map_location=device)
            suppress_base_dict = suppress_checkpoint['state_dict']
            self.suppress_model.load_state_dict(suppress_base_dict)
            self.suppress_model.eval()
        
        return opt

    def annotate_video_with_actions(
        self,
        video_id: str,
        pred_segments: List[Dict],
        gt_segments: List[Dict],
        video_path: str,
        save_dir: str = VIS_CONFIG['video_save_dir'],
        text_scale: float = VIS_CONFIG['video_text_scale'] * 1.2,
        gt_text_color: tuple = VIS_CONFIG['video_gt_text_color'],
        pred_text_color: tuple = VIS_CONFIG['video_pred_text_color'],
        text_thickness: int = VIS_CONFIG['video_text_thickness'],
        task_id: Optional[str] = None,
        opt: Optional[Dict] = None
    ) -> str:
        os.makedirs(save_dir, exist_ok=True)
        if task_id:
            self.task_status[task_id] = {"status": "processing"}
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                if task_id:
                    self.task_status[task_id] = {"status": "failed", "error": f"Could not open video {video_path}"}
                raise HTTPException(status_code=400, detail=f"Could not open video {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            footer_height = VIS_CONFIG['video_footer_height']
            output_height = frame_height + footer_height
            output_path = os.path.join(save_dir, f"annotated_{video_id}_{opt['exp']}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v for MP4 output
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, output_height))

            if not out.isOpened():
                cap.release()
                if task_id:
                    self.task_status[task_id] = {"status": "failed", "error": f"Could not initialize video writer for {output_path}"}
                raise HTTPException(status_code=500, detail=f"Could not initialize video writer for {output_path}")

            min_duration = VIS_CONFIG['min_segment_duration']
            gt_segments = [seg for seg in gt_segments if seg['duration'] >= min_duration]
            pred_segments = [seg for seg in pred_segments if seg['duration'] >= min_duration]

            color_palette = [
                (128, 0, 0), (60, 20, 220), (0, 128, 0), (128, 0, 128), (79, 69, 54),
                (128, 128, 0), (0, 0, 128), (130, 0, 75), (34, 139, 34), (0, 85, 204),
                (149, 146, 209), (235, 206, 135), (250, 230, 230), (191, 226, 159),
                (185, 218, 255), (255, 204, 204), (193, 182, 255), (201, 252, 189),
                (144, 128, 112), (112, 25, 25), (102, 51, 102), (0, 128, 128), (171, 71, 0)
            ]
            action_labels = set(seg['label'] for seg in gt_segments).union(seg['label'] for seg in pred_segments)
            action_color_map = {label: color_palette[i % len(color_palette)] for i, label in enumerate(action_labels)}

            gt_color_rgb = (gt_text_color[2], gt_text_color[1], gt_text_color[0])
            pred_color_rgb = (pred_text_color[2], pred_text_color[1], pred_text_color[0])

            font_path = VIS_CONFIG['video_font_path']
            font_fallback = VIS_CONFIG['video_font_fallback']
            font_size = int(20 * text_scale)
            bar_font_size = int(20 * VIS_CONFIG['video_bar_text_scale'])
            font = None
            bar_font = None
            if font_path:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    bar_font = ImageFont.truetype(font_path, bar_font_size)
                except IOError:
                    try:
                        font = ImageFont.truetype(font_fallback, font_size)
                        bar_font = ImageFont.truetype(font_fallback, bar_font_size)
                    except IOError:
                        font = None
                        bar_font = None

            window_size = 20.0
            num_windows = int(np.ceil(duration / window_size))
            text_bar_gap = 48
            text_x = 10

            frame_idx = 0
            written_frames = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                extended_frame = np.zeros((output_height, frame_width, 3), dtype=np.uint8)
                extended_frame[:frame_height, :, :] = frame
                extended_frame[frame_height:, :, :] = 255

                timestamp = frame_idx / fps
                window_idx = int(timestamp // window_size)
                window_start = window_idx * window_size
                window_end = min(window_start + window_size, duration)
                window_duration = window_end - window_start
                window_timestamp = timestamp - window_start

                gt_labels = [seg['label'] for seg in gt_segments if seg['start'] <= timestamp <= seg['end']]
                gt_text = "GT: " + ", ".join(gt_labels) if gt_labels else ""
                pred_labels = [seg['label'] for seg in pred_segments if seg['start'] <= timestamp <= seg['end']]
                pred_text = "Pred: " + ", ".join(pred_labels) if pred_labels else ""

                footer_y = frame_height
                gt_bar_y = footer_y + int(0.2 * footer_height)
                pred_bar_y = footer_y + int(0.5 * footer_height)
                bar_height = int(VIS_CONFIG['video_bar_height'] * footer_height)

                if font:
                    gt_text_bbox = bar_font.getbbox("GT")
                    pred_text_bbox = bar_font.getbbox("Pred")
                    gt_text_width = gt_text_bbox[2] - gt_text_bbox[0]
                    pred_text_width = pred_text_bbox[2] - pred_text_bbox[0]
                else:
                    gt_text_size, _ = cv2.getTextSize("GT", cv2.FONT_HERSHEY_DUPLEX, VIS_CONFIG['video_bar_text_scale'], 1)
                    pred_text_size, _ = cv2.getTextSize("Pred", cv2.FONT_HERSHEY_DUPLEX, VIS_CONFIG['video_bar_text_scale'], 1)
                    gt_text_width = gt_text_size[0]
                    pred_text_width = pred_text_size[0]
                max_text_width = max(gt_text_width, pred_text_width)
                bar_start_x = text_x + max_text_width + text_bar_gap
                bar_width = frame_width - bar_start_x

                for seg in gt_segments:
                    if seg['start'] <= window_end and seg['end'] >= window_start:
                        start_t = max(seg['start'], window_start)
                        end_t = min(seg['end'], window_start + window_timestamp)
                        start_x = bar_start_x + int(((start_t - window_start) / window_duration) * bar_width)
                        end_x = bar_start_x + int(((end_t - window_start) / window_duration) * bar_width)
                        if end_x > start_x:
                            cv2.rectangle(
                                extended_frame,
                                (start_x, gt_bar_y),
                                (end_x, gt_bar_y + bar_height),
                                action_color_map[seg['label']],
                                -1
                            )

                for seg in pred_segments:
                    if seg['start'] <= window_end and seg['end'] >= window_start:
                        start_t = max(seg['start'], window_start)
                        end_t = min(seg['end'], window_start + window_timestamp)
                        start_x = bar_start_x + int(((start_t - window_start) / window_duration) * bar_width)
                        end_x = bar_start_x + int(((end_t - window_start) / window_duration) * bar_width)
                        if end_x > start_x:
                            cv2.rectangle(
                                extended_frame,
                                (start_x, pred_bar_y),
                                (end_x, pred_bar_y + bar_height),
                                action_color_map[seg['label']],
                                -1
                            )

                if font:
                    frame_rgb = cv2.cvtColor(extended_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    draw = ImageDraw.Draw(pil_image)
                    frame_info = f"Frame: {frame_idx} | FPS: {fps:.2f}"
                    frame_text_bbox = draw.textbbox((0, 0), frame_info, font=font)
                    frame_text_width = frame_text_bbox[2] - frame_text_bbox[0]
                    frame_text_x = (frame_width - frame_text_width) // 2
                    draw.text((frame_text_x, 10), frame_info, font=font, fill=(0, 0, 0))
                    window_info = f"{window_start:.1f}s - {window_end:.1f}s"
                    window_text_bbox = draw.textbbox((0, 0), window_info, font=bar_font)
                    window_text_width = window_text_bbox[2] - window_text_bbox[0]
                    window_text_x = (frame_width - window_text_width) // 2
                    draw.text((window_text_x, footer_y + 10), window_info, font=bar_font, fill=(0, 0, 0))
                    if gt_text:
                        gt_y = int(frame_height * VIS_CONFIG['video_gt_text_y'])
                        draw.text((10, gt_y), gt_text, font=font, fill=gt_color_rgb)
                    if pred_text:
                        pred_y = int(frame_height * VIS_CONFIG['video_pred_text_y'])
                        draw.text((10, pred_y), pred_text, font=font, fill=pred_color_rgb)
                    draw.text((text_x, gt_bar_y + bar_height // 2), "GT", font=bar_font, fill=gt_color_rgb)
                    draw.text((text_x, pred_bar_y + bar_height // 2), "Pred", font=bar_font, fill=pred_color_rgb)
                    extended_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                else:
                    frame_info = f"Frame: {frame_idx} | FPS: {fps:.2f}"
                    text_size, _ = cv2.getTextSize(frame_info, cv2.FONT_HERSHEY_DUPLEX, text_scale, text_thickness)
                    frame_text_x = (frame_width - text_size[0]) // 2
                    cv2.putText(
                        extended_frame,
                        frame_info,
                        (frame_text_x, 30),
                        cv2.FONT_HERSHEY_DUPLEX,
                        text_scale,
                        (0, 0, 0),
                        text_thickness,
                        cv2.LINE_AA
                    )
                    window_info = f"{window_start:.1f}s - {window_end:.1f}s"
                    window_text_size, _ = cv2.getTextSize(window_info, cv2.FONT_HERSHEY_DUPLEX, VIS_CONFIG['video_bar_text_scale'], 1)
                    window_text_x = (frame_width - window_text_size[0]) // 2
                    cv2.putText(
                        extended_frame,
                        window_info,
                        (window_text_x, footer_y + 20),
                        cv2.FONT_HERSHEY_DUPLEX,
                        VIS_CONFIG['video_bar_text_scale'],
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA
                    )
                    if gt_text:
                        cv2.putText(
                            extended_frame,
                            gt_text,
                            (10, int(frame_height * VIS_CONFIG['video_gt_text_y'])),
                            cv2.FONT_HERSHEY_DUPLEX,
                            text_scale,
                            gt_text_color,
                            text_thickness,
                            cv2.LINE_AA
                        )
                    if pred_text:
                        cv2.putText(
                            extended_frame,
                            pred_text,
                            (10, int(frame_height * VIS_CONFIG['video_pred_text_y'])),
                            cv2.FONT_HERSHEY_DUPLEX,
                            text_scale,
                            pred_text_color,
                            text_thickness,
                            cv2.LINE_AA
                        )
                    cv2.putText(
                        extended_frame,
                        "GT",
                        (text_x, gt_bar_y + bar_height // 2 + 5),
                        cv2.FONT_HERSHEY_DUPLEX,
                        VIS_CONFIG['video_bar_text_scale'],
                        gt_text_color,
                        1,
                        cv2.LINE_AA
                    )
                    cv2.putText(
                        extended_frame,
                        "Pred",
                        (text_x, pred_bar_y + bar_height // 2 + 5),
                        cv2.FONT_HERSHEY_DUPLEX,
                        VIS_CONFIG['video_bar_text_scale'],
                        pred_text_color,
                        1,
                        cv2.LINE_AA
                    )

                out.write(extended_frame)
                written_frames += 1
                frame_idx += 1

            cap.release()
            out.release()
            print(f"[✅ Saved Annotated Video]: {output_path}, Written Frames={written_frames}")
            if task_id:
                self.task_status[task_id] = {
                    "status": "completed",
                    "video_output_path": output_path,
                    "video_stream_url": f"{self.base_url}/stream/videos/{os.path.basename(output_path)}"
                }
            return output_path
        except Exception as e:
            if task_id:
                self.task_status[task_id] = {"status": "failed", "error": str(e)}
            raise e

    def visualize_action_lengths(
        self,
        video_id: str,
        pred_segments: List[Dict],
        gt_segments: List[Dict],
        video_path: str,
        duration: float,
        save_dir: str = VIS_CONFIG['save_dir'],
        frame_interval: float = VIS_CONFIG['frame_interval'],
        task_id: Optional[str] = None,
        opt: Optional[Dict] = None
    ) -> str:
        os.makedirs(save_dir, exist_ok=True)
        if task_id:
            self.task_status[task_id] = {"status": "processing"}
        try:
            num_frames = int(duration / frame_interval) + 1
            if num_frames > VIS_CONFIG['max_frames']:
                frame_interval = duration / (VIS_CONFIG['max_frames'] - 1)
                num_frames = VIS_CONFIG['max_frames']
                print(f"Warning: Adjusted frame_interval to {frame_interval:.2f}s.")

            frame_times = np.linspace(0, duration, num_frames, endpoint=False)
            frames = []
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}. Using placeholder frames.")
                frames = [np.ones((100, 100, 3), dtype=np.uint8) * 255 for _ in frame_times]
            else:
                for t in frame_times:
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
                        frames.append(frame)
                    else:
                        frames.append(np.ones((100, 100, 3), dtype=np.uint8) * 255)
                cap.release()

            fig = plt.figure(figsize=(num_frames * VIS_CONFIG['frame_scale_factor'], 6), constrained_layout=True)
            gs = fig.add_gridspec(3, num_frames, height_ratios=[3, 1, 1])

            for i, (t, frame) in enumerate(zip(frame_times, frames)):
                ax = fig.add_subplot(gs[0, i])
                gt_hit = any(seg['start'] <= t <= seg['end'] for seg in gt_segments)
                pred_hit = any(seg['start'] <= t <= seg['end'] for seg in pred_segments)
                border_color = None
                if gt_hit and pred_hit:
                    border_color = VIS_CONFIG['frame_highlight_both']
                elif gt_hit:
                    border_color = VIS_CONFIG['frame_highlight_gt']
                elif pred_hit:
                    border_color = VIS_CONFIG['frame_highlight_pred']
                ax.imshow(frame)
                ax.axis('off')
                if border_color:
                    for spine in ax.spines.values():
                        spine.set_edgecolor(border_color)
                        spine.set_linewidth(2)
                ax.set_title(f"{t:.1f}s", fontsize=VIS_CONFIG['fontsize_label'],
                             color=border_color if border_color else 'black')

            ax_gt = fig.add_subplot(gs[1, :])
            ax_gt.set_xlim(0, duration)
            ax_gt.set_ylim(0, 1)
            ax_gt.axis('off')
            ax_gt.text(-0.02 * duration, 0.5, "Ground Truth", fontsize=VIS_CONFIG['fontsize_title'],
                       va='center', ha='right', weight='bold')
            for seg in gt_segments:
                start, end = seg['start'], seg['end']
                width = end - start
                label = seg['label'][:10] + '...' if len(seg['label']) > 10 else seg['label']
                ax_gt.add_patch(patches.Rectangle(
                    (start, 0.3), width, 0.4, facecolor=VIS_CONFIG['gt_color'],
                    edgecolor='black', alpha=0.8
                ))
                ax_gt.text((start + end) / 2, 0.5, label, ha='center', va='center',
                           fontsize=VIS_CONFIG['fontsize_label'], color='white')
                ax_gt.text(start, 0.2, f"{start:.1f}", ha='center', fontsize=8, color='black')
                ax_gt.text(end, 0.2, f"{end:.1f}", ha='center', fontsize=8, color='black')

            ax_pred = fig.add_subplot(gs[2, :])
            ax_pred.set_xlim(0, duration)
            ax_pred.set_ylim(0, 1)
            ax_pred.axis('off')
            ax_pred.text(-0.02 * duration, 0.5, "Prediction", fontsize=VIS_CONFIG['fontsize_title'],
                         va='center', ha='right', weight='bold')
            for seg in pred_segments:
                start, end = seg['start'], seg['end']
                width = end - start
                label = seg['label'][:10] + '...' if len(seg['label']) > 10 else seg['label']
                ax_pred.add_patch(patches.Rectangle(
                    (start, 0.3), width, 0.4, facecolor=VIS_CONFIG['pred_color'],
                    edgecolor='black', alpha=0.8
                ))
                ax_pred.text((start + end) / 2, 0.5, label, ha='center', va='center',
                             fontsize=VIS_CONFIG['fontsize_label'], color='white')
                ax_pred.text(start, 0.8, f"{start:.1f}", ha='center', fontsize=8, color='black')
                ax_pred.text(end, 0.8, f"{end:.1f}", ha='center', fontsize=8, color='black')

            jpg_path = os.path.join(save_dir, f"viz_{video_id}_{opt['exp']}.png")
            plt.savefig(jpg_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"[✅ Saved Visualization]: {jpg_path}")
            if task_id:
                self.task_status[task_id] = {
                    "status": "completed",
                    "visualization_path": jpg_path,
                    "visualization_stream_url": f"{self.base_url}/stream/visualizations/{os.path.basename(jpg_path)}"
                }
            return jpg_path
        except Exception as e:
            if task_id:
                self.task_status[task_id] = {"status": "failed", "error": str(e)}
            raise e

    def eval_frame(self, dataset, opt):
        test_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=opt['batch_size'], shuffle=False,
                                                  num_workers=0, pin_memory=True, drop_last=False)
        labels_cls = {video_name: [] for video_name in dataset.video_list}
        labels_reg = {video_name: [] for video_name in dataset.video_list}
        output_cls = {video_name: [] for video_name in dataset.video_list}
        output_reg = {video_name: [] for video_name in dataset.video_list}

        start_time = time.time()
        total_frames = 0
        epoch_cost = 0
        epoch_cost_cls = 0
        epoch_cost_reg = 0

        for n_iter, (input_data, cls_label, reg_label, _) in enumerate(tqdm(test_loader)):
            input_data = input_data.to(device)
            cls_label = cls_label.to(device)
            reg_label = reg_label.to(device)
            act_cls, act_reg, _ = self.model(input_data.float())
            cost_reg = 0
            cost_cls = 0

            loss = cls_loss_func(cls_label, act_cls)
            cost_cls = loss
            epoch_cost_cls += cost_cls.detach().cpu().numpy()

            loss = regress_loss_func(reg_label, act_reg)
            cost_reg = loss
            epoch_cost_reg += cost_reg.detach().cpu().numpy()

            cost = opt['alpha'] * cost_cls + opt['beta'] * cost_reg
            epoch_cost += cost.detach().cpu().numpy()

            act_cls = torch.softmax(act_cls, dim=-1)

            total_frames += input_data.size(0)

            for b in range(0, input_data.size(0)):
                video_name, st, ed, data_idx = dataset.inputs[n_iter * opt['batch_size'] + b]
                output_cls[video_name] += [act_cls[b, :].detach().cpu().numpy()]
                output_reg[video_name] += [act_reg[b, :].detach().cpu().numpy()]
                labels_cls[video_name] += [cls_label[b, :].cpu().numpy()]
                labels_reg[video_name] += [reg_label[b, :].cpu().numpy()]

        end_time = time.time()
        working_time = end_time - start_time

        for video_name in dataset.video_list:
            labels_cls[video_name] = np.stack(labels_cls[video_name], axis=0)
            labels_reg[video_name] = np.stack(labels_reg[video_name], axis=0)
            output_cls[video_name] = np.stack(output_cls[video_name], axis=0)
            output_reg[video_name] = np.stack(output_reg[video_name], axis=0)

        cls_loss = epoch_cost_cls / n_iter
        reg_loss = epoch_cost_reg / n_iter
        tot_loss = epoch_cost / n_iter

        return cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames

    def eval_map_nms(self, dataset, output_cls, output_reg, opt):
        result_dict = {}
        proposal_dict = []
        num_class = opt["num_of_class"]
        unit_size = opt['segment_size']
        threshold = opt['threshold']
        anchors = opt['anchors']

        for video_name in dataset.video_list:
            duration = dataset.video_len[video_name]
            video_time = float(dataset.video_dict[video_name]["duration"])
            frame_to_time = 100.0 * video_time / duration

            for idx in range(0, duration):
                cls_anc = output_cls[video_name][idx]
                reg_anc = output_reg[video_name][idx]

                proposal_anc_dict = []
                for anc_idx in range(0, len(anchors)):
                    cls = np.argwhere(cls_anc[anc_idx][:-1] > opt['threshold']).reshape(-1)
                    if len(cls) == 0:
                        continue
                    ed = idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                    length = anchors[anc_idx] * np.exp(reg_anc[anc_idx][1])
                    st = ed - length
                    for cidx in range(0, len(cls)):
                        label = cls[cidx]
                        tmp_dict = {}
                        tmp_dict["segment"] = [float(st * frame_to_time / 100.0), float(ed * frame_to_time / 100.0)]
                        tmp_dict["score"] = float(cls_anc[anc_idx][label])
                        tmp_dict["label"] = dataset.label_name[label]
                        tmp_dict["gentime"] = float(idx * frame_to_time / 100.0)
                        proposal_anc_dict.append(tmp_dict)

                proposal_dict += proposal_anc_dict

            proposal_dict = non_max_suppression(proposal_dict, overlapThresh=opt['soft_nms'])
            result_dict[video_name] = proposal_dict
            proposal_dict = []

        return result_dict

    def eval_map_supnet(self, dataset, output_cls, output_reg, opt):
        result_dict = {}
        proposal_dict = []
        num_class = opt["num_of_class"]
        unit_size = opt['segment_size']
        threshold = opt['threshold']
        anchors = opt['anchors']

        for video_name in dataset.video_list:
            duration = dataset.video_len[video_name]
            video_time = float(dataset.video_dict[video_name]["duration"])
            frame_to_time = 100.0 * video_time / duration
            conf_queue = torch.zeros((unit_size, num_class - 1))

            for idx in range(0, duration):
                cls_anc = output_cls[video_name][idx]
                reg_anc = output_reg[video_name][idx]
                proposal_anc_dict = []
                for anc_idx in range(0, len(anchors)):
                    cls = np.argwhere(cls_anc[anc_idx][:-1] > opt['threshold']).reshape(-1)
                    if len(cls) == 0:
                        continue
                    ed = idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                    length = anchors[anc_idx] * np.exp(reg_anc[anc_idx][1])
                    st = ed - length
                    for cidx in range(0, len(cls)):
                        label = cls[cidx]
                        tmp_dict = {}
                        tmp_dict["segment"] = [float(st * frame_to_time / 100.0), float(ed * frame_to_time / 100.0)]
                        tmp_dict["score"] = float(cls_anc[anc_idx][label])
                        tmp_dict["label"] = dataset.label_name[label]
                        tmp_dict["gentime"] = float(idx * frame_to_time / 100.0)
                        proposal_anc_dict.append(tmp_dict)

                proposal_anc_dict = non_max_suppression(proposal_anc_dict, overlapThresh=opt['soft_nms'])
                conf_queue[:-1, :] = conf_queue[1:, :].clone()
                conf_queue[-1, :] = 0
                for proposal in proposal_anc_dict:
                    cls_idx = dataset.label_name.index(proposal['label'])
                    conf_queue[-1, cls_idx] = proposal["score"]
                minput = conf_queue.unsqueeze(0).to(device)
                suppress_conf = self.suppress_model(minput)
                suppress_conf = suppress_conf.squeeze(0).detach().cpu().numpy()
                for cls in range(0, num_class - 1):
                    if suppress_conf[cls] > opt['sup_threshold']:
                        for proposal in proposal_anc_dict:
                            if proposal['label'] == dataset.label_name[cls]:
                                if check_overlap_proposal(proposal_dict, proposal, overlapThresh=opt['soft_nms']) is None:
                                    proposal_dict.append(proposal)

            result_dict[video_name] = proposal_dict
            proposal_dict = []

        return result_dict

    async def generate_visualizations(self, video_name: str, pred_segments: List[Dict], gt_segments: List[Dict], video_path: str, duration: float, task_id: str, opt: Dict):
        """Generate visualizations and annotated video in the background"""
        try:
            visualization_path = self.visualize_action_lengths(
                video_id=video_name,
                pred_segments=pred_segments,
                gt_segments=gt_segments,
                video_path=video_path,
                duration=duration,
                task_id=f"viz_{task_id}",
                opt=opt
            )
            video_output_path = self.annotate_video_with_actions(
                video_id=video_name,
                pred_segments=pred_segments,
                gt_segments=gt_segments,
                video_path=video_path,
                task_id=f"video_{task_id}",
                opt=opt
            )
            self.task_status[task_id] = {
                "visualization_path": visualization_path,
                "video_output_path": video_output_path,
                "visualization_stream_url": f"{self.base_url}/stream/visualizations/{os.path.basename(visualization_path)}",
                "video_stream_url": f"{self.base_url}/stream/videos/{os.path.basename(video_output_path)}",
                "status": "completed"
            }
        except Exception as e:
            self.task_status[task_id] = {
                "visualization_path": None,
                "video_output_path": None,
                "visualization_stream_url": None,
                "video_stream_url": None,
                "status": "failed",
                "error": str(e)
            }

    async def predict(self, input_data: VideoInput, background_tasks: BackgroundTasks) -> VideoPrediction:
        input_data = input_data.dict()
        video_path = input_data['video_path']
        video_name = input_data['video_name'] or os.path.splitext(os.path.basename(video_path))[0]

        if not os.path.exists(video_path):
            raise HTTPException(status_code=400, detail=f"Video path {video_path} does not exist")

        # Load the appropriate model and options based on video name
        opt = self.load_model(video_name)
        
        if not self.model:
            raise HTTPException(status_code=400, detail="Model failed to load")

        dataset = VideoDataSet(opt, subset=opt['inference_subset'], video_name=video_name)
        cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = self.eval_frame(dataset, opt)

        if opt["pptype"] == "nms":
            result_dict = self.eval_map_nms(dataset, output_cls, output_reg, opt)
        elif opt["pptype"] == "net":
            result_dict = self.eval_map_supnet(dataset, output_cls, output_reg, opt)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid pptype: {opt['pptype']}")

        mAP_result = evaluation_detection(opt)
        mAP = float(np.mean(mAP_result)) if isinstance(mAP_result, (list, np.ndarray)) else float(mAP_result)

        video_anno_file = opt["video_anno"].format(opt["split"])
        with open(video_anno_file, 'r') as f:
            anno_data = json.load(f)
        gt_annotations = anno_data['database'][video_name]['annotations']
        duration = anno_data['database'][video_name]['duration']

        gt_segments = [
            {
                'label': anno['label'],
                'start': anno['segment'][0],
                'end': anno['segment'][1],
                'duration': anno['segment'][1] - anno['segment'][0]
            } for anno in gt_annotations
        ]
        pred_segments = [
            {
                'label': pred['label'],
                'start': pred['segment'][0],
                'end': pred['segment'][1],
                'duration': pred['segment'][1] - pred['segment'][0],
                'score': pred['score']
            } for pred in result_dict[video_name]
        ]

        matches = []
        iou_threshold = VIS_CONFIG['iou_threshold']
        used_gt_indices = set()
        for pred in pred_segments:
            best_iou = 0
            best_gt_idx = None
            for gt_idx, gt in enumerate(gt_segments):
                if gt_idx in used_gt_indices:
                    continue
                iou = calc_iou([pred['end'], pred['duration']], [gt['end'], gt['duration']])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_gt_idx is not None:
                matches.append({
                    'pred': pred,
                    'gt': gt_segments[best_gt_idx],
                    'iou': best_iou
                })
                used_gt_indices.add(best_gt_idx)
            else:
                matches.append({'pred': pred, 'gt': None, 'iou': 0})
        for gt_idx, gt in enumerate(gt_segments):
            if gt_idx not in used_gt_indices:
                matches.append({'pred': None, 'gt': gt, 'iou': 0})

        matched_count = sum(1 for m in matches if m['pred'] and m['gt'])
        avg_duration_diff = np.mean([m['pred']['duration'] - m['gt']['duration'] for m in matches if m['pred'] and m['gt']]) if matched_count > 0 else 0
        avg_iou = np.mean([m['iou'] for m in matches if m['iou'] > 0]) if any(m['iou'] > 0 for m in matches) else 0
        summary = {
            "total_predictions": len(pred_segments),
            "total_ground_truth": len(gt_segments),
            "matched_segments": matched_count,
            "avg_duration_diff": float(avg_duration_diff),
            "avg_iou": float(avg_iou)
        }

        task_id = str(uuid.uuid4())

        if os.path.exists(video_path):
            background_tasks.add_task(
                self.generate_visualizations,
                video_name=video_name,
                pred_segments=pred_segments,
                gt_segments=gt_segments,
                video_path=video_path,
                duration=duration,
                task_id=task_id,
                opt=opt
            )
            self.task_status[task_id] = {
                "visualization_path": None,
                "video_output_path": None,
                "visualization_stream_url": None,
                "video_stream_url": None,
                "status": "pending"
            }

        return VideoPrediction(
            video_name=video_name,
            pred_segments=[Segment(**seg) for seg in pred_segments],
            gt_segments=[Segment(**{k: v for k, v in seg.items() if k != 'score'}) for seg in gt_segments],
            summary=summary,
            mAP=mAP,
            visualization_path=None,
            video_output_path=None,
            video_task_id=task_id if os.path.exists(video_path) else None
        )

    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Retrieve the status of a background visualization task"""
        status = self.task_status.get(task_id, None)
        if not status:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return TaskStatus(**status)

    def stream_file(self, file_type: str, file_name: str, range_header: Optional[str] = None) -> Tuple[StreamingResponse, dict]:
        """
        Stream a video or visualization file with range-based streaming support.
        """
        if file_type not in ["visualizations", "videos"]:
            raise HTTPException(status_code=400, detail="Invalid file_type. Use 'visualizations' or 'videos'.")
        
        file_path = os.path.join("output", file_type, file_name)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {file_path} not found.")
        
        file_size = os.stat(file_path).st_size
        media_type = "video/mp4" if file_type == "videos" else "image/png"
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{file_name}"',
            "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
        }

        # Handle range requests
        start = 0
        end = file_size - 1
        status_code = 200

        if range_header:
            range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if range_match:
                start = int(range_match.group(1))
                end_str = range_match.group(2)
                end = int(end_str) if end_str else file_size - 1
                if start >= file_size or end >= file_size or start > end:
                    raise HTTPException(
                        status_code=416,
                        detail="Requested Range Not Satisfiable"
                    )
                headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
                headers["Content-Length"] = str(end - start + 1)
                status_code = 206  # Partial Content

        def iterfile():
            with open(file_path, mode="rb") as file_like:
                file_like.seek(start)
                remaining = end - start + 1
                chunk_size = 1024 * 1024  # 1MB chunks
                while remaining > 0:
                    chunk = file_like.read(min(chunk_size, remaining))
                    if not chunk:
                        break
                    yield chunk
                    remaining -= len(chunk)

        return StreamingResponse(
            iterfile(),
            media_type=media_type,
            headers=headers,
            status_code=status_code
        ), headers

# Initialize FastAPI app
app = FastAPI(title="Action Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Initialize model without loading (load dynamically in predict)
action_model = ActionDetectionModel()

@app.get('/')
def index():
    return {'message': 'This is the Action Detection API!'}

@app.post('/predict', response_model=VideoPrediction)
async def predict(prediction: VideoPrediction = Depends(action_model.predict)):
    """
    Predict action segments in a video.

    Example POST request body:
    {
        "video_path": "/path/to/video.mp4",
        "video_name": "example_video"
    }

    Example response:
    {
        "video_name": "example_video",
        "pred_segments": [
            {"label": "action1", "start": 0.0, "end": 5.0, "duration": 5.0, "score": 0.95},
            ...
        ],
        "gt_segments": [
            {"label": "action1", "start": 0.1, "end": 5.2, "duration": 5.1},
            ...
        ],
        "summary": {
            "total_predictions": 10,
            "total_ground_truth": 8,
            "matched_segments": 7,
            "avg_duration_diff": 0.15,
            "avg_iou": 0.85
        },
        "mAP": 0.75,
        "visualization_path": null,
        "video_output_path": null,
        "video_task_id": "550e8400-e29b-41d4-a716-446655440000"
    }
    """
    return prediction

@app.get("/task_status/{task_id}", response_model=TaskStatus)
async def task_status(task_status: TaskStatus = Depends(action_model.get_task_status)):
    """
    Check the status of a background visualization task.

    Example response:
    {
        "visualization_path": "output/visualizations/viz_example_video_exp.png",
        "video_output_path": "output/videos/annotated_example_video_exp.mp4",
        "visualization_stream_url": "https://<ngrok-id>.ngrok.io/stream/visualizations/viz_example_video_exp.png",
        "video_stream_url": "https://<ngrok-id>.ngrok.io/stream/videos/annotated_example_video_exp.mp4",
        "status": "completed"
    }
    """
    return task_status

@app.get("/stream/{file_type}/{file_name}")
async def stream_file(file_type: str, file_name: str, request: Request):
    """
    Stream a video or visualization file with range-based streaming support.
    
    Example:
        GET /stream/videos/annotated_example_video_exp.mp4
        Headers: Range: bytes=0-1048576
    """
    response, headers = action_model.stream_file(file_type, file_name, request.headers.get("Range"))
    return response

@app.get("/watch/{video_name}", response_class=HTMLResponse)
async def watch_video(video_name: str):
    """
    Serve an HTML page with a video player for the annotated video.
    
    Example:
        GET /watch/example_video
    """
    opt = action_model.thumos_opt if video_name.startswith("video") else action_model.egtea_opt
    file_name = f"annotated_{video_name}_{opt['exp']}.mp4"
    file_path = os.path.join("output", "videos", file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Video {file_name} not found")
    
    stream_url = f"{action_model.base_url}/stream/videos/{file_name}"
    html_content = VIDEO_PLAYER_HTML.format(video_name=video_name, stream_url=stream_url)
    return HTMLResponse(content=html_content)

@app.get("/download/{file_type}/{file_name}")
async def download_file(file_type: str, file_name: str):
    """
    Download a video or visualization file.
    """
    if file_type not in ["visualizations", "videos"]:
        raise HTTPException(status_code=400, detail="Invalid file_type. Use 'visualizations' or 'videos'.")
    file_path = os.path.join("output", file_type, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {file_path} not found.")
    return FileResponse(file_path, media_type="application/octet-stream", filename=file_name)

@app.on_event("startup")
async def startup():
    # Load both options but defer model loading to predict
    print("Action detection options loaded successfully")
    port = 8004
    ngrok_tunnel = ngrok.connect(port)
    action_model.set_base_url(ngrok_tunnel.public_url)
    print('Public URL:', ngrok_tunnel.public_url)

if __name__ == '__main__':
    port = 8004
    nest_asyncio.apply()
    uvicorn.run(app, port=port)