import os
import json
import torch
import torchvision
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import opts_egtea as opts

import time
import h5py
from tqdm import tqdm
from iou_utils import *
from eval import evaluation_detection
from tensorboardX import SummaryWriter
from dataset import VideoDataSet, calc_iou
from models import MYNET, SuppressNet
from loss_func import cls_loss_func, cls_loss_func_, regress_loss_func
from loss_func import MultiCrossEntropyLoss
from functools import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from typing import List, Dict, Optional

from PIL import Image, ImageDraw, ImageFont
import warnings

# Visualization Configuration (Updated)
VIS_CONFIG = {
    'frame_interval': 1.0,
    'max_frames': 20,
    'save_dir': './output/visualizations',
    'video_save_dir': './output/videos',
    'gt_color': '#1f77b4',  # Blue for ground truth (RGB: 31, 119, 180)
    'pred_color': '#ff7f0e',  # Orange for predictions (RGB: 255, 127, 14)
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
    'video_font_path': "./data/Poppins ExtraBold Italic 800.ttf",
    'video_font_fallback': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    'video_pred_text_y': 0.45,
    'video_gt_text_y': 0.55,
    'video_footer_height': 150,  # Increased to accommodate labels
    'video_gt_bar_y': 0.5,
    'video_pred_bar_y': 0.8,
    'video_bar_height': 0.15,
    'video_bar_text_scale': 0.7,
    'min_segment_duration': 1.0,
    'video_frame_text_y': 0.05,  # Position for frame number and FPS
    'video_bar_label_x': 10,  # X-position for GT/Pred labels
    'video_bar_label_scale': 0.5,
    'scroll_window_duration': 30.0,  # Duration of the visible time window (seconds)
    'scroll_speed': 0.5,  # Seconds to advance the window per second of video
}


def annotate_video_with_actions(
    video_id: str,
    pred_segments: List[Dict],
    gt_segments: List[Dict],
    video_path: str,
    save_dir: str = VIS_CONFIG['video_save_dir'],
    text_scale: float = VIS_CONFIG['video_text_scale'] * 1.5,  # Increased text size by 50%
    gt_text_color: tuple = VIS_CONFIG['video_gt_text_color'],
    pred_text_color: tuple = VIS_CONFIG['video_pred_text_color'],
    text_thickness: int = VIS_CONFIG['video_text_thickness']
) -> None:
    """
    Annotate a video with predicted and ground truth action labels, cumulative bars, frame number, and FPS.
    Use fixed 20-second windows with original bar animation, resetting bars at each window boundary.
    Different colors for different action classes, no labels or timestamps on bars, increased text size.
    GT and Pred text labels are on the left, with bars starting 0.5 inches (48 pixels) to the right.

    Args:
        video_id: Video identifier (e.g., 'my_video').
        pred_segments: List of predicted segments with 'label', 'start', 'end', 'duration', 'score'.
        gt_segments: List of ground truth segments with 'label', 'start', 'end', 'duration'.
        video_path: Path to the input video file.
        save_dir: Directory to save the annotated video.
        text_scale: Scale factor for text size in video (increased).
        gt_text_color: BGR color tuple for ground truth text.
        pred_text_color: BGR color tuple for predicted text.
        text_thickness: Thickness of text strokes.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}. Skipping video annotation.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"Input Video: FPS={fps:.2f}, Resolution={frame_width}x{frame_height}, Total Frames={total_frames}, Duration={duration:.2f}s")

    # Define output video with extended height for footer
    footer_height = VIS_CONFIG['video_footer_height']
    output_height = frame_height + footer_height
    output_path = os.path.join(save_dir, f"annotated_{video_id}_{opt['exp']}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, output_height))

    if not out.isOpened():
        print(f"Error: Could not initialize video writer for {output_path}. Check codec availability.")
        cap.release()
        return

    # Filter short segments
    min_duration = VIS_CONFIG['min_segment_duration']
    gt_segments = [seg for seg in gt_segments if seg['duration'] >= min_duration]
    pred_segments = [seg for seg in pred_segments if seg['duration'] >= min_duration]
    print(f"Filtered Segments: GT={len(gt_segments)}, Pred={len(pred_segments)} (min_duration={min_duration}s)")

    # Define color palette (BGR)
    color_palette = [
        (128, 0, 0),      # Navy Blue
        (60, 20, 220),    # Crimson Red
        (0, 128, 0),      # Emerald Green
        (128, 0, 128),    # Royal Purple
        (79, 69, 54),     # Charcoal Gray
        (128, 128, 0),    # Teal
        (0, 0, 128),      # Maroon
        (130, 0, 75),     # Indigo
        (34, 139, 34),    # Forest Green
        (0, 85, 204),     # Burnt Orange
        (149, 146, 209),  # Dusty Rose
        (235, 206, 135),  # Sky Blue
        (250, 230, 230),  # Lavender
        (191, 226, 159),  # Seafoam Green
        (185, 218, 255),  # Peach
        (255, 204, 204),  # Periwinkle
        (193, 182, 255),  # Blush Pink
        (201, 252, 189),  # Mint Green
        (144, 128, 112),  # Slate Gray
        (112, 25, 25),    # Midnight Blue
        (102, 51, 102),   # Deep Plum
        (0, 128, 128),    # Olive Green
        (171, 71, 0)      # Cobalt Blue
    ]

    # Create color mapping for actions
    action_labels = set(seg['label'] for seg in gt_segments).union(seg['label'] for seg in pred_segments)
    action_color_map = {label: color_palette[i % len(color_palette)] for i, label in enumerate(action_labels)}
    print(f"Action Color Mapping: {action_color_map}")

    # Convert fallback colors to RGB for PIL
    gt_color_rgb = (gt_text_color[2], gt_text_color[1], gt_text_color[0])  # BGR to RGB
    pred_color_rgb = (pred_text_color[2], pred_text_color[1], pred_text_color[0])  # BGR to RGB

    # Load font
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
            print(f"Using font: {font_path}")
        except IOError:
            print(f"Warning: Font {font_path} not found. Trying fallback font.")
    if not font:
        try:
            font = ImageFont.truetype(font_fallback, font_size)
            bar_font = ImageFont.truetype(font_fallback, bar_font_size)
            print(f"Using fallback font: {font_fallback}")
        except IOError:
            print(f"Warning: Fallback font {font_fallback} not found. Using OpenCV default font.")
            font = None
            bar_font = None

    # Fixed window configuration
    window_size = 20.0  # 20-second windows
    num_windows = int(np.ceil(duration / window_size))

    # Define horizontal gap (0.5 inch = 48 pixels at 96 DPI)
    text_bar_gap = 48  # Pixels
    text_x = 10  # Fixed x-position for GT and Pred labels

    frame_idx = 0
    written_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create extended frame with footer
        extended_frame = np.zeros((output_height, frame_width, 3), dtype=np.uint8)
        extended_frame[:frame_height, :, :] = frame
        extended_frame[frame_height:, :, :] = 255  # White footer

        # Calculate current timestamp
        timestamp = frame_idx / fps

        # Determine current window
        window_idx = int(timestamp // window_size)
        window_start = window_idx * window_size
        window_end = min(window_start + window_size, duration)
        window_duration = window_end - window_start
        window_timestamp = timestamp - window_start  # Relative timestamp within window

        # Find active GT actions (for text overlay)
        gt_labels = [seg['label'] for seg in gt_segments if seg['start'] <= timestamp <= seg['end']]
        gt_text = "GT: " + ", ".join(gt_labels) if gt_labels else ""

        # Find active predicted actions (for text overlay)
        pred_labels = [seg['label'] for seg in pred_segments if seg['start'] <= timestamp <= seg['end']]
        pred_text = "Pred: " + ", ".join(pred_labels) if pred_labels else ""

        # Draw GT and prediction bars in footer (within current window, using original animation)
        footer_y = frame_height
        gt_bar_y = footer_y + int(0.2 * footer_height)  # GT bar position
        pred_bar_y = footer_y + int(0.5 * footer_height)  # Pred bar position
        bar_height = int(VIS_CONFIG['video_bar_height'] * footer_height)

        # Calculate text width for GT and Pred labels to determine bar start
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
        bar_start_x = text_x + max_text_width + text_bar_gap  # Bars start after text + 0.5-inch gap
        bar_width = frame_width - bar_start_x  # Adjust bar width to fit remaining space

        # Draw bars with action-specific colors
        for seg in gt_segments:
            if seg['start'] <= window_end and seg['end'] >= window_start:
                start_t = max(seg['start'], window_start)
                end_t = min(seg['end'], window_start + window_timestamp)  # Original animation
                start_x = bar_start_x + int(((start_t - window_start) / window_duration) * bar_width)
                end_x = bar_start_x + int(((end_t - window_start) / window_duration) * bar_width)
                if end_x > start_x:
                    cv2.rectangle(
                        extended_frame,
                        (start_x, gt_bar_y),
                        (end_x, gt_bar_y + bar_height),
                        action_color_map[seg['label']],  # Action-specific color
                        -1
                    )

        for seg in pred_segments:
            if seg['start'] <= window_end and seg['end'] >= window_start:
                start_t = max(seg['start'], window_start)
                end_t = min(seg['end'], window_start + window_timestamp)  # Original animation
                start_x = bar_start_x + int(((start_t - window_start) / window_duration) * bar_width)
                end_x = bar_start_x + int(((end_t - window_start) / window_duration) * bar_width)
                if end_x > start_x:
                    cv2.rectangle(
                        extended_frame,
                        (start_x, pred_bar_y),
                        (end_x, pred_bar_y + bar_height),
                        action_color_map[seg['label']],  # Action-specific color
                        -1
                    )

        if font:
            # Convert frame to PIL image
            frame_rgb = cv2.cvtColor(extended_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_image)

            # Draw frame number and FPS at top center
            frame_info = f"Frame: {frame_idx} | FPS: {fps:.2f}"
            frame_text_bbox = draw.textbbox((0, 0), frame_info, font=font)
            frame_text_width = frame_text_bbox[2] - frame_text_bbox[0]
            frame_text_x = (frame_width - frame_text_width) // 2
            draw.text((frame_text_x, 10), frame_info, font=font, fill=(0, 0, 0))

            # Draw window timestamp range at top of footer
            window_info = f"{window_start:.1f}s - {window_end:.1f}s"
            window_text_bbox = draw.textbbox((0, 0), window_info, font=bar_font)
            window_text_width = window_text_bbox[2] - window_text_bbox[0]
            window_text_x = (frame_width - window_text_width) // 2
            draw.text((window_text_x, footer_y + 10), window_info, font=bar_font, fill=(0, 0, 0))

            # Draw GT text in video only if there are actions
            if gt_text:
                gt_y = int(frame_height * VIS_CONFIG['video_gt_text_y'])
                draw.text((10, gt_y), gt_text, font=font, fill=gt_color_rgb)

            # Draw predicted text in video only if there are actions
            if pred_text:
                pred_y = int(frame_height * VIS_CONFIG['video_pred_text_y'])
                draw.text((10, pred_y), pred_text, font=font, fill=pred_color_rgb)

            # Draw GT and Pred labels in footer
            draw.text((text_x, gt_bar_y + bar_height // 2), "GT", font=bar_font, fill=gt_color_rgb)
            draw.text((text_x, pred_bar_y + bar_height // 2), "Pred", font=bar_font, fill=pred_color_rgb)

            # Convert back to OpenCV frame
            extended_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            # Fallback to OpenCV font
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

        # Write frame to output video
        out.write(extended_frame)
        written_frames += 1
        frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    print(f"[✅ Saved Annotated Video]: {output_path}, Written Frames={written_frames}")
    print("Note: If .avi is not playable, convert to .mp4 using FFmpeg:")
    print(f"ffmpeg -i {output_path} -vcodec libx264 -acodec aac {output_path.replace('.avi', '.mp4')}")








def visualize_action_lengths(
    video_id: str,
    pred_segments: List[Dict],
    gt_segments: List[Dict],
    video_path: str,
    duration: float,
    save_dir: str = VIS_CONFIG['save_dir'],
    frame_interval: float = VIS_CONFIG['frame_interval']
) -> None:
    """
    Generate a visualization plot comparing ground truth and predicted action lengths with video frames.

    Args:
        video_id: Video identifier (e.g., 'my_video').
        pred_segments: List of predicted segments with 'label', 'start', 'end', 'duration', 'score'.
        gt_segments: List of ground truth segments with 'label', 'start', 'end', 'duration'.
        video_path: Path to the input video file.
        duration: Total duration of the video in seconds.
        save_dir: Directory to save the output image.
        frame_interval: Time interval between sampled frames (seconds).
    """
    os.makedirs(save_dir, exist_ok=True)

    # Calculate frame sampling times
    num_frames = int(duration / frame_interval) + 1
    if num_frames > VIS_CONFIG['max_frames']:
        frame_interval = duration / (VIS_CONFIG['max_frames'] - 1)
        num_frames = VIS_CONFIG['max_frames']
        print(f"Warning: Video duration ({duration:.1f}s) requires {num_frames} frames. Adjusted frame_interval to {frame_interval:.2f}s.")
    
    frame_times = np.linspace(0, duration, num_frames, endpoint=False)

    # Load video frames
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
                # Resize frame to reduce memory usage
                frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
                frames.append(frame)
            else:
                frames.append(np.ones((100, 100, 3), dtype=np.uint8) * 255)
        cap.release()

    # Initialize figure
    fig = plt.figure(figsize=(num_frames * VIS_CONFIG['frame_scale_factor'], 6), constrained_layout=True)
    gs = fig.add_gridspec(3, num_frames, height_ratios=[3, 1, 1])

    # Plot frames
    for i, (t, frame) in enumerate(zip(frame_times, frames)):
        ax = fig.add_subplot(gs[0, i])
        
        # Check if frame falls within GT or predicted segments
        gt_hit = any(seg['start'] <= t <= seg['end'] for seg in gt_segments)
        pred_hit = any(seg['start'] <= t <= seg['end'] for seg in pred_segments)
        
        # Set border color
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

    # Plot ground truth bar
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

    # Plot prediction bar
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

    # Save plot
    jpg_path = os.path.join(save_dir, f"viz_{video_id}_{opt['exp']}.png")  # Use PNG
    plt.savefig(jpg_path, dpi=100, bbox_inches='tight')  # Lower DPI
    print(f"[✅ Saved Visualization]: {jpg_path}")
    plt.close()



def train_one_epoch(opt, model, train_dataset, optimizer, warmup=False):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt['batch_size'], shuffle=True,
                                               num_workers=0, pin_memory=True, drop_last=False)
    epoch_cost = 0
    epoch_cost_cls = 0
    epoch_cost_reg = 0
    epoch_cost_snip = 0
    
    total_iter = len(train_dataset) // opt['batch_size']
    cls_loss = MultiCrossEntropyLoss(focal=True)
    snip_loss = MultiCrossEntropyLoss(focal=True)
    for n_iter, (input_data, cls_label, reg_label, snip_label) in enumerate(tqdm(train_loader)):
        if warmup:
            for g in optimizer.param_groups:
                g['lr'] = n_iter * (opt['lr']) / total_iter
        
        act_cls, act_reg, snip_cls = model(input_data.float().cuda())
        
        act_cls.register_hook(partial(cls_loss.collect_grad, cls_label))
        snip_cls.register_hook(partial(snip_loss.collect_grad, snip_label))
        
        cost_reg = 0
        cost_cls = 0

        loss = cls_loss_func_(cls_loss, cls_label, act_cls)
        cost_cls = loss
        epoch_cost_cls += cost_cls.detach().cpu().numpy()
               
        loss = regress_loss_func(reg_label, act_reg)
        cost_reg = loss
        epoch_cost_reg += cost_reg.detach().cpu().numpy()

        loss = cls_loss_func_(snip_loss, snip_label, snip_cls)
        cost_snip = loss
        epoch_cost_snip += cost_snip.detach().cpu().numpy()
        
        cost = opt['alpha'] * cost_cls + opt['beta'] * cost_reg + opt['gamma'] * cost_snip
        epoch_cost += cost.detach().cpu().numpy()

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
                
    return n_iter, epoch_cost, epoch_cost_cls, epoch_cost_reg, epoch_cost_snip

def eval_one_epoch(opt, model, test_dataset):
    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(opt, model, test_dataset)
        
    result_dict = eval_map_nms(opt, test_dataset, output_cls, output_reg, labels_cls, labels_reg)
    output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}
    outfile = open(opt["result_file"].format(opt['exp']), "w")
    json.dump(output_dict, outfile, indent=2)
    outfile.close()
    
    IoUmAP = evaluation_detection(opt, verbose=False)
    IoUmAP_5 = sum(IoUmAP[0:]) / len(IoUmAP[0:])

    return cls_loss, reg_loss, tot_loss, IoUmAP_5

def train(opt):
    writer = SummaryWriter()
    model = MYNET(opt).cuda()
    
    rest_of_model_params = [param for name, param in model.named_parameters() if "history_unit" not in name]
    optimizer = optim.Adam([{'params': model.history_unit.parameters(), 'lr': 1e-6}, {'params': rest_of_model_params}], lr=opt["lr"], weight_decay=opt["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["lr_step"])
    
    train_dataset = VideoDataSet(opt, subset="train")
    test_dataset = VideoDataSet(opt, subset=opt['inference_subset'])
    
    warmup = False
    
    for n_epoch in range(opt['epoch']):
        if n_epoch >= 1:
            warmup = False
        
        n_iter, epoch_cost, epoch_cost_cls, epoch_cost_reg, epoch_cost_snip = train_one_epoch(opt, model, train_dataset, optimizer, warmup)
            
        writer.add_scalars('data/cost', {'train': epoch_cost / (n_iter + 1)}, n_epoch)
        print("training loss(epoch %d): %.03f, cls - %f, reg - %f, snip - %f, lr - %f" % (n_epoch,
                                                                                         epoch_cost / (n_iter + 1),
                                                                                         epoch_cost_cls / (n_iter + 1),
                                                                                         epoch_cost_reg / (n_iter + 1),
                                                                                         epoch_cost_snip / (n_iter + 1),
                                                                                         optimizer.param_groups[-1]["lr"]))
        
        scheduler.step()
        model.eval()
        
        cls_loss, reg_loss, tot_loss, IoUmAP_5 = eval_one_epoch(opt, model, test_dataset)
        
        writer.add_scalars('data/mAP', {'test': IoUmAP_5}, n_epoch)
        print("testing loss(epoch %d): %.03f, cls - %f, reg - %f, mAP Avg - %f" % (n_epoch, tot_loss, cls_loss, reg_loss, IoUmAP_5))
                    
        state = {'epoch': n_epoch + 1, 'state_dict': model.state_dict()}
        torch.save(state, opt["checkpoint_path"] + "/" + opt["exp"] + "_checkpoint_" + str(n_epoch + 1) + ".pth.tar")
        if IoUmAP_5 > model.best_map:
            model.best_map = IoUmAP_5
            torch.save(state, opt["checkpoint_path"] + "/" + opt["exp"] + "_ckp_best.pth.tar")
            
        model.train()
                
    writer.close()
    return model.best_map

def eval_frame(opt, model, dataset):
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt['batch_size'], shuffle=False,
                                              num_workers=0, pin_memory=True, drop_last=False)
    
    labels_cls = {}
    labels_reg = {}
    output_cls = {}
    output_reg = {}
    for video_name in dataset.video_list:
        labels_cls[video_name] = []
        labels_reg[video_name] = []
        output_cls[video_name] = []
        output_reg[video_name] = []
        
    start_time = time.time()
    total_frames = 0
    epoch_cost = 0
    epoch_cost_cls = 0
    epoch_cost_reg = 0
    
    for n_iter, (input_data, cls_label, reg_label, _) in enumerate(tqdm(test_loader)):
        act_cls, act_reg, _ = model(input_data.float().cuda())
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
            labels_cls[video_name] += [cls_label[b, :].numpy()]
            labels_reg[video_name] += [reg_label[b, :].numpy()]
        
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

def eval_map_nms(opt, dataset, output_cls, output_reg, labels_cls, labels_reg):
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

def eval_map_supnet(opt, dataset, output_cls, output_reg, labels_cls, labels_reg):
    model = SuppressNet(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/ckp_best_suppress.pth.tar")
    base_dict = checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
    
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
            
            minput = conf_queue.unsqueeze(0)
            suppress_conf = model(minput.cuda())
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

def test_frame(opt, video_name=None):
    model = MYNET(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/ckp_best.pth.tar")
    base_dict = checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
    
    dataset = VideoDataSet(opt, subset=opt['inference_subset'], video_name=video_name)
    outfile = h5py.File(opt['frame_result_file'].format(opt['exp']), 'w')
    
    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(opt, model, dataset)
    
    print("testing loss: %f, cls_loss: %f, reg_loss: %f" % (tot_loss, cls_loss, reg_loss))
    
    for video_name in dataset.video_list:
        o_cls = output_cls[video_name]
        o_reg = output_reg[video_name]
        l_cls = labels_cls[video_name]
        l_reg = labels_reg[video_name]
        
        dset_predcls = outfile.create_dataset(video_name + '/pred_cls', o_cls.shape, maxshape=o_cls.shape, chunks=True, dtype=np.float32)
        dset_predcls[:, :] = o_cls[:, :]
        dset_predreg = outfile.create_dataset(video_name + '/pred_reg', o_reg.shape, maxshape=o_reg.shape, chunks=True, dtype=np.float32)
        dset_predreg[:, :] = o_reg[:, :]
        dset_labelcls = outfile.create_dataset(video_name + '/label_cls', l_cls.shape, maxshape=l_cls.shape, chunks=True, dtype=np.float32)
        dset_labelcls[:, :] = l_cls[:, :]
        dset_labelreg = outfile.create_dataset(video_name + '/label_reg', l_reg.shape, maxshape=l_reg.shape, chunks=True, dtype=np.float32)
        dset_labelreg[:, :] = l_reg[:, :]
    outfile.close()
    
    print("working time : {}s, {}fps, {} frames".format(working_time, total_frames / working_time, total_frames))
    return cls_loss, reg_loss, tot_loss

def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return forward_orig(*args, **kwargs)

    m.forward = wrap

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

def test(opt, video_name=None):
    model = MYNET(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/" + opt['exp'] + "_ckp_best.pth.tar")
    base_dict = checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
    
    dataset = VideoDataSet(opt, subset=opt['inference_subset'], video_name=video_name)
    
    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(opt, model, dataset)
    
    if opt["pptype"] == "nms":
        result_dict = eval_map_nms(opt, dataset, output_cls, output_reg, labels_cls, labels_reg)
    if opt["pptype"] == "net":
        result_dict = eval_map_supnet(opt, dataset, output_cls, output_reg, labels_cls, labels_reg)
    output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}
    outfile = open(opt["result_file"].format(opt['exp']), "w")
    json.dump(output_dict, outfile, indent=2)
    outfile.close()
    
    mAP = evaluation_detection(opt)
    
    # Compare predicted and ground truth action lengths
    if video_name:
        print("\nComparing Predicted and Ground Truth Action Lengths for Video:", video_name)
        with open(opt["video_anno"].format(opt["split"]), 'r') as f:
            anno_data = json.load(f)
        gt_annotations = anno_data['database'][video_name]['annotations']
        duration = anno_data['database'][video_name]['duration']
        
        gt_segments = []
        for anno in gt_annotations:
            start, end = anno['segment']
            label = anno['label']
            duration_seg = end - start
            gt_segments.append({'label': label, 'start': start, 'end': end, 'duration': duration_seg})
        
        pred_segments = []
        for pred in result_dict[video_name]:
            start, end = pred['segment']
            label = pred['label']
            score = pred['score']
            duration_seg = end - start
            pred_segments.append({'label': label, 'start': start, 'end': end, 'duration': duration_seg, 'score': score})
        
        # Print comparison table
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
        
        print("\n{:<20} {:<30} {:<30} {:<15} {:<10}".format(
            "Action Label", "Predicted Segment (s)", "Ground Truth Segment (s)", "Duration Diff (s)", "IoU"))
        print("-" * 105)
        for match in matches:
            pred = match['pred']
            gt = match['gt']
            iou = match['iou']
            if pred and gt:
                label = pred['label'] if pred['label'] == gt['label'] else f"{pred['label']} (GT: {gt['label']})"
                pred_str = f"[{pred['start']:.2f}, {pred['end']:.2f}] ({pred['duration']:.2f}s)"
                gt_str = f"[{gt['start']:.2f}, {gt['end']:.2f}] ({gt['duration']:.2f}s)"
                duration_diff = pred['duration'] - gt['duration']
                print("{:<20} {:<30} {:<30} {:<15.2f} {:<10.2f}".format(
                    label, pred_str, gt_str, duration_diff, iou))
            elif pred:
                pred_str = f"[{pred['start']:.2f}, {pred['end']:.2f}] ({pred['duration']:.2f}s)"
                print("{:<20} {:<30} {:<30} {:<15} {:<10.2f}".format(
                    pred['label'], pred_str, "None", "N/A", iou))
            elif gt:
                gt_str = f"[{gt['start']:.2f}, {gt['end']:.2f}] ({gt['duration']:.2f}s)"
                print("{:<20} {:<30} {:<30} {:<15} {:<10.2f}".format(
                    gt['label'], "None", gt_str, "N/A", iou))
        
        # Summarize
        matched_count = sum(1 for m in matches if m['pred'] and m['gt'])
        avg_duration_diff = np.mean([m['pred']['duration'] - m['gt']['duration'] for m in matches if m['pred'] and m['gt']]) if matched_count > 0 else 0
        avg_iou = np.mean([m['iou'] for m in matches if m['iou'] > 0]) if any(m['iou'] > 0 for m in matches) else 0
        print(f"\nSummary:")
        print(f"- Total Predictions: {len(pred_segments)}")
        print(f"- Total Ground Truth: {len(gt_segments)}")
        print(f"- Matched Segments: {matched_count}")
        print(f"- Average Duration Difference (Matched): {avg_duration_diff:.2f}s")
        print(f"- Average IoU (Matched): {avg_iou:.2f}")

        # Generate static visualization
        video_path = opt.get('video_path', '')
        if os.path.exists(video_path):
            visualize_action_lengths(
                video_id=video_name,
                pred_segments=pred_segments,
                gt_segments=gt_segments,
                video_path=video_path,
                duration=duration
            )
            # Generate annotated video
            annotate_video_with_actions(
                video_id=video_name,
                pred_segments=pred_segments,
                gt_segments=gt_segments,
                video_path=video_path
            )
        else:
            print(f"Warning: Video path {video_path} not found. Skipping visualization and video annotation.")

    return mAP

def test_online(opt, video_name=None):
    model = MYNET(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/ckp_best.pth.tar")
    base_dict = checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
    
    sup_model = SuppressNet(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/ckp_best_suppress.pth.tar")
    base_dict = checkpoint['state_dict']
    sup_model.load_state_dict(base_dict)
    sup_model.eval()
    
    dataset = VideoDataSet(opt, subset=opt['inference_subset'], video_name=video_name)
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1, shuffle=False,
                                              num_workers=0, pin_memory=True, drop_last=False)
    
    result_dict = {}
    proposal_dict = []
    
    num_class = opt["num_of_class"]
    unit_size = opt['segment_size']
    threshold = opt['threshold']
    anchors = opt['anchors']
    
    start_time = time.time()
    total_frames = 0
    
    for video_name in dataset.video_list:
        input_queue = torch.zeros((unit_size, opt['feat_dim']))
        sup_queue = torch.zeros(((unit_size, num_class - 1)))
    
        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = 100.0 * video_time / duration
        
        for idx in range(0, duration):
            total_frames += 1
            input_queue[:-1, :] = input_queue[1:, :].clone()
            input_queue[-1:, :] = dataset._get_base_data(video_name, idx, idx + 1)
            
            minput = input_queue.unsqueeze(0)
            act_cls, act_reg, _ = model(minput.cuda())
            act_cls = torch.softmax(act_cls, dim=-1)
            
            cls_anc = act_cls.squeeze(0).detach().cpu().numpy()
            reg_anc = act_reg.squeeze(0).detach().cpu().numpy()
            
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
                
            sup_queue[:-1, :] = sup_queue[1:, :].clone()
            sup_queue[-1, :] = 0
            for proposal in proposal_anc_dict:
                cls_idx = dataset.label_name.index(proposal['label'])
                sup_queue[-1, cls_idx] = proposal["score"]
            
            minput = sup_queue.unsqueeze(0)
            suppress_conf = sup_model(minput.cuda())
            suppress_conf = suppress_conf.squeeze(0).detach().cpu().numpy()
            
            for cls in range(0, num_class - 1):
                if suppress_conf[cls] > opt['sup_threshold']:
                    for proposal in proposal_anc_dict:
                        if proposal['label'] == dataset.label_name[cls]:
                            if check_overlap_proposal(proposal_dict, proposal, overlapThresh=opt['soft_nms']) is None:
                                proposal_dict.append(proposal)
            
        result_dict[video_name] = proposal_dict
        proposal_dict = []
    
    end_time = time.time()
    working_time = end_time - start_time
    print("working time : {}s, {}fps, {} frames".format(working_time, total_frames / working_time, total_frames))
    
    output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}
    outfile = open(opt["result_file"].format(opt['exp']), "w")
    json.dump(output_dict, outfile, indent=2)
    outfile.close()
    
    mAP = evaluation_detection(opt)
    return mAP

def main(opt, video_name=None):
    max_perf = 0
    if not video_name and 'video_name' in opt:
        video_name = opt['video_name']
    
    if opt['mode'] == 'train':
        max_perf = train(opt)
    if opt['mode'] == 'test':
        max_perf = test(opt, video_name=video_name)
    if opt['mode'] == 'test_frame':
        max_perf = test_frame(opt, video_name=video_name)
    if opt['mode'] == 'test_online':
        max_perf = test_online(opt, video_name=video_name)
    if opt['mode'] == 'eval':
        max_perf = evaluation_detection(opt)
        
    return max_perf

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "/" + opt["exp"] + "_opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()
    
    if opt['seed'] >= 0:
        seed = opt['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
           
    opt['anchors'] = [int(item) for item in opt['anchors'].split(',')]
           
    video_name = opt.get('video_name', None)
    main(opt, video_name=video_name)
    while(opt['wterm']):
        pass