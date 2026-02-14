import time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import csv
import os
from collections import deque
import psutil
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# === UNet Model Definition ===
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up3 = DoubleConv(512 + 256, 256)
        self.up2 = DoubleConv(256 + 128, 128)
        self.up1 = DoubleConv(128 + 64, 64)

        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        c1 = self.down1(x)
        x = self.pool(c1)

        c2 = self.down2(x)
        x = self.pool(c2)

        c3 = self.down3(x)
        x = self.pool(c3)

        x = self.down4(x)

        x = self.up(x)
        x = torch.cat([x, c3], dim=1)

        x = self.up3(x)
        x = self.up(x)
        x = torch.cat([x, c2], dim=1)

        x = self.up2(x)
        x = self.up(x)
        x = torch.cat([x, c1], dim=1)

        x = self.up1(x)
        return torch.sigmoid(self.final(x))

# =============================================================
# ---------------------- CONFIG SECTION -----------------------
# =============================================================
VIDEO_SOURCE = r"C:\Users\athan\Downloads\Video\ASMR Relaxing Truck Driving in a Rain Storm at Night in Atlanta - Sleepy Trucker (720p, h264).mp4" # Change to your video path
UNET_MODEL_PATH = r"C:\Users\athan\Desktop\Final Project\my_trained_model\lane_unet_final.pth"

# Lane smoothing factor (0=no smoothing, 0.7 = strong smoothing)
LANE_SMOOTHING_FACTOR = 0.7

# If True, show the warped bird's-eye lane mask in a separate window (debug)
SHOW_DEBUG_WINDOWS = False

# === INSET PREVIEW CONTROLS ===
SHOW_LANE_INSET_METRICS = True
LANE_INSET_MODE = 'bw'          # 'bw' for black/white binary, 'color' for colored overlay
LANE_INSET_SCALE = 0.25

# === METRICS SETTINGS ===
SAVE_METRICS_CSV = True
LOG_BUFFER_SIZE = 10

# =============================================================
# ---------------------- METRICS CLASSES ----------------------
# =============================================================
class CoreMetricsExporter:
    def __init__(self, image_height, image_width):
        self.frame_count = 0
        self.image_height = image_height
        self.image_width = image_width
        
        os.makedirs('unet_video_metrics', exist_ok=True)
        
        timestamp = int(time.time())
        self.csv_filename = f'unet_video_metrics/unet_video_metrics_{timestamp}.csv'
        
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'frame', 'timestamp', 'left_detected', 'right_detected',
                'lane_center_bottom', 'vehicle_position', 'lane_offset_pixels',
                'curvature', 'steering_angle', 'environment', 'model_inference_time'
            ])
        
        print(f"🎯 UNet Video metrics will be saved to: {self.csv_filename}")
    
    def save_core_metrics(self, left_fit, right_fit, curvature, steering_angle, environment, inference_time):
        self.frame_count += 1
        
        y_bottom = self.image_height - 1
        left_detected = left_fit is not None
        right_detected = right_fit is not None
        
        STANDARD_LANE_WIDTH = 350
        MAX_REASONABLE_OFFSET = 200
        
        left_x_bottom = 0.0
        right_x_bottom = 0.0
        
        if left_fit is not None:
            left_x_bottom = left_fit[0]*y_bottom**2 + left_fit[1]*y_bottom + left_fit[2]
        
        if right_fit is not None:
            right_x_bottom = right_fit[0]*y_bottom**2 + right_fit[1]*y_bottom + right_fit[2]
        
        if left_detected and right_detected:
            lane_center_bottom = (left_x_bottom + right_x_bottom) / 2
        elif left_detected:
            lane_center_bottom = left_x_bottom + (STANDARD_LANE_WIDTH / 2)
        elif right_detected:
            lane_center_bottom = right_x_bottom - (STANDARD_LANE_WIDTH / 2)
        else:
            lane_center_bottom = self.image_width / 2

        vehicle_position = self.image_width / 2
        lane_offset_pixels = vehicle_position - lane_center_bottom
        lane_offset_pixels = max(min(lane_offset_pixels, MAX_REASONABLE_OFFSET), -MAX_REASONABLE_OFFSET)
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                self.frame_count, time.time(),
                left_detected, right_detected,
                lane_center_bottom, vehicle_position, lane_offset_pixels,
                curvature if curvature is not None else 0.0,
                steering_angle, environment, inference_time
            ])
        
        return {
            'frame': self.frame_count,
            'left_detected': left_detected,
            'right_detected': right_detected,
            'lane_center': lane_center_bottom,
            'offset': lane_offset_pixels,
            'curvature': curvature,
            'steering': steering_angle,
            'inference_time': inference_time
        }

# =============================================================
# ----------------- PERSPECTIVE TRANSFORM ---------------------
# =============================================================
def create_perspective_transform(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create perspective transform matrices."""
    src = np.float32([
        [width * 0.45, height * 0.6],
        [width * 0.55, height * 0.6],
        [width * 0.95, height * 0.95],
        [width * 0.05, height * 0.95],
    ])
    dst = np.float32([
        [0, 0], [width, 0], [width, height], [0, height]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

# =============================================================
# --------------------- LANE PROCESSING -----------------------
# =============================================================
def fit_lane_lines(binary):
    """Detect left and right lane lines using sliding windows."""
    if binary is None or binary.size == 0:
        return None, None, np.array([]), np.array([]), np.array([]), np.array([])

    histogram = np.sum(binary[binary.shape[0]//2:, :], axis=0)
    if histogram.size == 0:
        return None, None, np.array([]), np.array([]), np.array([]), np.array([])

    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows, margin, minpix = 9, 100, 50
    window_height = binary.shape[0] // nwindows
    nonzero = binary.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

    if len(nonzerox) == 0 or len(nonzeroy) == 0:
        return None, None, np.array([]), np.array([]), np.array([]), np.array([])

    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []

    for window in range(nwindows):
        win_y_low = binary.shape[0] - (window + 1) * window_height
        win_y_high = binary.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate indices safely
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
    except ValueError:
        left_lane_inds = np.array([], dtype=int)
    try:
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        right_lane_inds = np.array([], dtype=int)

    left_fit = None
    right_fit = None
    leftx, lefty, rightx, righty = np.array([]), np.array([]), np.array([]), np.array([])

    if len(left_lane_inds) > 50:
        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        if len(leftx) > 50 and len(lefty) > 50:
            left_fit = np.polyfit(lefty, leftx, 2)

    if len(right_lane_inds) > 50:
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
        if len(rightx) > 50 and len(righty) > 50:
            right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, leftx, lefty, rightx, righty

# =============================================================
# --------------------- METRICS HELPERS -----------------------
# =============================================================
def calculate_curvature(left_fit, right_fit, height: int) -> float:
    """Return radius of curvature in meters."""
    ym_per_pix, xm_per_pix = 30 / 720, 3.7 / 700
    
    if left_fit is None and right_fit is None:
        return 1000.0
    
    ploty = np.linspace(0, height - 1, height)
    curvature = 0.0

    if left_fit is not None:
        left_fit_cr = np.polyfit(ploty * ym_per_pix,
                                (left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]) * xm_per_pix, 2)
        y_eval = np.max(ploty) * ym_per_pix
        left_curvature = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / abs(2 * left_fit_cr[0])
        curvature = left_curvature

    if right_fit is not None and left_fit is None:
        right_fit_cr = np.polyfit(ploty * ym_per_pix,
                                 (right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]) * xm_per_pix, 2)
        y_eval = np.max(ploty) * ym_per_pix
        right_curvature = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) / abs(2 * right_fit_cr[0])
        curvature = right_curvature

    return curvature if curvature > 0 else 1000.0

def calculate_offset(left_fit, right_fit, width: int, height: int) -> float:
    """Return lateral vehicle offset (+ right, - left) in meters."""
    xm_per_pix = 3.7 / 700
    if left_fit is None or right_fit is None:
        return 0.0
    y_eval = height - 1
    left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_center = (left_x + right_x) / 2.0
    vehicle_center = width / 2.0
    return (vehicle_center - lane_center) * xm_per_pix

# =============================================================
# ------------------- VISUALIZATION ---------------------------
# =============================================================
def make_lane_inset(lane_mask: np.ndarray, mode: str = 'bw', scale: float = 0.25) -> np.ndarray:
    """Build a small 3-channel preview image of the lane mask."""
    if lane_mask.ndim == 3:
        lane_gray = cv2.cvtColor(lane_mask, cv2.COLOR_BGR2GRAY)
    else:
        lane_gray = lane_mask

    if mode == 'color':
        preview = np.zeros((lane_gray.shape[0], lane_gray.shape[1], 3), dtype=np.uint8)
        preview[lane_gray > 0] = (0, 255, 0)
    else:  # bw
        preview = cv2.cvtColor(lane_gray, cv2.COLOR_GRAY2BGR)

    h = max(1, int(lane_gray.shape[0] * scale))
    w = max(1, int(lane_gray.shape[1] * scale))
    preview = cv2.resize(preview, (w, h), interpolation=cv2.INTER_NEAREST)
    return preview

def visualize_lanes(original: np.ndarray,
                    lane_mask: np.ndarray,
                    fit_lines: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                    Minv: np.ndarray,
                    metrics: Dict,
                    lane_inset: Optional[np.ndarray] = None) -> np.ndarray:
    """Render lane polygon & metrics onto original frame."""
    warp_zero = np.zeros_like(lane_mask).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    if fit_lines is not None:
        left_fitx, right_fitx, ploty = fit_lines
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right)).astype(np.int32)
        cv2.fillPoly(color_warp, [pts], (0, 255, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (original.shape[1], original.shape[0]))
    result = cv2.addWeighted(original, 1, newwarp, 0.3, 0)
    result = overlay_metrics(result, metrics, lane_inset=lane_inset)
    return result

def overlay_metrics(frame: np.ndarray, metrics: Dict, lane_inset: Optional[np.ndarray] = None) -> np.ndarray:
    """Draw lane metrics with optional lane inset preview."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    y0, dy = 30, 25

    rect_h = 200
    rect_w = 450

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (rect_w, rect_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    cv2.putText(frame, "UNet Lane Detection Metrics:", (10, y0), font, 0.7, (0, 200, 255), 2)

    # Detection quality
    left_px = metrics.get("left_pixels", 0)
    right_px = metrics.get("right_pixels", 0)
    quality, color = "GOOD", (0, 255, 0)
    if left_px < 500 or right_px < 500:
        quality, color = "WARNING", (0, 255, 255)
    if left_px < 300 or right_px < 300:
        quality, color = "POOR", (0, 0, 255)

    cv2.putText(frame, f"Detection Quality: {quality}", (10, y0 + dy), font, 0.6, color, 2)
    cv2.putText(frame, f"Left Lane Pixels: {left_px}", (10, y0 + dy * 2), font, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Right Lane Pixels: {right_px}", (10, y0 + dy * 3), font, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Curvature: {metrics.get('curvature', 0.0):.1f} m", (10, y0 + dy * 4), font, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Vehicle Offset: {metrics.get('offset', 0.0):.2f} m", (10, y0 + dy * 5), font, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Steering: {metrics.get('steering', 0.0):.3f} rad", (10, y0 + dy * 6), font, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {metrics.get('fps', 0.0):.1f}", (frame.shape[1] - 150, 30), font, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Inference: {metrics.get('inference_time', 0.0)*1000:.1f} ms", (frame.shape[1] - 150, 60), font, 0.7, (0, 255, 255), 2)

    # Detection status
    left_detected = metrics.get("left_detected", False)
    right_detected = metrics.get("right_detected", False)
    status = "BOTH" if left_detected and right_detected else "ONE" if left_detected or right_detected else "NONE"
    status_color = (0, 255, 0) if status == "BOTH" else (0, 255, 255) if status == "ONE" else (0, 0, 255)
    cv2.putText(frame, f"Detection: {status}", (10, y0 + dy * 7), font, 0.6, status_color, 2)

    # Lane inset preview
    if lane_inset is not None:
        ih, iw = lane_inset.shape[:2]
        x_off = 10
        y_off = rect_h + 10
        if y_off + ih > frame.shape[0]:
            ih = frame.shape[0] - y_off
            lane_inset = lane_inset[:ih, :iw]
        if x_off + iw > frame.shape[1]:
            iw = frame.shape[1] - x_off
            lane_inset = lane_inset[:ih, :iw]
        frame[y_off:y_off+ih, x_off:x_off+iw] = lane_inset
        cv2.rectangle(frame, (x_off, y_off), (x_off+iw, y_off+ih), (0, 255, 0), 1)
        cv2.putText(frame, "UNet Output", (x_off+5, y_off+15), font, 0.5, (0, 255, 0), 1)

    return frame

# =============================================================
# ------------------- STEERING CALCULATION --------------------
# =============================================================
def calculate_steering(left_fit, right_fit, width: int, height: int):
    """Calculate steering angle with PID control."""
    KP, KI, KD = 0.2, 0.0002, 0.8
    MAX_STEER = 0.35
    
    # Static variables for PID
    if not hasattr(calculate_steering, 'prev_error'):
        calculate_steering.prev_error = 0.0
        calculate_steering.integral = 0.0
    
    STANDARD_LANE_WIDTH = 350
    y = height - 1
    
    if left_fit is not None and right_fit is not None:
        left_x = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
        right_x = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
        lane_center = (left_x + right_x) / 2
    elif left_fit is not None:
        left_x = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
        lane_center = left_x + (STANDARD_LANE_WIDTH / 2)
    elif right_fit is not None:
        right_x = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
        lane_center = right_x - (STANDARD_LANE_WIDTH / 2)
    else:
        lane_center = width / 2

    vehicle_center = width / 2
    error = (vehicle_center - lane_center) / max(1, (width/2))

    # PID control
    derivative = error - calculate_steering.prev_error
    calculate_steering.integral += error
    steer = KP * error + KI * calculate_steering.integral + KD * derivative
    steer = max(min(steer, MAX_STEER), -MAX_STEER)
    calculate_steering.prev_error = error
    
    return steer, error

# =============================================================
# ------------------------- MAIN ------------------------------
# =============================================================
def main():
    # ------------------- Video Source -------------------
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # ------------------- Perspective --------------------
    M, Minv = create_perspective_transform(width, height)

    # ------------------- UNet Model ---------------------
    print(f"Loading UNet model: {UNET_MODEL_PATH} ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=device))
    model.eval()

    # ------------------- Image Transform ---------------
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    # ------------------- Metrics Setup ------------------
    metrics_exporter = CoreMetricsExporter(image_height=height, image_width=width)
    
    # Performance CSV setup
    perf_dir = "unet_video_metrics"
    os.makedirs(perf_dir, exist_ok=True)
    perf_file = os.path.join(perf_dir, "unet_video_performance.csv")
    perf_csv = open(perf_file, mode="w", newline="")
    csv_writer = csv.writer(perf_csv)
    csv_writer.writerow([
        "Time", "Error", "Curvature", "Steering", "FrameLatency",
        "CPU%", "LaneDetected", "Env", "LaneOffset", "InferenceTime"
    ])

    # Performance monitoring
    log_buffer = deque()
    steering_history = deque(maxlen=50)
    error_history = deque(maxlen=50)
    inference_time_history = deque(maxlen=50)
    lane_data_history = []

    # ------------------- Main Loop ----------------------
    frame_count = 0
    start_time = time.time()
    prev_left_fit, prev_right_fit = None, None

    print("🚗 UNet Lane Detection on Video Started...")
    print("📊 Tracking: Lane detection, curvature, offset, and computational performance")

    try:
        while cap.isOpened():
            frame_start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # ------------------- UNet Inference ----------------------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            inference_start = time.time()
            input_tensor = transform(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor).squeeze().cpu().numpy()
            inference_time = time.time() - inference_start

            # Process UNet output
            pred_mask = (output > 0.5).astype(np.uint8) * 255
            pred_mask = cv2.resize(pred_mask, (width, height))

            # Warp the UNet output for lane fitting
            warped_mask = cv2.warpPerspective(pred_mask, M, (width, height))

            # ------------------- Lane Fitting ------------------------
            left_fit, right_fit, leftx, lefty, rightx, righty = fit_lane_lines(warped_mask)

            # ------------------- Smoothing ---------------------------
            if prev_left_fit is not None and left_fit is not None:
                left_fit = LANE_SMOOTHING_FACTOR * prev_left_fit + (1 - LANE_SMOOTHING_FACTOR) * left_fit
            if prev_right_fit is not None and right_fit is not None:
                right_fit = LANE_SMOOTHING_FACTOR * prev_right_fit + (1 - LANE_SMOOTHING_FACTOR) * right_fit
            prev_left_fit, prev_right_fit = left_fit, right_fit

            # ------------------- Derived Metrics ---------------------
            ploty = np.linspace(0, height - 1, height)
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] if left_fit is not None else None
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] if right_fit is not None else None

            curvature = calculate_curvature(left_fit, right_fit, height)
            offset = calculate_offset(left_fit, right_fit, width, height)
            steer, error = calculate_steering(left_fit, right_fit, width, height)

            # ------------------- Lane Inset Preview ------------------
            lane_inset = None
            if SHOW_LANE_INSET_METRICS:
                lane_inset = make_lane_inset(warped_mask, mode=LANE_INSET_MODE, scale=LANE_INSET_SCALE)

            # ------------------- Environment Detection ---------------
            brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if brightness < 50:
                environment = "night"
            elif brightness < 100:
                environment = "dull"
            else:
                environment = "day"

            # ------------------- Save Metrics ------------------------
            lane_data = metrics_exporter.save_core_metrics(left_fit, right_fit, curvature, steer, environment, inference_time)
            lane_data_history.append(lane_data)

            # ------------------- Performance Logging -----------------
            processing_time = time.time() - frame_start_time
            cpu_percent = psutil.cpu_percent(interval=None)
            lane_detected = 1 if left_fit is not None and right_fit is not None else 0

            steering_history.append(steer)
            error_history.append(error)
            inference_time_history.append(inference_time)

            log_buffer.append([
                round(time.time() - start_time, 2),
                round(error, 4),
                round(curvature, 2),
                round(steer, 4),
                round(processing_time, 4),
                cpu_percent,
                lane_detected,
                environment,
                round(lane_data['offset'], 1),
                round(inference_time, 4)
            ])

            if len(log_buffer) >= LOG_BUFFER_SIZE:
                csv_writer.writerows(log_buffer)
                log_buffer.clear()

            # ------------------- Visualization -----------------------
            metrics = {
                "left_pixels": len(leftx),
                "right_pixels": len(rightx),
                "curvature": curvature,
                "offset": offset,
                "steering": steer,
                "fps": frame_count / (time.time() - start_time + 1e-9),
                "inference_time": inference_time,
                "left_detected": left_fit is not None,
                "right_detected": right_fit is not None
            }

            # Draw lanes + metrics
            if left_fitx is not None and right_fitx is not None:
                lane_viz = visualize_lanes(frame, warped_mask, (left_fitx, right_fitx, ploty), Minv, metrics, lane_inset=lane_inset)
            else:
                lane_viz = frame.copy()
                cv2.putText(lane_viz, "LANE DETECTION FAILED!", (width // 4, height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                lane_viz = overlay_metrics(lane_viz, metrics, lane_inset=lane_inset)

            # ------------------- Debug Windows -----------------------
            if SHOW_DEBUG_WINDOWS:
                cv2.imshow("UNet Output", pred_mask)
                cv2.imshow("Warped Mask", warped_mask)

            cv2.imshow("UNet Lane Detection", lane_viz)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('p'):
                while True:
                    k2 = cv2.waitKey(0) & 0xFF
                    if k2 in (ord('p'), ord('q'), 27):
                        if k2 in (ord('q'), 27):
                            break
                        else:
                            break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        # ------------------- Cleanup ------------------------
        cap.release()
        cv2.destroyAllWindows()

        if log_buffer:
            csv_writer.writerows(log_buffer)
        perf_csv.close()

        print(f"✅ UNet Video metrics saved to: {metrics_exporter.csv_filename}")
        print(f"📊 UNet Performance data saved to: {perf_file}")
        print(f"🎯 Processed {metrics_exporter.frame_count} frames")

        # Final statistics
        if lane_data_history:
            offsets = [data['offset'] for data in lane_data_history]
            detections = [1 if data['left_detected'] and data['right_detected'] else 0 for data in lane_data_history]
            inference_times = [data['inference_time'] for data in lane_data_history]
            steering_angles = [data['steering'] for data in lane_data_history]
            
            print(f"\n📈 UNet VIDEO FINAL METRICS:")
            print(f"   Average Lane Offset: {np.mean(offsets):.2f} pixels")
            print(f"   Offset Std Dev: {np.std(offsets):.2f} pixels")
            print(f"   Detection Rate: {np.mean(detections)*100:.1f}%")
            print(f"   Average Inference Time: {np.mean(inference_times)*1000:.1f} ms")
            print(f"   Average Steering: {np.mean(steering_angles):.3f} rad")
            print(f"   Total Frames: {len(lane_data_history)}")
            print(f"   Average FPS: {metrics_exporter.frame_count/(time.time() - start_time):.1f}")

        print("🧠 UNet Video processing complete!")

if __name__ == "__main__":
    main()