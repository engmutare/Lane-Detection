# === Thesis-Ready Unet Lane Following ===

from controller import Robot, Camera
from vehicle import Driver
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import torch.nn as nn
import time
import csv
from collections import deque
import psutil
import matplotlib.pyplot as plt
import os

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

# === Metrics Classes (ADDED) ===
class CoreMetricsExporter:
    def __init__(self, image_height, image_width):
        self.frame_count = 0
        self.image_height = image_height
        self.image_width = image_width
        
        os.makedirs('unet_metrics', exist_ok=True)
        
        timestamp = int(time.time())
        self.csv_filename = f'unet_metrics/unet_core_metrics_{timestamp}.csv'
        
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'frame', 'timestamp', 'left_detected', 'right_detected',
                'lane_center_bottom', 'vehicle_position', 'lane_offset_pixels',
                'curvature', 'steering_angle', 'environment', 'model_inference_time'
            ])
        
        print(f"🎯 UNet Core metrics will be saved to: {self.csv_filename}")
    
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

class SimpleVisualizer:
    def __init__(self):
        self.fig = None
        self.axs = None
        self.setup_plots()
        
    def setup_plots(self):
        plt.ion()
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('UNet Lane Following Core Metrics', fontsize=16)
        
    def update_plots(self, steering_history, error_history, detection_history, environment, inference_times):
        if len(steering_history) < 2:
            return
            
        for ax in self.axs.flat:
            ax.clear()
        
        frames = range(len(steering_history))
        
        self.axs[0,0].plot(frames, steering_history, 'b-', linewidth=2)
        self.axs[0,0].set_title('Steering Angle')
        self.axs[0,0].set_ylabel('Steering (rad)')
        self.axs[0,0].grid(True, alpha=0.3)
        
        self.axs[0,1].plot(frames, error_history, 'r-', linewidth=2)
        self.axs[0,1].set_title('Lane Following Error')
        self.axs[0,1].set_ylabel('Error (pixels)')
        self.axs[0,1].grid(True, alpha=0.3)
        
        if detection_history:
            detection_status = []
            for data in detection_history[-20:]:
                if data['left_detected'] and data['right_detected']:
                    detection_status.append(2)
                elif data['left_detected'] or data['right_detected']:
                    detection_status.append(1)
                else:
                    detection_status.append(0)
            
            if detection_status:
                self.axs[1,0].plot(range(len(detection_status)), detection_status, 'g-', linewidth=2)
                self.axs[1,0].set_title('Detection Status (0=None, 1=One, 2=Both)')
                self.axs[1,0].set_ylabel('Status')
                self.axs[1,0].set_ylim(-0.5, 2.5)
                self.axs[1,0].grid(True, alpha=0.3)
        
        if steering_history and error_history and inference_times:
            avg_steering = np.mean(steering_history)
            avg_error = np.mean(error_history)
            avg_inference_time = np.mean(inference_times) * 1000
            detection_rate = np.mean([1 if d['left_detected'] and d['right_detected'] else 0 for d in detection_history[-20:]]) * 100
            
            summary_text = f"""
UNet Performance Summary:
- Avg Steering: {avg_steering:.3f} rad
- Avg Error: {avg_error:.1f} pixels
- Detection Rate: {detection_rate:.1f}%
- Avg Inference: {avg_inference_time:.1f} ms
- Environment: {environment}
"""
            self.axs[1,1].text(0.5, 0.5, summary_text, fontsize=12, ha='center', va='center',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            self.axs[1,1].set_title('UNet Real-time Performance')
            self.axs[1,1].axis('off')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

# === Initialization ===
robot = Robot()
driver = Driver()
timestep = 10

camera = robot.getDevice("camera")
camera.enable(timestep)
width = camera.getWidth()
height = camera.getHeight()

driver.setHazardFlashers(True)
driver.setDippedBeams(True)
driver.setAntifogLights(True)
driver.setWiperMode(1)  # Slow

# === Load Trained Model ===
device = torch.device('cpu')
model = UNet(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load(
    r"C:\Users\athan\Desktop\Final Project\lane_unet_finaldataset.pth", 
    map_location=device))
model.eval()

# === Image Transform ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === PID Constants ===
KP, KI, KD = 0.2, 0.0002, 0.8
MAX_STEER = 0.35
BASE_SPEED = 70
prev_error = 0.0
integral = 0.0

# === Metrics Initialization (ADDED) ===
metrics_exporter = CoreMetricsExporter(image_height=height, image_width=width)
visualizer = SimpleVisualizer()

# Performance monitoring variables
steering_history = deque(maxlen=50)
error_history = deque(maxlen=50)
inference_time_history = deque(maxlen=50)
lane_data_history = []
log_buffer = deque()
LOG_BUFFER_SIZE = 10

# Performance CSV setup
perf_dir = "unet_metrics"
os.makedirs(perf_dir, exist_ok=True)
perf_file = os.path.join(perf_dir, "unet_performance_metrics.csv")
perf_csv = open(perf_file, mode="w", newline="")
csv_writer = csv.writer(perf_csv)
csv_writer.writerow([
    "Time", "Error", "Curvature", "Steering", "FrameLatency",
    "CPU%", "LaneDetected", "Env", "LaneOffset", "InferenceTime"
])

# === Helper Functions (ADDED) ===
def detect_environment_conditions(img):
    brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if brightness < 50:
        return "night"
    elif brightness < 100:
        return "dull"
    return "day"

def compute_curvature_real(fit, y_eval, ym_per_pix=30/720, xm_per_pix=3.7/700):
    if fit is None or len(fit) != 3:
        return 1000.0

    try:
        A, B, C = fit
        A_m = A * xm_per_pix / (ym_per_pix**2)
        B_m = B * xm_per_pix / ym_per_pix

        curvature = ((1 + (2*A_m*y_eval*ym_per_pix + B_m)**2)**1.5) / (np.abs(2*A_m))
        if np.isnan(curvature) or np.isinf(curvature) or curvature <= 0:
            return 1000.0
        return curvature
    except Exception:
        return 1000.0

def calculate_steering_with_error(left_fit, right_fit):
    global prev_error, integral
    if left_fit is None or right_fit is None:
        return 0.0, 0.0

    y_eval = height
    left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_center = (left_x + right_x) / 2
    vehicle_center = width / 2

    error = (vehicle_center - lane_center) / (width / 2)
    derivative = error - prev_error
    integral += error
    steer = KP * error + KI * integral + KD * derivative
    steer = max(min(steer, MAX_STEER), -MAX_STEER)
    prev_error = error
    return steer, error

def visualize_with_metrics(original, pred_mask, lane_data, environment, inference_time):
    overlay = cv2.addWeighted(original, 0.7, cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    
    y_offset = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    background_color = (0, 0, 0)
    
    lane_info = [
        f"Frame: {metrics_exporter.frame_count}",
        f"Detection: {'BOTH' if lane_data['left_detected'] and lane_data['right_detected'] else 'ONE' if lane_data['left_detected'] or lane_data['right_detected'] else 'NONE'}",
        f"Offset: {lane_data['offset']:.1f} px",
        f"Steering: {lane_data['steering']:.3f} rad",
        f"Inference: {inference_time*1000:.1f} ms",
        f"Environment: {environment}"
    ]
    
    for i, text in enumerate(lane_info):
        text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
        cv2.rectangle(overlay, (5, y_offset*i + 5), (5 + text_size[0] + 10, y_offset*i + text_size[1] + 10), 
                     background_color, -1)
        cv2.putText(overlay, text, (10, y_offset*i + 20), font, font_scale, font_color, 1)
    
    return overlay

# === Lane Fitting (ORIGINAL - UNCHANGED) ===
def fit_lane_lines(binary):
    histogram = np.sum(binary[binary.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows, margin, minpix = 9, 100, 50
    window_height = binary.shape[0] // nwindows
    nonzero = binary.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
    leftx_current, rightx_current = leftx_base, rightx_base
    left_inds, right_inds = [], []

    for window in range(nwindows):
        win_y_low = binary.shape[0] - (window + 1) * window_height
        win_y_high = binary.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_inds.append(good_left)
        right_inds.append(good_right)

        if len(good_left) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right]))

    left_inds = np.concatenate(left_inds)
    right_inds = np.concatenate(right_inds)
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 100 else None
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 100 else None
    return left_fit, right_fit

# === Main Loop ===
print("🚗 UNet Lane Following with Metrics Started")
start_time = time.time()

try:
    while robot.step(timestep) != -1:
        driver.step()
        frame_start_time = time.time()
        img = camera.getImage()
        if img is None:
            continue

        bgr = np.frombuffer(img, np.uint8).reshape((height, width, 4))[:, :, :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Environment detection (ADDED)
        environment = detect_environment_conditions(bgr)

        # UNet inference with timing (ADDED timing)
        inference_start = time.time()
        input_tensor = transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor).squeeze().cpu().numpy()
        inference_time = time.time() - inference_start

        pred_mask = (output > 0.5).astype(np.uint8) * 255
        pred_mask = cv2.resize(pred_mask, (width, height))

        left_fit, right_fit = fit_lane_lines(pred_mask)
        
        # Calculate steering with error (MODIFIED to get error)
        steer, error = calculate_steering_with_error(left_fit, right_fit)

        # Curvature calculation (ADDED)
        curvature = compute_curvature_real(left_fit, height) if left_fit is not None else None

        driver.setSteeringAngle(steer)
        driver.setCruisingSpeed(BASE_SPEED)

        # Save metrics (ADDED)
        lane_data = metrics_exporter.save_core_metrics(left_fit, right_fit, curvature, steer, environment, inference_time)
        lane_data_history.append(lane_data)

        # Performance logging (ADDED)
        processing_time = time.time() - frame_start_time
        cpu_percent = psutil.cpu_percent(interval=None)
        lane_detected = 1 if left_fit is not None and right_fit is not None else 0

        steering_history.append(steer)
        error_history.append(error)
        inference_time_history.append(inference_time)

        log_buffer.append([
            round(time.time() - start_time, 2),
            round(error, 4),
            round(curvature or 0, 2),
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

        # Visualization with metrics (MODIFIED)
        overlay = visualize_with_metrics(bgr, pred_mask, lane_data, environment, inference_time)
        cv2.imshow("UNet Lane Following", cv2.resize(overlay, (400, 300)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update plots (ADDED)
        if metrics_exporter.frame_count % 15 == 0 and lane_data_history:
            visualizer.update_plots(steering_history, error_history, lane_data_history, environment, inference_time_history)

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Cleanup (ADDED)
    driver.setCruisingSpeed(0)
    cv2.destroyAllWindows()

    if log_buffer:
        csv_writer.writerows(log_buffer)
    perf_csv.close()

    print(f"✅ UNet Core metrics saved to: {metrics_exporter.csv_filename}")
    print(f"📊 UNet Performance data saved to: {perf_file}")
    print(f"🎯 Processed {metrics_exporter.frame_count} frames")

    # Final statistics (ADDED)
    if lane_data_history:
        offsets = [data['offset'] for data in lane_data_history]
        detections = [1 if data['left_detected'] and data['right_detected'] else 0 for data in lane_data_history]
        inference_times = [data['inference_time'] for data in lane_data_history]
        
        print(f"\n📈 UNet FINAL THESIS METRICS:")
        print(f"   Average Lane Offset: {np.mean(offsets):.2f} pixels")
        print(f"   Offset Std Dev: {np.std(offsets):.2f} pixels")
        print(f"   Detection Rate: {np.mean(detections)*100:.1f}%")
        print(f"   Average Inference Time: {np.mean(inference_times)*1000:.1f} ms")
        print(f"   Total Frames: {len(lane_data_history)}")

    plt.ioff()
    plt.show()
    print("🛑 Vehicle stopped safely.")
    print("🧠 UNet Thesis data collection complete!")