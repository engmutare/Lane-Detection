from controller import Robot, Camera, Speaker
from vehicle import Driver
import numpy as np
import cv2
import datetime
import os

# === Initialization ===
robot = Robot()
driver = Driver()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)
width = camera.getWidth()
height = camera.getHeight()

# Vehicle startup settings
driver.setCruisingSpeed(50)
driver.setHazardFlashers(True)
driver.setDippedBeams(True)
driver.setAntifogLights(True)
driver.setWiperMode(1)

try:
    speaker = robot.getDevice("speaker")
    hazard_buffer = speaker.getSound("hazard.wav")
    speaker_enabled = True
except:
    speaker_enabled = False

# PID Parameters
KP, KI, KD = 0.2, 0.0002, 0.8
MAX_STEER, BASE_SPEED = 0.35, 80
prev_error, integral = 0.0, 0.0

prev_left_fit, prev_right_fit = None, None
SMOOTHING = 0.6

# Create Perspective Transform
def create_perspective_transform():
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

M, Minv = create_perspective_transform()

# Detect lanes in the warped image
def detect_lanes(warped):
    hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
    yellow_mask = cv2.inRange(hls, np.array([15, 40, 100]), np.array([35, 255, 255]))
    white_mask = cv2.inRange(hls, np.array([0, 180, 0]), np.array([180, 255, 255]))
    combined = cv2.bitwise_or(yellow_mask, white_mask)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobel = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobel / (np.max(abs_sobel) + 1e-5))
    gradient_mask = np.zeros_like(scaled_sobel)
    gradient_mask[(scaled_sobel >= 50)] = 255
    lane_mask = cv2.bitwise_or(combined, gradient_mask)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return lane_mask

# Fit polynomial to lane lines
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
    return left_fit, right_fit, leftx, lefty, rightx, righty

# Calculate steering based on lane fits
def calculate_steering(left_fit, right_fit):
    global prev_error, integral, prev_left_fit, prev_right_fit
    if left_fit is not None and prev_left_fit is not None:
        left_fit = SMOOTHING * prev_left_fit + (1 - SMOOTHING) * left_fit
    if right_fit is not None and prev_right_fit is not None:
        right_fit = SMOOTHING * prev_right_fit + (1 - SMOOTHING) * right_fit

    prev_left_fit, prev_right_fit = left_fit, right_fit

    if left_fit is None or right_fit is None:
        return 0.0

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
    return steer

# Create folder for saved images
save_dir = "collected_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("🚗 Lane Following (Green Lane Fill) Started")

# === Main Loop ===
while robot.step(timestep) != -1:
    img = camera.getImage()
    if img is None:
        continue

    bgr = np.frombuffer(img, np.uint8).reshape((height, width, 4))[:, :, :3]
    warped = cv2.warpPerspective(bgr, M, (width, height), flags=cv2.INTER_LINEAR)
    lane_mask = detect_lanes(warped)
    left_fit, right_fit, *_ = fit_lane_lines(lane_mask)
    steer = calculate_steering(left_fit, right_fit)

    # Create lane overlay
    plot_y = np.linspace(0, height - 1, height)
    if left_fit is not None and right_fit is not None:
        left_fitx = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
        right_fitx = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]

        lane_area = np.zeros_like(warped)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, plot_y])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(lane_area, [np.int32(pts)], (0, 255, 0))

        lane_overlay = cv2.warpPerspective(lane_area, Minv, (width, height))
        result = cv2.addWeighted(bgr, 1, lane_overlay, 0.3, 0)
    else:
        result = bgr

    # Save images (optional)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cv2.imwrite(os.path.join(save_dir, f"img_{timestamp}.png"), bgr)
    cv2.imwrite(os.path.join(save_dir, f"mask_{timestamp}.png"), lane_mask)

    # Apply control
    driver.setSteeringAngle(steer)
    driver.setCruisingSpeed(BASE_SPEED)

    # Show result
    cv2.imshow("Lane Detection", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
driver.setCruisingSpeed(0)
