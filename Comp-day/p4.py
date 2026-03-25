#!/usr/bin/env python3
"""
Autonomous Car — SQU University Competition (2nd Place)
=======================================================
Camera-only autonomous car using Distance-Transform Centerline navigation.
No ROS required. Follows an orange-boundary track and starts only when a
green traffic light is detected.

Hardware:
    - BTS7960 motor driver: GPIO13 (RPWM/forward), GPIO26 (LPWM/backward)
    - Servo:                 GPIO12
    - USB camera:            /dev/video0

Usage:
    Headless / SSH:   python3 p4.py --headless
    With GUI display: python3 p4.py --show-gui

Author: Mohammed Saleh Alshuraiqi
License: MIT
"""

import os
import time
import subprocess
import signal
import atexit
import argparse

import cv2
import numpy as np

# ==============================================================================
# CLI ARGUMENTS
# ==============================================================================
ap = argparse.ArgumentParser(description="Orange-boundary autonomous track follower.")

ap.add_argument("--headless",  action="store_true", help="No GUI — safe for SSH.")
ap.add_argument("--show-gui",  action="store_true", help="Force GUI (requires a display).")
ap.add_argument("--no-v4l2",   action="store_true", help="Skip v4l2-ctl camera tweaks.")

# --- Speed (duty cycle range: 0.0 → 1.0) ---
ap.add_argument("--base-duty",  type=float, default=0.18, help="Base motor duty cycle.")
ap.add_argument("--max-duty",   type=float, default=0.26, help="Maximum motor duty cycle.")
ap.add_argument("--min-duty",   type=float, default=0.15, help="Minimum motor duty cycle.")
ap.add_argument("--turn-slow",  type=float, default=0.55,
                help="Speed reduction factor in turns (0.4 = slow, 0.8 = fast).")

# --- Steering ---
ap.add_argument("--servo-max",     type=float, default=0.92, help="Max servo deflection (0–1).")
ap.add_argument("--kp-center",     type=float, default=1.55, help="Gain for lateral center error.")
ap.add_argument("--kp-heading",    type=float, default=0.85, help="Gain for lookahead heading error.")
ap.add_argument("--corner-boost",  type=float, default=1.35, help="Steering multiplier in sharp corners.")
ap.add_argument("--steer-smooth",  type=float, default=0.35,
                help="Steering low-pass alpha (0.2=aggressive, 0.6=smooth).")
ap.add_argument("--heading-alpha", type=float, default=0.55,
                help="Heading low-pass alpha (0.4=responsive, 0.7=smooth).")

# --- Corridor safety buffers ---
ap.add_argument("--orange-dilate", type=int, default=13,
                help="Dilation kernel for orange boundary (larger = more clearance).")
ap.add_argument("--black-dilate",  type=int, default=9,
                help="Dilation kernel for black obstacles.")

# --- Camera ---
ap.add_argument("--cam-w",    type=int,   default=352)
ap.add_argument("--cam-h",    type=int,   default=288)
ap.add_argument("--cam-fps",  type=int,   default=30)
ap.add_argument("--roi-start", type=float, default=0.45,
                help="Y fraction where the track ROI begins (0=top, 1=bottom).")

# --- Emergency stop ---
ap.add_argument("--emg-block-frac", type=float, default=0.55,
                help="Blocked-corridor fraction that triggers emergency stop.")
ap.add_argument("--emg-frames",     type=int,   default=5,
                help="Consecutive blocked frames before emergency stop fires.")

# --- Debug ---
ap.add_argument("--print-rate", type=float, default=1.0,
                help="Seconds between status prints in headless mode.")

# --- Green traffic light start gate ---
ap.add_argument("--green-frames",    type=int,   default=4,
                help="Consecutive green frames needed to start the car.")
ap.add_argument("--green-min-frac",  type=float, default=0.06,
                help="Minimum green pixel fraction in ROI to count as a green light.")
ap.add_argument("--green-roi",       type=str,
                default="0.35,0.02,0.65,0.22",
                help="Green-light ROI as x0,y0,x1,y1 (normalized 0–1). Default: top-center.")

args = ap.parse_args()

# Auto-enable headless when running over SSH (unless --show-gui is forced)
if (os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY")) and not args.show_gui:
    args.headless = True
if args.show_gui:
    args.headless = False

SHOW_GUI  = not args.headless
GUI_SCALE = 1.35   # Preview window scale factor

# ==============================================================================
# CAMERA SETTINGS
# ==============================================================================
CAM_DEV = "/dev/video0"
CAM_W   = int(args.cam_w)
CAM_H   = int(args.cam_h)
CAM_FPS = int(args.cam_fps)

# Region of interest — bottom portion of the frame (track area)
ROI_Y0 = int(CAM_H * float(args.roi_start))
ROI_Y1 = CAM_H

# Two horizontal bands inside the ROI:
#   NEAR — close to the car (used for lateral centering)
#   FAR  — further ahead   (used for heading/lookahead)
BAND_NEAR_Y0 = 0.72
BAND_FAR_Y0  = 0.38

# ==============================================================================
# HSV COLOR THRESHOLDS
# ==============================================================================

# Orange track boundaries
ORANGE_H_LO = 5
ORANGE_H_HI = 28
ORANGE_S_LO = 90
ORANGE_V_LO = 70

# Black / dark obstacles
BLACK_V_HI = 65
BLACK_S_HI = 95

# Green traffic light (start gate)
GREEN_H_LO = 40
GREEN_H_HI = 90
GREEN_S_LO = 80
GREEN_V_LO = 80

# ==============================================================================
# V4L2 CAMERA TWEAKS
# ==============================================================================
V4L2_SET_SHARPNESS  = 1
V4L2_SET_CONTRAST   = 28
V4L2_AUTO_EXPOSURE  = 3   # Manual exposure mode

# ==============================================================================
# GPIO / HARDWARE INTERFACE
# ==============================================================================
from gpiozero import PWMOutputDevice, Servo
from gpiozero.pins.lgpio import LGPIOFactory

GPIO_FWD   = 13   # BTS7960 RPWM (forward)
GPIO_BWD   = 26   # BTS7960 LPWM (backward)
GPIO_SERVO = 12   # Steering servo

SERVO_CENTER = 0.00  # Servo value for straight-ahead

class CarIO:
    """Manages the motor driver and steering servo via GPIO."""

    def __init__(self):
        self.factory    = LGPIOFactory()
        self.motor_fwd  = PWMOutputDevice(GPIO_FWD,   frequency=1000, pin_factory=self.factory)
        self.motor_bwd  = PWMOutputDevice(GPIO_BWD,   frequency=1000, pin_factory=self.factory)
        self.servo      = Servo(
            GPIO_SERVO,
            min_pulse_width=0.5 / 1000,
            max_pulse_width=2.5 / 1000,
            pin_factory=self.factory,
        )
        self._steer = 0.0
        self._duty  = 0.0
        self.stop()

    def stop(self):
        """Cut motor power and center the steering."""
        self.motor_fwd.value = 0.0
        self.motor_bwd.value = 0.0
        self.set_steer(0.0)
        self._duty = 0.0

    def set_steer(self, steer_norm: float):
        """
        Set steering. steer_norm is in [-1.0, 1.0]:
            -1.0 = full left, 0.0 = straight, +1.0 = full right
        """
        steer_norm = float(np.clip(steer_norm, -1.0, 1.0))
        v = float(np.clip(SERVO_CENTER + steer_norm * args.servo_max, -1.0, 1.0))
        self.servo.value = v
        self._steer = steer_norm

    def set_forward_duty(self, duty: float):
        """Drive forward at the given duty cycle (0.0–1.0)."""
        duty = float(np.clip(duty, 0.0, 1.0))
        self.motor_bwd.value = 0.0
        self.motor_fwd.value = duty
        self._duty = duty


# Global car instance (used by signal/exit handlers)
car = None

def safe_stop(*_):
    """Emergency stop — called on exit or signal."""
    global car
    try:
        if car is not None:
            car.stop()
    except Exception:
        pass

atexit.register(safe_stop)
for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    try:
        signal.signal(sig, safe_stop)
    except Exception:
        pass

# ==============================================================================
# CAMERA UTILITIES
# ==============================================================================

def v4l2_set(camdev: str):
    """Apply v4l2-ctl settings to improve camera consistency."""
    cmds = [
        ["v4l2-ctl", "-d", camdev, "--set-ctrl", f"sharpness={V4L2_SET_SHARPNESS}"],
        ["v4l2-ctl", "-d", camdev, "--set-ctrl", f"contrast={V4L2_SET_CONTRAST}"],
        ["v4l2-ctl", "-d", camdev, "--set-ctrl", f"auto_exposure={V4L2_AUTO_EXPOSURE}"],
    ]
    for c in cmds:
        try:
            subprocess.run(c, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

def open_cam() -> cv2.VideoCapture:
    """Open the USB camera and configure resolution/FPS."""
    cap = cv2.VideoCapture(CAM_DEV, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    except Exception:
        pass
    return cap

def normalize_lighting(bgr: np.ndarray) -> np.ndarray:
    """
    Improve color consistency under varying lighting:
      1. CLAHE on the L channel (LAB) to normalize local contrast
      2. Adaptive gamma correction based on overall frame brightness
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2  = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    m    = float(np.mean(gray))

    # Choose gamma based on brightness level
    gamma = 1.0
    if   m < 85:  gamma = 0.78   # Dark — brighten
    elif m < 110: gamma = 0.90
    elif m > 175: gamma = 1.15   # Bright — darken

    if abs(gamma - 1.0) > 1e-3:
        inv   = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
        out   = cv2.LUT(out, table)

    return out

# ==============================================================================
# GREEN TRAFFIC LIGHT DETECTION (start gate)
# ==============================================================================

def parse_green_roi():
    """Parse the --green-roi argument into normalized (x0, y0, x1, y1)."""
    try:
        x0, y0, x1, y1 = [float(v.strip()) for v in str(args.green_roi).split(",")]
        x0 = float(np.clip(x0, 0.0, 1.0))
        y0 = float(np.clip(y0, 0.0, 1.0))
        x1 = float(np.clip(x1, 0.0, 1.0))
        y1 = float(np.clip(y1, 0.0, 1.0))
        if x1 <= x0: x1 = min(1.0, x0 + 0.1)
        if y1 <= y0: y1 = min(1.0, y0 + 0.1)
        return x0, y0, x1, y1
    except Exception:
        return 0.35, 0.02, 0.65, 0.22

GREEN_ROI = parse_green_roi()

def green_seen(frame_bgr: np.ndarray):
    """
    Check whether a green traffic light is visible in the top-center ROI.

    Returns:
        is_green (bool):     True if green fraction meets threshold
        green_frac (float):  Fraction of ROI pixels that are green
        box (tuple):         Pixel bounding box (x0, y0, x1, y1)
    """
    x0n, y0n, x1n, y1n = GREEN_ROI
    x0 = max(0,       min(CAM_W - 1, int(x0n * CAM_W)))
    x1 = max(1,       min(CAM_W,     int(x1n * CAM_W)))
    y0 = max(0,       min(CAM_H - 1, int(y0n * CAM_H)))
    y1 = max(1,       min(CAM_H,     int(y1n * CAM_H)))

    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return False, 0.0, (x0, y0, x1, y1)

    hsv  = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       (GREEN_H_LO, GREEN_S_LO, GREEN_V_LO),
                       (GREEN_H_HI, 255, 255))

    # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    green_frac = float(cv2.countNonZero(mask) / max(1, mask.size))
    return (green_frac >= float(args.green_min_frac)), green_frac, (x0, y0, x1, y1)

# ==============================================================================
# TRACK SEGMENTATION
# ==============================================================================

def masks_track(frame_bgr: np.ndarray):
    """
    Segment the track ROI into orange boundaries and black obstacles.

    Returns:
        roi_bgr:     Cropped ROI image
        orange_mask: Binary mask of orange pixels
        black_mask:  Binary mask of dark/black pixels
    """
    roi = frame_bgr[ROI_Y0:ROI_Y1, 0:CAM_W]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Orange boundary mask
    orange = cv2.inRange(hsv,
                         (ORANGE_H_LO, ORANGE_S_LO, ORANGE_V_LO),
                         (ORANGE_H_HI, 255, 255))

    # Black / dark obstacle mask (low value + low saturation)
    h, s, v = cv2.split(hsv)
    black_v = cv2.inRange(v, 0, BLACK_V_HI)
    black_s = cv2.inRange(s, 0, BLACK_S_HI)
    black   = cv2.bitwise_and(black_v, black_s)

    k5 = np.ones((5, 5), np.uint8)
    k3 = np.ones((3, 3), np.uint8)

    orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, k5)
    orange = cv2.morphologyEx(orange, cv2.MORPH_OPEN,  k5)
    black  = cv2.morphologyEx(black,  cv2.MORPH_OPEN,  k3)
    black  = cv2.morphologyEx(black,  cv2.MORPH_CLOSE, k3)

    return roi, orange, black

def build_corridor(orange_mask: np.ndarray, black_mask: np.ndarray):
    """
    Build the drivable corridor by dilating boundary masks and inverting.

    The dilation creates a safety buffer around each boundary, so the
    distance-transform centerline stays well away from the edges.

    Returns:
        corridor:  Binary mask — white = drivable, black = off-limits
        non_drive: Combined dilated obstacle mask
    """
    ok = max(3, int(args.orange_dilate)) | 1   # ensure odd
    bk = max(3, int(args.black_dilate))  | 1

    orange_thick = cv2.dilate(orange_mask, np.ones((ok, ok), np.uint8), iterations=1)
    black_thick  = cv2.dilate(black_mask,  np.ones((bk, bk), np.uint8), iterations=1)

    non_drive = cv2.bitwise_or(orange_thick, black_thick)
    corridor  = cv2.bitwise_not(non_drive)

    # Clean up small islands in the corridor
    corridor = cv2.morphologyEx(corridor, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
    corridor = cv2.morphologyEx(corridor, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    return corridor, non_drive

# ==============================================================================
# DISTANCE TRANSFORM CENTERLINE
# ==============================================================================

def dt_center_and_heading(corridor: np.ndarray):
    """
    Use the Distance Transform to find the safest path through the corridor.

    The DT assigns each free pixel a value equal to its distance from the
    nearest obstacle. The column with the highest DT value in each band is
    the center of the widest gap — i.e., the optimal path.

    Returns:
        center_err  (float): Lateral error in [-1, 1]. Positive = car is left of center.
        heading_err (float): Lookahead error in [-1, 1].
        conf        (float): Confidence 0–1 based on corridor clearance.
        dbg         (tuple): (x_near, x_far, d_near, d_far) for visualisation.
    """
    h, w = corridor.shape[:2]

    free = corridor.copy()
    free[free > 0] = 255
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)

    y_far0  = int(h * BAND_FAR_Y0)
    y_near0 = int(h * BAND_NEAR_Y0)

    far  = dist[y_far0:h,  :]
    near = dist[y_near0:h, :]

    near_col = np.max(near, axis=0)
    far_col  = np.max(far,  axis=0)

    x_near = int(np.argmax(near_col))
    x_far  = int(np.argmax(far_col))
    d_near = float(np.max(near_col))
    d_far  = float(np.max(far_col))

    # Confidence scales with corridor clearance (sum of near + far distances)
    conf = float(np.clip((d_near + d_far) / 50.0, 0.0, 1.0))

    # Normalise errors to [-1, 1] relative to frame centre
    center_err  = float(np.clip((x_near - w / 2.0) / (w / 2.0), -1.0, 1.0))
    heading_err = float(np.clip((x_far  - w / 2.0) / (w / 2.0), -1.0, 1.0))

    return center_err, heading_err, conf, (x_near, x_far, d_near, d_far)

def emergency_blocked(corridor: np.ndarray) -> float:
    """
    Check how much of the centre-forward corridor is blocked.

    Returns the blocked fraction (0.0 = clear, 1.0 = fully blocked).
    Emergency stop fires when this exceeds --emg-block-frac for N frames.
    """
    h, w    = corridor.shape[:2]
    y_near0 = int(h * BAND_NEAR_Y0)
    cx0     = int(w * 0.44)
    cx1     = int(w * 0.56)

    centre_slice = corridor[y_near0:h, cx0:cx1]
    free_frac    = float(cv2.countNonZero(centre_slice) / max(1, centre_slice.size))
    return 1.0 - free_frac

# ==============================================================================
# MAIN LOOP
# ==============================================================================

def main():
    global car

    if not args.no_v4l2:
        v4l2_set(CAM_DEV)

    car = CarIO()
    cap = open_cam()

    if not cap.isOpened():
        print("❌ Camera not opened. Check /dev/video0 and video group permissions.")
        return

    # Steering and heading low-pass filter states
    steer_f = 0.0
    head_f  = 0.0

    # Sliding window for emergency stop votes
    emg_hist = [0] * max(1, int(args.emg_frames))
    emg_i    = 0

    last_print = time.time()

    # Start-gate state
    started     = False
    green_count = 0
    need_green  = max(1, int(args.green_frames))

    print("✅ ORANGE-BOUNDARY DT centerline started")
    print("✅ Start gate: WAIT for GREEN, then RUN forever")
    print(f"   Headless: {'YES' if args.headless else 'NO'}")
    if SHOW_GUI:
        print("   Press 'q' to quit")
    else:
        print("   Ctrl+C to quit")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            car.stop()
            time.sleep(0.05)
            continue

        frame   = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
        frame_n = normalize_lighting(frame)

        # ---- START GATE: wait for green light --------------------------------
        if not started:
            is_g, gfrac, gbox = green_seen(frame_n)

            if is_g:
                green_count += 1
            else:
                green_count = 0

            car.stop()
            duty  = 0.0
            state = f"WAIT_GREEN {green_count}/{need_green} g={gfrac:.2f}"

            if green_count >= need_green:
                started     = True
                green_count = 0
                print("🟢 GREEN seen → STARTING RUN")

            # Status print (headless)
            if not SHOW_GUI:
                now = time.time()
                if now - last_print >= float(args.print_rate):
                    last_print = now
                    print(f"[{state}]")

            # Status overlay (GUI)
            if SHOW_GUI:
                vis = frame_n.copy()
                x0, y0, x1, y1 = gbox
                colour = (0, 255, 0) if is_g else (255, 255, 255)
                cv2.rectangle(vis, (x0, y0), (x1, y1), colour, 2)
                cv2.putText(vis, state,
                            (6, CAM_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 2)
                vis2 = cv2.resize(vis, (int(CAM_W * GUI_SCALE), int(CAM_H * GUI_SCALE)))
                cv2.imshow("ORANGE TRACK (DT centerline)", vis2)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            else:
                time.sleep(0.001)

            continue  # Skip track logic until start gate clears

        # ---- TRACK FOLLOWING ------------------------------------------------

        roi_bgr, orange_mask, black_mask = masks_track(frame_n)
        corridor, non_drive              = build_corridor(orange_mask, black_mask)

        center_err, heading_err, conf, dbg = dt_center_and_heading(corridor)
        x_near, x_far, d_near, d_far       = dbg

        # Low-pass filter the heading error for smoother lookahead
        head_f = args.heading_alpha * head_f + (1.0 - args.heading_alpha) * heading_err

        # PD-style steering: lateral error + filtered heading error
        steer_cmd = args.kp_center * center_err + args.kp_heading * head_f

        # Extra steering authority in sharp corners
        if abs(head_f) > 0.30:
            steer_cmd *= args.corner_boost

        # Reduce steering when corridor detection is uncertain
        if conf < 0.25:
            steer_cmd *= 0.35

        steer_cmd = float(np.clip(steer_cmd, -1.0, 1.0))

        # Low-pass filter the final steering command
        steer_f = args.steer_smooth * steer_f + (1.0 - args.steer_smooth) * steer_cmd
        steer_f = float(np.clip(steer_f, -1.0, 1.0))

        # ---- Emergency stop check -------------------------------------------
        blocked_frac      = emergency_blocked(corridor)
        emg               = 1 if blocked_frac > float(args.emg_block_frac) else 0
        emg_hist[emg_i]   = emg
        emg_i             = (emg_i + 1) % len(emg_hist)
        emg_on            = (sum(emg_hist) >= len(emg_hist))

        if emg_on:
            car.stop()
            duty  = 0.0
            state = "EMG_STOP"
        else:
            car.set_steer(steer_f)

            # Slow down proportionally to steering angle
            turn_scale = 1.0 - (1.0 - float(args.turn_slow)) * min(1.0, abs(steer_f))
            duty       = float(args.base_duty) * turn_scale

            # Extra slow-down when corridor is very narrow (near obstacle)
            if d_near < 10.0:
                duty *= 0.75
            if d_near < 7.0:
                duty *= 0.65

            duty = float(np.clip(duty, float(args.min_duty), float(args.max_duty)))
            car.set_forward_duty(duty)
            state = "RUN"

        # ---- Console output (headless) --------------------------------------
        if not SHOW_GUI:
            now = time.time()
            if now - last_print >= float(args.print_rate):
                last_print = now
                print(f"[{state}] duty={duty:.2f} steer={steer_f:+.2f} conf={conf:.2f} "
                      f"d_near={d_near:.1f} blocked={blocked_frac:.2f} "
                      f"xN={x_near} xF={x_far}")

        # ---- Visual overlay (GUI) -------------------------------------------
        if SHOW_GUI:
            vis = frame_n.copy()

            # ROI boundary line
            cv2.line(vis, (0, ROI_Y0), (CAM_W, ROI_Y0), (255, 255, 255), 1)

            # Near/far DT centreline points
            y_near_vis = ROI_Y0 + int((ROI_Y1 - ROI_Y0) * BAND_NEAR_Y0)
            y_far_vis  = ROI_Y0 + int((ROI_Y1 - ROI_Y0) * BAND_FAR_Y0)
            cv2.circle(vis, (x_near, y_near_vis), 6, (0, 255, 255), -1)   # cyan = near
            cv2.circle(vis, (x_far,  y_far_vis),  6, (255, 255, 0), -1)   # yellow = far

            cv2.putText(
                vis,
                f"{state} duty:{duty:.2f} steer:{steer_f:+.2f} "
                f"conf:{conf:.2f} dNear:{d_near:.1f} blk:{blocked_frac:.2f}",
                (6, CAM_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2,
            )

            vis2 = cv2.resize(vis, (int(CAM_W * GUI_SCALE), int(CAM_H * GUI_SCALE)))
            cv2.imshow("ORANGE TRACK (DT centerline)", vis2)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        else:
            time.sleep(0.001)

    # ---- Clean up -----------------------------------------------------------
    safe_stop()
    cap.release()
    if SHOW_GUI:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        safe_stop()
