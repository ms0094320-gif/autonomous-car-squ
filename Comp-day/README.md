# Autonomous Car — SQU University Competition (2nd Place)

A camera-only autonomous car built for the SQU University autonomous vehicle competition. The car uses a **Distance-Transform Centerline** algorithm to follow an orange-boundary track, with a green traffic light as a start gate — all without ROS.

---

## Result

> **2nd Place** — SQU University Autonomous Car Competition

---

## Features

- **Camera-only** — no LiDAR, no ultrasonic sensors
- **Orange boundary detection** using HSV color masking
- **Distance Transform Centerline** — car stays in the center of the corridor, away from boundaries
- **Green light start gate** — car waits for a green traffic light before moving
- **Emergency stop** — halts if the path ahead is fully blocked
- **SSH-safe headless mode** — runs without a display over SSH
- **Fully tunable** via CLI arguments (speed, steering, camera, safety buffers)

---

## Hardware

| Component        | Detail                              |
|-----------------|-------------------------------------|
| **Computer**    | Raspberry Pi (with GPIO)            |
| **Camera**      | USB Camera at `/dev/video0`         |
| **Motor Driver**| BTS7960                             |
| **Steering**    | Servo motor                         |

### Wiring

| Signal         | GPIO Pin |
|---------------|----------|
| Motor Forward (RPWM) | GPIO 13 |
| Motor Backward (LPWM) | GPIO 26 |
| Servo          | GPIO 12  |

---

## Dependencies

### System packages

```bash
sudo apt update
sudo apt install python3-pip python3-opencv v4l-utils -y
```

### Python packages

```bash
pip3 install -r requirements.txt
```

---

## Usage

### Run over SSH (headless — no display needed)

```bash
python3 p4.py --headless
```

### Run with GUI (on Pi screen only)

```bash
python3 p4.py --show-gui
```

Press **`q`** to quit in GUI mode, or **`Ctrl+C`** in headless mode.

---

## CLI Arguments

### Speed

| Argument | Default | Description |
|----------|---------|-------------|
| `--base-duty` | `0.18` | Base motor duty cycle (0–1) |
| `--max-duty` | `0.26` | Maximum duty cycle |
| `--min-duty` | `0.15` | Minimum duty cycle |
| `--turn-slow` | `0.55` | Speed reduction in turns (lower = slower) |

### Steering

| Argument | Default | Description |
|----------|---------|-------------|
| `--servo-max` | `0.92` | Max steering deflection (0–1) |
| `--kp-center` | `1.55` | Center error gain |
| `--kp-heading` | `0.85` | Heading error gain |
| `--corner-boost` | `1.35` | Extra steering in sharp corners |
| `--steer-smooth` | `0.35` | Steering smoothing (lower = more responsive) |
| `--heading-alpha` | `0.55` | Lookahead heading smoothing |

### Safety / Corridor

| Argument | Default | Description |
|----------|---------|-------------|
| `--orange-dilate` | `13` | Orange boundary buffer (bigger = more clearance) |
| `--black-dilate` | `9` | Black boundary buffer |
| `--emg-block-frac` | `0.55` | Blocked fraction threshold for emergency stop |
| `--emg-frames` | `5` | Consecutive frames before emergency stop triggers |

### Camera

| Argument | Default | Description |
|----------|---------|-------------|
| `--cam-w` | `352` | Camera frame width |
| `--cam-h` | `288` | Camera frame height |
| `--cam-fps` | `30` | Camera FPS |
| `--roi-start` | `0.45` | Top of the region of interest (fraction of frame height) |

### Green Light Start Gate

| Argument | Default | Description |
|----------|---------|-------------|
| `--green-frames` | `4` | Consecutive green frames needed to start |
| `--green-min-frac` | `0.06` | Minimum green pixel fraction to count as green |
| `--green-roi` | `0.35,0.02,0.65,0.22` | Green detection ROI (normalized x0,y0,x1,y1) |

---

## How It Works

```
Camera Frame
     │
     ▼
Normalize Lighting (CLAHE + gamma correction)
     │
     ├──► Green ROI check (top-center of frame)
     │         └── Wait until green detected N consecutive frames → START
     │
     ▼
Crop ROI (lower portion of frame)
     │
     ▼
HSV Masking
  ├── Orange mask  (track boundaries)
  └── Black mask   (dark obstacles)
     │
     ▼
Dilate masks → build drivable Corridor
     │
     ▼
Distance Transform on corridor
  ├── Near band  → center error   (lateral position)
  └── Far band   → heading error  (lookahead direction)
     │
     ▼
PD-style steering command
  └── Smoothed → servo output
     │
     ▼
Speed scaled by turn sharpness + corridor clearance
     │
     ▼
Emergency stop check (center corridor fully blocked?)
```

---

## Project Structure

```
autonomous-car/
├── p4.py              # Main script
├── requirements.txt   # Python dependencies
├── LICENSE            # MIT License
└── README.md          # This file
```

---

## 👤 Author

**Mohammed Saleh Alshuraiqi**

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
