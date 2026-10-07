# Smart Security Camera System
### 
### Object detection/Warehouse Unauthorized Access Detection

The **Smart Security Camera System** is a real-time computer vision application designed to enhance warehouse security by automatically detecting and monitoring people and objects within a camera feed. The system uses **YOLOv8** for object detection and a **centroid-based tracking algorithm** to maintain object identities across video frames. It allows users to define restricted zones, configure after-hours monitoring, and specify restricted objects that should trigger security alerts. When unauthorized activity is detected, the system records the event, generates alerts, and can automatically save annotated video footage for later investigation. The project demonstrates the practical application of **computer vision, object detection, real-time video processing, object tracking, and automated security monitoring** in a real-world scenario.


## Project Structure

```
├── tracker.py          # Object identity tracking (Centroid Tracker + label preservation)
├── motion_detector.py  # Core CV pipeline (YOLOv8 detection, zone checking)
├── gui.py              # Main Tkinter dashboard (entry point)
├── zones.py            # Restricted zone management and alert logging
├── recordings/         # Auto-created — annotated video clips saved here
├── events.log          # Auto-created — file-based event log
├── security_alerts.log # Auto-created — zone violation and alert log
├── zones_config.json   # Zone configuration and alert settings
└── README.md           # This file
```

---

## Installation

```bash
pip install opencv-python pillow numpy ultralytics
```

> **Note:** `tkinter` ships with standard Python on Windows and macOS.
> On Ubuntu/Debian: `sudo apt install python3-tk`

---

## Running the Application

```bash
python gui.py
```

The dashboard will open. Use the right-hand control panel to:

| Control | Action |
|---|---|
|  Start Webcam | Opens the default camera (index 0) |
|  Load Video File | Browse for .mp4 / .avi / .mov etc. |
|  Stop | Stops capture and finalises any open clip |
| Detection Confidence Slider | Adjusts YOLOv8 detection confidence threshold |
|  Add Zone | Create a new restricted zone |
|  View Zones | Display all configured zones |
|  Alert Settings | Configure after-hours window and restricted objects |

---

## How It Works

### Detection Pipeline (per frame)

```
Raw Frame
  → Resize to 960 px wide          (speed optimisation)
  → YOLOv8 Inference               (class-aware object detection)
  → Class Mapping                  (person → HUMAN, others → OBJECT)
  → Zone Intersection Checking     (trigger alerts if in restricted zones)
  → Centroid Tracker               (maintain object IDs across frames)
  → Alert Logging                  (log zone violations and restricted objects)
  → Annotation                     (bounding boxes, labels, timestamp)
  → Zone Visualization             (draw zone boundaries on frame)
  → Display + Conditional Recording
```

### Object Classification

| Detection | Label |
|---|---|
| person (YOLO class) | **HUMAN** |
| All other classes | **OBJECT** |

### Warehouse Security Features

#### 1. **Restricted Zones**
- Define restricted areas within the warehouse (e.g., storage, equipment areas)
- Zones are visualized as colored polygons on the video feed
- Configurable alerts:
  - Alert when humans enter zone
  - Alert when objects enter zone
  - Alert only during after-hours

#### 2. **After-Hours Monitoring**
- Set a time window (e.g., 10 PM to 6 AM) for unauthorized access detection
- Trigger alerts when humans are detected during off-hours in monitored zones

#### 3. **Restricted Object Tracking**
- Define objects that should not be moved (e.g., backpack, suitcase, equipment)
- System automatically alerts when restricted objects are detected
- Default restricted objects: backpack, suitcase, handbag, laptop, briefcase, box, package

#### 4. **Real-Time Alerts**
- Instant notifications when zone violations occur
- Alert types:
  - Zone breach (unauthorized human/object in restricted area)
  - After-hours detection (human detected during off-hours)
  - Restricted object movement (unauthorized item removal)
- All alerts logged to `security_alerts.log`

#### 5. **Automatic Recording**
- Video clips recorded whenever motion is detected
- All zones highlighted and labeled on recorded footage
- Facilitates forensic review of security incidents

---

## Zone Configuration

### Creating a Zone

1. Click **"🔲 Add Zone"** in the control panel
2. Enter zone name (e.g., "Storage Area")
3. Select alert options:
   - **Alert on Human Detection** – Trigger when humans enter
   - **Alert on Object Detection** – Trigger when objects enter
   - **Alert After Hours Only** – Only alert outside specified hours

### Editing Zones

Edit `zones_config.json` directly or use the UI dialogs:

```json
{
  "zones": {
    "zone_storage": {
      "zone_id": "zone_storage",
      "name": "Storage Area",
      "coords": [[100, 150], [350, 150], [350, 400], [100, 400]],
      "enabled": true,
      "alert_on_human": true,
      "alert_on_object": true,
      "alert_after_hours": false,
      "color": [255, 0, 0]
    }
  },
  "restricted_objects": ["backpack", "suitcase", "laptop"],
  "after_hours": [22, 6]
}
```

---

## Performance

- Target: **25–30 FPS** using YOLOv8n (nano model)
- Worker thread and GUI thread decoupled via queue
- Frames dropped if GUI falls behind, preventing memory buildup
- Zone checking performs O(n) complexity per detection

---

## Multimedia Principles Demonstrated

| Principle | Where Applied |
|---|---|
| Object Detection (CNN) | YOLOv8 in `motion_detector.py` |
| Polygon Region Detection | Zone containment in `zones.py` |
| Real-time annotation | Drawing zones and objects on frames |
| Video codec selection | MJPG in `VideoRecorder.start()` |
| Annotated video saving | `VideoRecorder.write(annotated_frame)` |
| Threaded pipeline | Worker thread + queue in `gui.py` |
| Frame rate management | `root.after(33, ...)` = 30 FPS UI ceiling |
| Alert state management | Queue-based event logging |

