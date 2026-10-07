"""
motion_detector.py — Core CV Pipeline for Smart Security Camera System
CSY3058 Media Technology | Assignment 1

Multimedia Principles Applied:
    • Frame resizing        – Scales input frames to a fixed width (960 px)
      before inference, reducing YOLO input resolution and cutting processing
      time proportionally while preserving aspect ratio.
    • YOLOv8 inference      – Convolutional Neural Network (CNN) that performs
      single-pass object detection: a feature pyramid network extracts
      multi-scale spatial features; a detection head predicts class, bounding
      box, and confidence score for each anchor region simultaneously.
    • Class mapping         – YOLO's 80-class COCO output is collapsed to a
      two-class schema (HUMAN / OBJECT) appropriate for warehouse security,
      reducing downstream complexity without retraining the model.
    • Alert deduplication   – Per-(zone, label) time-window cooldown prevents
      repeated alerts for the same continuously-present detection, avoiding
      log spam while still catching re-entries after a genuine departure.
    • MJPG codec recording  – Intra-frame (I-frame only) compression ensures
      every recorded frame is independently decodable — essential for forensic
      review where arbitrary seek points are required.
"""

import time
import cv2
import numpy as np
from datetime import datetime
from zones import ZoneManager, AlertLogger, AlertType


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Frame resize width (px). Smaller frames = faster processing = higher FPS.
FRAME_WIDTH = 960

# Default YOLO confidence threshold.
CONFIDENCE_THRESHOLD_DEFAULT = 0.40

# Minimum seconds between repeated alerts for the same (zone, label) pair.
# Prevents log spam when a person/object remains in a zone continuously.
ALERT_COOLDOWN_SECONDS = 5.0


# ──────────────────────────────────────────────────────────────────────────────
# MotionDetector
# ──────────────────────────────────────────────────────────────────────────────

class MotionDetector:
    """
    Encapsulates the full per-frame detection pipeline using YOLOv8.

    The pipeline follows an object detection architecture:
        raw frame
            → resize              (speed optimisation)
            → YOLOv8 inference   (class-aware detection)
            → bbox conversion    (corner coordinates → x, y, w, h)
            → label mapping      (HUMAN / OBJECT)
            → zone checking      (trigger alerts if in restricted zones)
            → alert dedup        (suppress repeats within cooldown window)

    Attributes:
        confidence (float): YOLO confidence threshold used to filter detections.
    """

    def __init__(
        self,
        confidence: float = CONFIDENCE_THRESHOLD_DEFAULT,
        zone_manager: ZoneManager | None = None,
        alert_logger: AlertLogger | None = None,
    ):
        self._confidence = float(confidence)
        self._zone_manager = zone_manager or ZoneManager()
        self._alert_logger = alert_logger or AlertLogger()

        # Maps (zone_id, alert_type_value, label) → timestamp of last alert.
        # Uses stable identifiers (not pixel coords) so the same object
        # staying in the same zone is correctly deduplicated across frames.
        self._last_alert_times: dict[tuple, float] = {}

        try:
            from ultralytics import YOLO
            self._model = YOLO("yolov8n.pt")
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize YOLOv8 model. Ensure ultralytics is installed."
            ) from exc

    # ── Properties ────────────────────────────────────────────────────

    @property
    def confidence(self) -> float:
        return self._confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        self._confidence = float(value)

    # ── Core processing ───────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> tuple:
        """
        Run the YOLOv8 detection pipeline on a single raw video frame.

        Args:
            frame: BGR image array from cv2.VideoCapture.read().

        Returns:
            display_frame: Resized BGR frame for annotation / GUI.
            detections:    List of ((x, y, w, h), label) tuples.
            alerts:        List of (AlertType, Zone) tuples for newly
                           triggered alerts (deduplicated by cooldown).
        """
        # ── 1. Resize frame ───────────────────────────────────────────
        h, w = frame.shape[:2]
        scale = FRAME_WIDTH / w
        new_h = int(h * scale)
        display_frame = cv2.resize(frame, (FRAME_WIDTH, new_h))

        # ── 2. YOLOv8 inference ───────────────────────────────────────
        results = self._model(display_frame, conf=self._confidence, verbose=False)
        detections = []
        alerts = []
        now = time.monotonic()

        if len(results) > 0:
            boxes = results[0].boxes
            class_names = self._model.names

            for xyxy, cls, conf in zip(
                boxes.xyxy.tolist(),
                boxes.cls.tolist(),
                boxes.conf.tolist(),
            ):
                # ── 3. Convert bbox and map label ─────────────────────
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                x, y, w_box, h_box = x1, y1, x2 - x1, y2 - y1
                class_name = class_names.get(int(cls), "unknown").lower()
                vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}
                if class_name == "person":
                    label = "HUMAN"
                elif class_name in vehicle_classes:
                    label = "VEHICLE"
                else:
                    label = "OBJECT"
                detections.append(((x, y, w_box, h_box), label))

                # ── 4. Zone checking ──────────────────────────────────
                zone_alerts = self._zone_manager.check_detection(
                    (x, y, w_box, h_box), label, class_name
                )

                for alert_type, zone in zone_alerts:
                    # ── 5. Alert deduplication ────────────────────────
                    # Key is (zone_id, alert_type, label) — stable across
                    # frames regardless of pixel-level bbox jitter.
                    # This correctly suppresses repeated alerts while the
                    # same person/object stays in a zone, but will re-alert
                    # after ALERT_COOLDOWN_SECONDS (e.g. a fresh entry).
                    alert_key = (zone.zone_id, alert_type.value, label)
                    last_time = self._last_alert_times.get(alert_key, 0.0)

                    if now - last_time >= ALERT_COOLDOWN_SECONDS:
                        alerts.append((alert_type, zone))
                        self._alert_logger.log_alert(
                            alert_type, zone, label, class_name, float(conf)
                        )
                        self._last_alert_times[alert_key] = now

        # ── 6. Prune stale cooldown entries ───────────────────────────
        # Remove entries older than 2× the cooldown window so the dict
        # doesn't grow unboundedly during long sessions. Safe to do here
        # because any recently-alerted key will have been updated above.
        if len(self._last_alert_times) > 200:
            cutoff = now - (ALERT_COOLDOWN_SECONDS * 2)
            self._last_alert_times = {
                k: v for k, v in self._last_alert_times.items() if v >= cutoff
            }

        return display_frame, detections, alerts

    # ── Annotation helpers ─────────────────────────────────────────────

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        tracked_objects: dict,
        recording: bool = False,
    ) -> np.ndarray:
        """
        Draw bounding boxes, labels, object IDs, and a timestamp onto
        a copy of the frame. Returns the annotated copy.

        Args:
            frame:           BGR image to annotate.
            tracked_objects: OrderedDict from CentroidTracker.update().
                             Maps oid → (centroid, bbox, label).
            recording:       If True, draws a red REC indicator.

        Returns:
            Annotated BGR image (new array, original untouched).
        """
        annotated = frame.copy()

        colours = {
            "HUMAN":  (0, 200, 0),
            "OBJECT": (235, 180, 35),
            "VEHICLE": (0, 150, 255),
        }

        for oid, (centroid, bbox, label) in tracked_objects.items():
            x, y, w, h = bbox
            colour = colours.get(label, (0, 200, 0))

            # Bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)

            # Label tag background + text
            tag = f"{label} #{oid}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(
                annotated,
                (x, y - th - 8),
                (x + tw + 4, y),
                colour,
                -1,
            )
            cv2.putText(
                annotated,
                tag,
                (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            # Centroid dot
            cx, cy = centroid
            cv2.circle(annotated, (cx, cy), 4, colour, -1)

        # Timestamp overlay
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(
            annotated,
            ts,
            (10, annotated.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # REC indicator
        if recording:
            fw = annotated.shape[1]
            cv2.circle(annotated, (fw - 20, 20), 8, (0, 0, 220), -1)
            cv2.putText(
                annotated,
                "REC",
                (fw - 50, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 220),
                2,
                cv2.LINE_AA,
            )

        return annotated


# ──────────────────────────────────────────────────────────────────────────────
# VideoRecorder
# ──────────────────────────────────────────────────────────────────────────────

class VideoRecorder:
    """
    Manages writing annotated frames to disk using the MJPG codec.

    Codec choice — MJPG (Motion JPEG):
        • Each frame is individually JPEG-compressed (intra-frame codec).
        • No inter-frame dependencies, so any frame can be a random-access
          point — ideal for forensic review of security footage.
        • Widely supported without additional codecs on Windows/Linux/macOS.
        • Unlike H.264, it does not require a constant-bitrate GOP structure,
          simplifying variable-duration clip recording.

    Attributes:
        writer    (cv2.VideoWriter | None): Active writer, or None if idle.
        clip_path (str | None):            Path of the file being written.
    """

    def __init__(self):
        self.writer: cv2.VideoWriter | None = None
        self.clip_path: str | None = None
        self.frame_size: tuple | None = None

    def start(self, path: str, frame_size: tuple, fps: float = 20.0) -> None:
        """
        Open a new output file and begin recording.

        Args:
            path:       Destination file path (e.g. "clip_20260515_143012.avi").
            frame_size: (width, height) of frames that will be written.
            fps:        Playback frame rate for the output file.
        """
        width, height = frame_size
        if width % 2 != 0 or height % 2 != 0:
            width -= width % 2
            height -= height % 2
            frame_size = (width, height)
            print(f"VideoRecorder: adjusted frame_size to even dimensions {frame_size} for FFmpeg.")

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        if not self.writer or not self.writer.isOpened():
            print(f"VideoRecorder: failed to open writer for {path} with size {frame_size}.")
            self.writer = None
            self.clip_path = None
            self.frame_size = None
            return

        self.clip_path = path
        self.frame_size = frame_size

    def write(self, frame: np.ndarray) -> None:
        """Write a single annotated frame to the open clip."""
        if not self.writer or not self.writer.isOpened():
            return

        if self.frame_size is not None:
            expected_w, expected_h = self.frame_size
            height, width = frame.shape[:2]
            if (width, height) != (expected_w, expected_h):
                frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)
                print(
                    f"VideoRecorder: resized frame from {(width, height)} to "
                    f"{self.frame_size} before writing."
                )

        success = self.writer.write(frame)
        if success is False:
            print(f"VideoRecorder: failed to write frame to {self.clip_path}.")

    def stop(self) -> str | None:
        """
        Flush and close the current clip.

        Returns:
            Path of the saved file, or None if nothing was open.
        """
        if self.writer:
            self.writer.release()
            self.writer = None
            path = self.clip_path
            self.clip_path = None
            return path
        return None

    @property
    def is_recording(self) -> bool:
        return self.writer is not None and self.writer.isOpened()