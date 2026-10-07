"""
gui.py — Main Application: Smart Security Camera System Dashboard
CSY3058 Media Technology | Assignment 1

Architecture:
    • The GUI runs on the main thread (Tkinter requirement).
    • Video capture and frame processing run in a separate daemon thread
      to prevent the UI from blocking/freezing during heavy CV operations.
    • Processed frames are passed to the GUI via a thread-safe queue.
    • The GUI polls the queue at ~33 ms intervals (≈30 FPS ceiling) and
      renders the latest frame using Pillow's ImageTk bridge.

Multimedia Principle — Frame Display:
    OpenCV stores images as BGR numpy arrays.  Tkinter's PhotoImage accepts
    only RGB.  Pillow (PIL) acts as the conversion layer:
        cv2 BGR array → PIL Image (RGB) → ImageTk.PhotoImage → Tkinter Canvas
"""

import cv2
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import queue
import logging
import os
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np

from motion_detector import MotionDetector, VideoRecorder
from tracker import CentroidTracker
from zones import ZoneManager, AlertLogger, AlertType


# ──────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    filename="events.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SecurityCamera")

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (dark professional theme)
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg":           "#0d1117",   # near-black page background
    "panel":        "#161b22",   # slightly lighter panel background
    "border":       "#30363d",   # subtle border / divider
    "accent":       "#238636",   # green accent (GitHub-inspired)
    "accent_hover": "#2ea043",
    "danger":       "#da3633",   # red for stop / rec indicator
    "warning":      "#d29922",   # amber for caution states
    "text":         "#e6edf3",   # primary text
    "text_muted":   "#8b949e",   # secondary / label text
    "button":       "#21262d",   # default button bg
    "button_hover": "#30363d",
    "log_bg":       "#0d1117",
    "log_text":     "#7ee787",   # green terminal-style log text
    "canvas_bg":    "#010409",   # video canvas background
}

# ──────────────────────────────────────────────────────────────────────────────
# Recording directory
# ──────────────────────────────────────────────────────────────────────────────

RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helper: styled button factory
# ──────────────────────────────────────────────────────────────────────────────

def make_button(parent, text, command, colour=None, width=18):
    bg = colour or PALETTE["button"]
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=PALETTE["text"],
        activebackground=PALETTE["button_hover"],
        activeforeground=PALETTE["text"],
        relief="flat",
        bd=0,
        padx=10,
        pady=8,
        font=("Segoe UI", 10, "bold"),
        width=width,
        cursor="hand2",
    )
    return btn


# ──────────────────────────────────────────────────────────────────────────────
# SecurityCameraApp
# ──────────────────────────────────────────────────────────────────────────────

class SecurityCameraApp:
    """
    Main application class.  Owns the Tkinter root window, the CV worker
    thread, and all state related to capture, detection, tracking, and
    recording.

    Threading model:
        GUI thread  → Tkinter mainloop, canvas updates, slider callbacks
        Worker thread → cap.read() → MotionDetector.process() → annotate
                        → push (display_frame, metadata) onto self._queue
        GUI thread polls self._queue via root.after() to refresh canvas.
    """

    # ── Construction ──────────────────────────────────────────────────

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Smart Security Camera System  |  CSY3058")
        self.root.configure(bg=PALETTE["bg"])
        self.root.resizable(True, True)
        self.root.minsize(1100, 680)

        # CV components
        self._zone_manager = ZoneManager()
        self._alert_logger = AlertLogger()
        self._detector = MotionDetector(zone_manager=self._zone_manager, 
                                        alert_logger=self._alert_logger)
        self._tracker = CentroidTracker(max_disappeared=40, max_distance=80)
        self._recorder = VideoRecorder()

        # Capture state
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._worker: threading.Thread | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=2)

        # Zone drawing state
        self._drawing_zone = False
        self._zone_points = []
        self._temp_zone_preview = None

        # Stats
        self._frame_count = 0
        self._fps_display = 0.0
        self._fps_timer = datetime.now()
        self._events_count = 0

        # Post-motion cooldown: keep recording N frames after motion stops
        self._motion_cooldown = 0
        self._COOLDOWN_FRAMES = 60   # ≈ 2-3 seconds at 25 FPS

        # Fix 5 — Periodic event re-logging: log an update every 10 s while
        # motion is continuously active (not just at the start of a new clip).
        self._last_log_time = datetime.now()

        # Build UI
        self._build_ui()

        # Start GUI refresh loop
        self._refresh_canvas()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────────────

    def _build_ui(self):
        """Construct the full dashboard layout."""
        # ── Top title bar ─────────────────────────────────────────────
        title_bar = tk.Frame(self.root, bg=PALETTE["panel"], height=48)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)

        tk.Label(
            title_bar,
            text="⚠  SMART SECURITY CAMERA SYSTEM",
            bg=PALETTE["panel"],
            fg=PALETTE["accent"],
            font=("Segoe UI", 13, "bold"),
        ).pack(side="left", padx=16, pady=10)

        self._status_var = tk.StringVar(value="● IDLE")
        self._status_label = tk.Label(
            title_bar,
            textvariable=self._status_var,
            bg=PALETTE["panel"],
            fg=PALETTE["text_muted"],
            font=("Segoe UI", 10),
        )
        self._status_label.pack(side="right", padx=16)

        self._alert_var = tk.StringVar(value="")
        self._alert_label = tk.Label(
            title_bar,
            textvariable=self._alert_var,
            bg=PALETTE["panel"],
            fg=PALETTE["danger"],
            font=("Segoe UI", 10, "bold"),
        )
        self._alert_label.pack(side="right", padx=8)

        self._rec_var = tk.StringVar(value="")
        tk.Label(
            title_bar,
            textvariable=self._rec_var,
            bg=PALETTE["panel"],
            fg=PALETTE["danger"],
            font=("Segoe UI", 10, "bold"),
        ).pack(side="right", padx=8)

        # ── Main body ─────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=PALETTE["bg"])
        body.pack(fill="both", expand=True, padx=10, pady=10)

        # Left: video canvas
        left = tk.Frame(body, bg=PALETTE["bg"])
        left.pack(side="left", fill="both", expand=True)

        self._canvas = tk.Canvas(
            left,
            bg=PALETTE["canvas_bg"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
        )
        self._canvas.pack(fill="both", expand=True)
        self._canvas_image_id = None
        self._tk_image = None  # must hold reference to prevent GC

        # Placeholder text when idle
        self._canvas.create_text(
            480, 320,
            text="No video source active.\nUse the controls on the right to begin.",
            fill=PALETTE["text_muted"],
            font=("Segoe UI", 13),
            justify="center",
            tags="placeholder",
        )

        # Right: control panel
        right = tk.Frame(body, bg=PALETTE["panel"], width=280)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        self._build_control_panel(right)

    def _build_control_panel(self, parent):
        """Build all controls in the right-side panel."""
        pad = {"padx": 14, "pady": 6}

        # ── Section: Source ───────────────────────────────────────────
        self._section_label(parent, "VIDEO SOURCE")

        self._btn_webcam = make_button(parent, "▶  Start Webcam", self._start_webcam,
                                       colour=PALETTE["accent"])
        self._btn_webcam.pack(fill="x", **pad)

        self._btn_file = make_button(parent, "📂  Load Video File", self._load_file)
        self._btn_file.pack(fill="x", **pad)

        self._btn_stop = make_button(parent, "■  Stop", self._stop,
                                     colour=PALETTE["danger"])
        self._btn_stop.pack(fill="x", **pad)
        self._btn_stop.config(state="disabled")

        self._sep(parent)

        # ── Section: Detection Confidence ────────────────────────────────
        self._section_label(parent, "DETECTION CONFIDENCE")

        sens_frame = tk.Frame(parent, bg=PALETTE["panel"])
        sens_frame.pack(fill="x", **pad)

        tk.Label(
            sens_frame, text="Low confidence", bg=PALETTE["panel"],
            fg=PALETTE["text_muted"], font=("Segoe UI", 8)
        ).pack(side="left")
        tk.Label(
            sens_frame, text="High confidence", bg=PALETTE["panel"],
            fg=PALETTE["text_muted"], font=("Segoe UI", 8)
        ).pack(side="right")

        self._conf_var = tk.IntVar(value=40)
        self._conf_slider = ttk.Scale(
            parent,
            from_=20,
            to=80,
            orient="horizontal",
            variable=self._conf_var,
            command=self._on_confidence_change,
        )
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Horizontal.TScale",
            background=PALETTE["panel"],
            troughcolor=PALETTE["border"],
            sliderthickness=18,
        )
        self._conf_slider.pack(fill="x", padx=14, pady=2)

        self._conf_val_label = tk.Label(
            parent,
            text="Confidence: 0.40  (medium)",
            bg=PALETTE["panel"],
            fg=PALETTE["text_muted"],
            font=("Segoe UI", 8),
        )
        self._conf_val_label.pack(padx=14, anchor="w")

        self._sep(parent)

        # ── Section: Stats ────────────────────────────────────────────
        self._section_label(parent, "LIVE STATISTICS")

        stats_frame = tk.Frame(parent, bg=PALETTE["panel"])
        stats_frame.pack(fill="x", padx=14, pady=4)

        self._fps_var = tk.StringVar(value="FPS: —")
        self._events_var = tk.StringVar(value="Events: 0")
        self._counts_var = tk.StringVar(value="Humans: 0  Objects: 0  Vehicles: 0")

        for var in (self._fps_var, self._events_var, self._counts_var):
            tk.Label(
                stats_frame,
                textvariable=var,
                bg=PALETTE["panel"],
                fg=PALETTE["text"],
                font=("Segoe UI", 9),
                anchor="w",
            ).pack(fill="x", pady=1)

        self._sep(parent)

        # ── Section: Zone Management ──────────────────────────────────
        self._section_label(parent, "RESTRICTED ZONES")

        self._btn_add_zone = make_button(parent, "🔲  Add Zone", self._add_zone_dialog)
        self._btn_add_zone.pack(fill="x", **pad)

        self._btn_view_zones = make_button(parent, "👁  View Zones", self._show_zones_dialog)
        self._btn_view_zones.pack(fill="x", **pad)

        self._btn_alert_config = make_button(parent, "⚙  Alert Settings", self._show_alert_config)
        self._btn_alert_config.pack(fill="x", **pad)

        self._sep(parent)

        # ── Section: Event Log ────────────────────────────────────────
        self._section_label(parent, "EVENT LOG")

        self._log_box = scrolledtext.ScrolledText(
            parent,
            bg=PALETTE["log_bg"],
            fg=PALETTE["log_text"],
            font=("Courier New", 8),
            relief="flat",
            bd=0,
            wrap="word",
            state="disabled",
            height=10,
        )
        self._log_box.pack(fill="both", expand=True, padx=14, pady=(4, 10))

    # ── Layout helpers ─────────────────────────────────────────────────

    def _section_label(self, parent, text):
        tk.Label(
            parent,
            text=text,
            bg=PALETTE["panel"],
            fg=PALETTE["text_muted"],
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", padx=14, pady=(10, 2))

    def _sep(self, parent):
        tk.Frame(parent, bg=PALETTE["border"], height=1).pack(
            fill="x", padx=14, pady=4
        )

    # ── Callbacks: Source controls ────────────────────────────────────

    def _start_webcam(self):
        """Open the default webcam (index 0) and begin processing."""
        self._start_capture(0)

    def _load_file(self):
        """Open a file dialog and begin processing the selected video."""
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._start_capture(path)

    def _start_capture(self, source):
        """Common entry point to open a capture source and start the worker."""
        if self._running:
            self._stop()

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self._log_event("ERROR: Could not open video source.")
            return

        self._cap = cap
        self._running = True
        self._frame_count = 0
        self._fps_timer = datetime.now()

        # Reset tracker and detector for fresh session
        self._tracker = CentroidTracker(max_disappeared=40, max_distance=80)
        self._detector = MotionDetector(
            confidence=self._conf_var.get() / 100.0,
            zone_manager=self._zone_manager,
            alert_logger=self._alert_logger
        )
        self._last_log_time = datetime.now()  # reset periodic log timer

        # Update UI state
        self._btn_webcam.config(state="disabled")
        self._btn_file.config(state="disabled")
        self._btn_stop.config(state="normal")
        src_name = "Webcam" if source == 0 else os.path.basename(str(source))
        self._set_status(f"● LIVE — {src_name}", PALETTE["accent"])
        self._log_event(f"Session started: {src_name}")
        logger.info("Session started: %s", src_name)

        # Remove placeholder text
        self._canvas.delete("placeholder")

        # Start worker thread
        self._worker = threading.Thread(target=self._capture_loop, daemon=True)
        self._worker.start()

    def _stop(self):
        """Signal the worker thread to stop and clean up resources."""
        self._running = False
        if self._recorder.is_recording:
            clip = self._recorder.stop()
            if clip:
                self._log_event(f"Clip saved: {os.path.basename(clip)}")
                logger.info("Clip saved: %s", clip)
        if self._cap:
            self._cap.release()
            self._cap = None

        self._btn_webcam.config(state="normal")
        self._btn_file.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._set_status("● IDLE", PALETTE["text_muted"])
        self._rec_var.set("")
        self._log_event("Session stopped.")
        logger.info("Session stopped.")

    # ── Callback: Confidence slider ──────────────────────────────────

    def _on_confidence_change(self, value):
        """
        Called continuously as the slider moves.
        Maps slider value to the YOLO confidence threshold.
        Higher slider position means stricter confidence filtering.
        """
        percent = int(float(value))
        confidence = percent / 100.0
        self._detector.confidence = confidence
        if percent < 30:
            desc = "low"
        elif percent < 50:
            desc = "medium"
        elif percent < 70:
            desc = "high"
        else:
            desc = "very high"
        self._conf_val_label.config(text=f"Confidence: {confidence:.2f}  ({desc})")

    # ── Worker thread: capture loop ───────────────────────────────────

    def _capture_loop(self):
        """
        Runs in a background daemon thread.
        Reads frames from the capture source, runs the full CV pipeline,
        and pushes results to the GUI queue.

        Performance:
            • The queue is size-limited to 2 entries.  If the GUI thread
              hasn't consumed the previous frame yet, we drop the oldest
              (non-blocking put_nowait) to prevent queue back-pressure
              causing unbounded memory growth.
        """
        while self._running and self._cap and self._cap.isOpened():
            ret, raw_frame = self._cap.read()
            if not ret:
                # End of file or camera disconnected
                self.root.after(0, self._stop)
                break

            # Run CV pipeline
            display_frame, detections, alerts = self._detector.process(raw_frame)

            # Log and handle alerts
            for alert_type, zone in alerts:
                alert_msg = f"🚨 ALERT [{alert_type.value}] in zone '{zone.name}'"
                self.root.after(0, self._log_event, alert_msg)
                self.root.after(0, self._set_alert, alert_msg)
                logger.warning(alert_msg)

            # Update tracker — track YOLO detections with their labels.
            fh, fw = display_frame.shape[:2]
            tracked = self._tracker.update(detections, frame_size=(fw, fh))

            human_count = sum(1 for _, _, label in tracked.values() if label == "HUMAN")
            vehicle_count = sum(1 for _, _, label in tracked.values() if label == "VEHICLE")
            object_count = sum(1 for _, _, label in tracked.values() if label == "OBJECT")
            motion_detected = (human_count + vehicle_count + object_count) > 0

            # Manage recording
            if motion_detected:
                self._motion_cooldown = self._COOLDOWN_FRAMES
                if not self._recorder.is_recording:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    clip_path = os.path.join(RECORDINGS_DIR, f"clip_{ts}.avi")
                    h, w = display_frame.shape[:2]
                    self._recorder.start(clip_path, (w, h), fps=20.0)
                    if human_count > 0:
                        event_msg = f"HUMAN DETECTED — {human_count} human(s), {vehicle_count} vehicle(s), {object_count} object(s)"
                    elif vehicle_count > 0:
                        event_msg = f"VEHICLE DETECTED — {vehicle_count} vehicle(s), {object_count} object(s)"
                    else:
                        event_msg = f"OBJECT DETECTED — {object_count} object(s)"
                    self.root.after(0, self._log_event, event_msg)
                    logger.info(event_msg)
                    self._events_count += 1
                    self._last_log_time = datetime.now()

                else:
                    now = datetime.now()
                    if (now - self._last_log_time).total_seconds() >= 10:
                        if human_count > 0:
                            msg = f"Human ongoing — {human_count} human(s), {vehicle_count} vehicle(s), {object_count} object(s)"
                        elif vehicle_count > 0:
                            msg = f"Vehicle ongoing — {vehicle_count} vehicle(s), {object_count} object(s)"
                        else:
                            msg = f"Objects ongoing — {object_count} object(s)"
                        self.root.after(0, self._log_event, msg)
                        logger.info(msg)
                        self._events_count += 1
                        self._last_log_time = now
            else:
                if self._motion_cooldown > 0:
                    self._motion_cooldown -= 1
                elif self._recorder.is_recording:
                    clip = self._recorder.stop()
                    if clip:
                        msg = f"Clip saved: {os.path.basename(clip)}"
                        self.root.after(0, self._log_event, msg)
                        logger.info(msg)

            # Annotate display frame
            annotated = MotionDetector.annotate_frame(
                display_frame, tracked, recording=self._recorder.is_recording
            )

            # Draw zones on frame
            annotated = self._draw_zones(annotated, self._zone_manager)

            # Write annotated frame to recorder (crucial: saves overlays)
            if self._recorder.is_recording:
                self._recorder.write(annotated)

            # FPS calculation
            self._frame_count += 1
            elapsed = (datetime.now() - self._fps_timer).total_seconds()
            if elapsed >= 1.0:
                self._fps_display = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = datetime.now()

            # Push to GUI queue (drop if full to avoid lag)
            payload = (annotated, human_count, vehicle_count, object_count, self._fps_display,
                       self._recorder.is_recording)
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(payload)
                except queue.Full:
                    pass

    # ── GUI refresh loop ──────────────────────────────────────────────

    def _refresh_canvas(self):
        """
        Polls the frame queue and updates the canvas.
        Scheduled via root.after() to keep Tkinter responsive.
        ~33 ms interval = 30 FPS ceiling for the UI.
        """
        try:
            annotated, human_count, vehicle_count, object_count, fps, is_rec = self._queue.get_nowait()
            self._render_frame(annotated)
            self._fps_var.set(f"FPS: {fps:.1f}")
            self._counts_var.set(f"Humans: {human_count}  Vehicles: {vehicle_count}  Objects: {object_count}")
            self._events_var.set(f"Events: {self._events_count}")
            self._rec_var.set("● REC" if is_rec else "")
        except queue.Empty:
            pass
        finally:
            self.root.after(33, self._refresh_canvas)

    def _render_frame(self, bgr_frame: np.ndarray):
        """
        Convert a BGR numpy array to a Tkinter-compatible image and
        display it on the canvas, scaled to fit the current canvas size.

        Conversion chain:
            BGR (OpenCV) → RGB (PIL) → ImageTk.PhotoImage (Tkinter)
        """
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        fh, fw = bgr_frame.shape[:2]
        scale = min(cw / fw, ch / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)

        # Resize with INTER_LINEAR (bilinear) — good quality/speed balance
        resized = cv2.resize(bgr_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Fix 4: Letterbox — centre the scaled frame on a black canvas-sized
        # background so no grey/white strips appear when aspect ratios differ.
        canvas_bg = np.zeros((ch, cw, 3), dtype=np.uint8)
        x_off = (cw - new_w) // 2
        y_off = (ch - new_h) // 2
        canvas_bg[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # BGR → RGB for PIL
        rgb = cv2.cvtColor(canvas_bg, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._tk_image = ImageTk.PhotoImage(pil_img)  # hold reference

        if self._canvas_image_id is None:
            self._canvas_image_id = self._canvas.create_image(
                0, 0, anchor="nw", image=self._tk_image
            )
        else:
            self._canvas.itemconfig(self._canvas_image_id, image=self._tk_image)
            self._canvas.coords(self._canvas_image_id, 0, 0)

    # ── Helpers ───────────────────────────────────────────────────────

    def _set_status(self, text: str, colour: str):
        self._status_var.set(text)
        self._status_label.config(fg=colour)

    def _set_alert(self, message: str, duration_ms: int = 6000):
        """Show a temporary alert message in the title bar."""
        self._alert_var.set(message)
        self.root.after(duration_ms, lambda: self._alert_var.set(""))

    def _log_event(self, message: str):
        """Append a timestamped line to the on-screen event log."""
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {message}\n"
        self._log_box.config(state="normal")
        self._log_box.insert("end", line)
        self._log_box.see("end")
        self._log_box.config(state="disabled")

    @staticmethod
    def _draw_zones(frame: np.ndarray, zone_manager: ZoneManager) -> np.ndarray:
        """Draw all enabled zones on the frame"""
        annotated = frame.copy()
        for zone in zone_manager.list_zones():
            if not zone.enabled:
                continue
            
            # Draw zone polygon
            pts = np.array(zone.coords, dtype=np.int32)
            cv2.polylines(annotated, [pts], True, zone.color, 2)
            
            # Draw zone name
            if len(zone.coords) > 0:
                x, y = zone.coords[0]
                cv2.putText(annotated, zone.name, (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone.color, 1)
        
        return annotated

    def _on_close(self):
        """Gracefully shut down before destroying the window."""
        self._running = False
        if self._recorder.is_recording:
            self._recorder.stop()
        if self._cap:
            self._cap.release()
        self.root.destroy()

    # ── Zone Management ───────────────────────────────────────────────

    def _add_zone_dialog(self):
        """Open dialog to add a new restricted zone"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Restricted Zone")
        dialog.geometry("400x250")
        dialog.configure(bg=PALETTE["panel"])

        tk.Label(dialog, text="Zone Name:", bg=PALETTE["panel"],
                fg=PALETTE["text"], font=("Segoe UI", 10)).pack(pady=5)
        name_entry = tk.Entry(dialog, bg=PALETTE["button"], fg=PALETTE["text"],
                             font=("Segoe UI", 10), width=30)
        name_entry.pack(pady=5)

        tk.Label(dialog, text="Alert Settings:", bg=PALETTE["panel"],
                fg=PALETTE["text"], font=("Segoe UI", 10)).pack(pady=5)

        alert_human = tk.BooleanVar(value=True)
        alert_object = tk.BooleanVar(value=False)
        alert_after_hours = tk.BooleanVar(value=False)

        tk.Checkbutton(dialog, text="Alert on Human Detection", variable=alert_human,
                      bg=PALETTE["panel"], fg=PALETTE["text"], selectcolor=PALETTE["panel"]).pack(anchor="w", padx=20)
        tk.Checkbutton(dialog, text="Alert on Object Detection", variable=alert_object,
                      bg=PALETTE["panel"], fg=PALETTE["text"], selectcolor=PALETTE["panel"]).pack(anchor="w", padx=20)
        tk.Checkbutton(dialog, text="Alert After Hours Only", variable=alert_after_hours,
                      bg=PALETTE["panel"], fg=PALETTE["text"], selectcolor=PALETTE["panel"]).pack(anchor="w", padx=20)

        def save_zone():
            name = name_entry.get().strip()
            if not name:
                messagebox.showerror("Error", "Please enter a zone name")
                return
            
            zone_id = f"zone_{len(self._zone_manager.zones) + 1}"
            self._zone_manager.add_zone(
                zone_id, name,
                [(50, 50), (400, 50), (400, 400), (50, 400)],
                alert_on_human=alert_human.get(),
                alert_on_object=alert_object.get(),
                alert_after_hours=alert_after_hours.get()
            )
            self._log_event(f"Zone created: {name}")
            dialog.destroy()

        make_button(dialog, "Create Zone", save_zone, colour=PALETTE["accent"]).pack(pady=10)

    def _show_zones_dialog(self):
        """Display configured zones and allow deletion."""
        zones = self._zone_manager.list_zones()
        if not zones:
            messagebox.showinfo("No Zones", "No restricted zones configured yet.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Configured Zones")
        dialog.geometry("540x320")
        dialog.configure(bg=PALETTE["panel"])

        left_frame = tk.Frame(dialog, bg=PALETTE["panel"])
        left_frame.pack(side="left", fill="y", padx=(10, 5), pady=10)

        tk.Label(
            left_frame,
            text="Zones",
            bg=PALETTE["panel"],
            fg=PALETTE["text"],
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        listbox = tk.Listbox(
            left_frame,
            bg=PALETTE["button"],
            fg=PALETTE["text"],
            selectbackground=PALETTE["accent"],
            activestyle="none",
            width=24,
            height=12,
            exportselection=False,
        )
        listbox.pack(fill="y", pady=8)

        for zone in zones:
            listbox.insert("end", zone.name)

        detail_frame = tk.Frame(dialog, bg=PALETTE["panel"])
        detail_frame.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)

        tk.Label(
            detail_frame,
            text="Zone Details",
            bg=PALETTE["panel"],
            fg=PALETTE["text"],
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        details_text = scrolledtext.ScrolledText(
            detail_frame,
            bg=PALETTE["log_bg"],
            fg=PALETTE["log_text"],
            font=("Courier New", 9),
            relief="flat",
            bd=0,
            wrap="word",
            state="disabled",
            height=12,
        )
        details_text.pack(fill="both", expand=True, pady=8)

        def refresh_zone_details(event=None):
            selection = listbox.curselection()
            if not selection:
                details_text.config(state="normal")
                details_text.delete("1.0", "end")
                details_text.config(state="disabled")
                delete_button.config(state="disabled")
                return

            index = selection[0]
            zone = zones[index]
            info = f"Zone: {zone.name}\n"
            info += f"ID: {zone.zone_id}\n"
            info += f"Alerts:\n"
            info += f"  Human: {zone.alert_on_human}\n"
            info += f"  Object: {zone.alert_on_object}\n"
            info += f"  After-Hours: {zone.alert_after_hours}\n"
            info += f"Enabled: {zone.enabled}\n"
            info += f"Coords: {zone.coords}\n"

            details_text.config(state="normal")
            details_text.delete("1.0", "end")
            details_text.insert("end", info)
            details_text.config(state="disabled")
            delete_button.config(state="normal")

        listbox.bind("<<ListboxSelect>>", refresh_zone_details)

        def delete_selected_zone():
            selection = listbox.curselection()
            if not selection:
                return

            index = selection[0]
            zone = zones[index]
            confirm = messagebox.askyesno(
                "Delete Zone",
                f"Are you sure you want to delete the zone '{zone.name}'?",
                parent=dialog,
            )
            if not confirm:
                return

            self._zone_manager.remove_zone(zone.zone_id)
            self._log_event(f"Zone deleted: {zone.name}")
            zones.pop(index)
            listbox.delete(index)
            details_text.config(state="normal")
            details_text.delete("1.0", "end")
            details_text.config(state="disabled")
            delete_button.config(state="disabled")

            if not zones:
                messagebox.showinfo("No Zones", "All restricted zones have been deleted.", parent=dialog)
                dialog.destroy()

        delete_button = make_button(detail_frame, "🗑  Delete Selected Zone", delete_selected_zone,
                                    colour=PALETTE["danger"], width=22)
        delete_button.pack(pady=4)
        delete_button.config(state="disabled")

        # Pre-select the first zone if available
        if zones:
            listbox.selection_set(0)
            refresh_zone_details()

    def _show_alert_config(self):
        """Open alert configuration dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Alert Configuration")
        dialog.geometry("400x200")
        dialog.configure(bg=PALETTE["panel"])

        tk.Label(dialog, text="After-Hours Window:", bg=PALETTE["panel"],
                fg=PALETTE["text"], font=("Segoe UI", 10, "bold")).pack(pady=10)

        frame = tk.Frame(dialog, bg=PALETTE["panel"])
        frame.pack(pady=5)

        tk.Label(frame, text="Start Hour:", bg=PALETTE["panel"],
                fg=PALETTE["text"]).pack(side="left", padx=5)
        start_var = tk.IntVar(value=self._zone_manager.after_hours[0])
        tk.Spinbox(frame, from_=0, to=23, textvariable=start_var,
                  bg=PALETTE["button"], fg=PALETTE["text"],
                  width=3).pack(side="left", padx=5)

        tk.Label(frame, text="End Hour:", bg=PALETTE["panel"],
                fg=PALETTE["text"]).pack(side="left", padx=5)
        end_var = tk.IntVar(value=self._zone_manager.after_hours[1])
        tk.Spinbox(frame, from_=0, to=23, textvariable=end_var,
                  bg=PALETTE["button"], fg=PALETTE["text"],
                  width=3).pack(side="left", padx=5)

        tk.Label(dialog, text="Restricted Objects:", bg=PALETTE["panel"],
                fg=PALETTE["text"], font=("Segoe UI", 10, "bold")).pack(pady=5)

        obj_text = tk.Text(dialog, bg=PALETTE["log_bg"], fg=PALETTE["log_text"],
                          font=("Courier New", 8), height=4, width=40)
        obj_text.pack(pady=5)
        obj_text.insert("end", ", ".join(self._zone_manager.restricted_objects))

        def save_config():
            self._zone_manager.set_after_hours(start_var.get(), end_var.get())
            new_objects = [s.strip() for s in obj_text.get("1.0", "end").split(",")]
            self._zone_manager.restricted_objects = [o.lower() for o in new_objects if o]
            self._zone_manager.save_config()
            self._log_event("Alert configuration updated")
            dialog.destroy()

        make_button(dialog, "Save Configuration", save_config, colour=PALETTE["accent"]).pack(pady=10)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()

    # Apply a basic dark window icon if possible (silent fail if unavailable)
    try:
        root.iconbitmap("icon.ico")
    except Exception:
        pass

    app = SecurityCameraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
