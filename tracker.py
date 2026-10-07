"""
tracker.py — Centroid Tracker for Smart Security Camera System
CSY3058 Media Technology | Assignment 1

Multimedia Principle:
    Object tracking maintains temporal identity across video frames.
    By calculating the Euclidean distance between centroids of detected
    contours across consecutive frames, we can associate the same physical
    object with the same ID over time — essential for meaningful event logging.
"""

import math
from collections import OrderedDict


class CentroidTracker:
    """
    Tracks objects across video frames using centroid (geometric centre)
    proximity as the association metric.

    Algorithm:
        1. On each frame, receive a list of bounding boxes from the detector.
        2. Compute the centroid (cx, cy) of each box.
        3. For each new centroid, find the closest existing tracked object.
        4. If the distance is within `max_distance`, update that object's
           centroid; otherwise register it as a new object.
        5. Objects not seen for `max_disappeared` consecutive frames are
           deregistered to prevent ghost IDs accumulating.

    Attributes:
        next_object_id (int): Auto-incrementing ID counter.
        objects (OrderedDict): Maps object_id → centroid (cx, cy).
        bboxes (OrderedDict): Maps object_id → bounding box (x, y, w, h).
        disappeared (OrderedDict): Maps object_id → consecutive missed frames.
        max_disappeared (int): Frames before an object is deregistered.
        max_distance (int): Max pixel distance to consider same object.
    """

    def __init__(self, max_disappeared: int = 40, max_distance: int = 80):
        self.next_object_id = 0
        self.objects: OrderedDict = OrderedDict()
        self.bboxes: OrderedDict = OrderedDict()
        self.labels: OrderedDict = OrderedDict()
        self.disappeared: OrderedDict = OrderedDict()

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register(self, centroid: tuple, bbox: tuple, label: str) -> None:
        """Register a brand-new object with the next available ID."""
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.labels[self.next_object_id] = label
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def _deregister(self, object_id: int) -> None:
        """Remove a lost object from all tracking dictionaries."""
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.labels[object_id]
        del self.disappeared[object_id]

    @staticmethod
    def _centroid(bbox: tuple) -> tuple:
        """Return the (cx, cy) centroid of a bounding box (x, y, w, h)."""
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    @staticmethod
    def _euclidean(a: tuple, b: tuple) -> float:
        """Euclidean distance between two 2-D points."""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: list, frame_size: tuple = (960, 510)) -> OrderedDict:
        """
        Update tracker state with the current frame's detected objects.

        Args:
            detections: List of ((x, y, w, h), label) tuples from the detector.
            frame_size: (width, height) of the display frame.

        Returns:
            OrderedDict mapping object_id → (centroid, bbox, label).
        """
        self._frame_size = frame_size

        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self._build_result()

        input_bboxes = [bbox for bbox, _ in detections]
        input_labels = [label for _, label in detections]
        input_centroids = [self._centroid(bbox) for bbox in input_bboxes]

        if len(self.objects) == 0:
            for bbox, label, centroid in zip(input_bboxes, input_labels, input_centroids):
                self._register(centroid, bbox, label)
            return self._build_result()

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = [
            [self._euclidean(oc, ic) for ic in input_centroids]
            for oc in object_centroids
        ]

        rows = sorted(range(len(D)), key=lambda r: min(D[r]))
        used_rows, used_cols = set(), set()

        for row in rows:
            col = min(range(len(D[row])), key=lambda c: D[row][c])
            if row in used_rows or col in used_cols:
                continue
            bbox = input_bboxes[col]
            dynamic_max = max(self.max_distance, int(max(bbox[2], bbox[3]) * 0.7))
            if D[row][col] > dynamic_max:
                continue
            oid = object_ids[row]
            self.objects[oid] = input_centroids[col]
            self.bboxes[oid] = input_bboxes[col]
            self.labels[oid] = input_labels[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(len(object_ids))) - used_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self._deregister(oid)

        for col in set(range(len(input_centroids))) - used_cols:
            self._register(input_centroids[col], input_bboxes[col], input_labels[col])

        return self._build_result()

    def _build_result(self) -> OrderedDict:
        """
        Return a dict of {object_id: (centroid, bbox, label)} for all
        currently tracked objects.
        """
        result = OrderedDict()
        for oid, centroid in self.objects.items():
            bbox = self.bboxes[oid]
            label = self.labels[oid]
            result[oid] = (centroid, bbox, label)
        return result
