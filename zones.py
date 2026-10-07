"""
zones.py — Restricted Zone Management and Alert Configuration

Provides:
  • Zone drawing and management
  • ROI (Region of Interest) detection
  • Time-based alert configuration
  • Restricted object tracking
  • Alert logging and notifications
"""

import json
import os
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional


class AlertType(Enum):
    """Types of security alerts"""
    UNAUTHORIZED_HUMAN = "unauthorized_human"
    UNAUTHORIZED_OBJECT = "unauthorized_object"
    AFTER_HOURS = "after_hours"
    RESTRICTED_OBJECT_MOVED = "restricted_object_moved"
    ZONE_BREACHED = "zone_breached"


@dataclass
class Zone:
    """Represents a restricted zone in the warehouse"""
    zone_id: str
    name: str
    coords: List[Tuple[int, int]]  # [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    enabled: bool = True
    alert_on_human: bool = True
    alert_on_object: bool = False
    alert_after_hours: bool = False
    color: Tuple[int, int, int] = (0, 0, 255)  # BGR: red by default

    def __post_init__(self):
        """Normalize coords to 4 points if needed"""
        if len(self.coords) == 2:
            # Convert rect (x1,y1,x2,y2) to 4 points
            x1, y1 = self.coords[0]
            x2, y2 = self.coords[1]
            self.coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if point is inside zone using ray casting"""
        x, y = point
        n = len(self.coords)
        inside = False

        p1x, p1y = self.coords[0]
        for i in range(1, n + 1):
            p2x, p2y = self.coords[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def bbox_intersects(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if bbox (x,y,w,h) overlaps with zone"""
        x, y, w, h = bbox
        bbox_x2 = x + w
        bbox_y2 = y + h

        # Get zone bounds
        xs = [coord[0] for coord in self.coords]
        ys = [coord[1] for coord in self.coords]
        zone_x1, zone_y1 = min(xs), min(ys)
        zone_x2, zone_y2 = max(xs), max(ys)

        # Check intersection
        if x < zone_x2 and bbox_x2 > zone_x1 and y < zone_y2 and bbox_y2 > zone_y1:
            return True
        return False


class ZoneManager:
    """Manage restricted zones and alert configurations"""

    def __init__(self, config_file: str = "zones_config.json"):
        self.config_file = config_file
        self.zones: dict[str, Zone] = {}
        self.restricted_objects = [
            "backpack", "suitcase", "handbag", "laptop", "briefcase",
            "bag", "box", "package", "cardboard box"
        ]
        self.after_hours = (22, 6)  # 10 PM to 6 AM
        self.load_config()

    def add_zone(self, zone_id: str, name: str, coords: List[Tuple[int, int]],
                 alert_on_human: bool = True, alert_on_object: bool = False,
                 alert_after_hours: bool = False, color: Tuple[int, int, int] = (0, 0, 255)) -> Zone:
        """Add a new restricted zone"""
        zone = Zone(
            zone_id=zone_id,
            name=name,
            coords=coords,
            alert_on_human=alert_on_human,
            alert_on_object=alert_on_object,
            alert_after_hours=alert_after_hours,
            color=color
        )
        self.zones[zone_id] = zone
        self.save_config()
        return zone

    def remove_zone(self, zone_id: str):
        """Remove a zone"""
        if zone_id in self.zones:
            del self.zones[zone_id]
            self.save_config()

    def get_zone(self, zone_id: str) -> Optional[Zone]:
        """Get zone by ID"""
        return self.zones.get(zone_id)

    def list_zones(self) -> List[Zone]:
        """Get all zones"""
        return list(self.zones.values())

    def check_detection(self, bbox: Tuple[int, int, int, int], label: str,
                        object_name: str = "") -> List[Tuple[AlertType, Zone]]:
        """
        Check if a detection triggers any alerts.
        Returns list of (AlertType, Zone) tuples for triggered alerts.
        """
        alerts = []

        is_after_hours = self._is_after_hours()
        is_human = label == "HUMAN"
        is_object = label == "OBJECT"
        is_vehicle = label == "VEHICLE"

        for zone in self.zones.values():
            if not zone.enabled or not zone.bbox_intersects(bbox):
                continue

            # Check human alerts
            if is_human and zone.alert_on_human:
                alerts.append((AlertType.ZONE_BREACHED, zone))
                if zone.alert_after_hours and is_after_hours:
                    alerts.append((AlertType.AFTER_HOURS, zone))

            # Check object/vehicle alerts
            if (is_object or is_vehicle) and zone.alert_on_object:
                alerts.append((AlertType.ZONE_BREACHED, zone))
                if object_name.lower() in self.restricted_objects:
                    alerts.append((AlertType.RESTRICTED_OBJECT_MOVED, zone))

        return alerts

    def is_restricted_object(self, object_name: str) -> bool:
        """Check if object is in restricted list"""
        return object_name.lower() in self.restricted_objects

    def add_restricted_object(self, object_name: str):
        """Add object to restricted list"""
        if object_name.lower() not in self.restricted_objects:
            self.restricted_objects.append(object_name.lower())
            self.save_config()

    def remove_restricted_object(self, object_name: str):
        """Remove object from restricted list"""
        if object_name.lower() in self.restricted_objects:
            self.restricted_objects.remove(object_name.lower())
            self.save_config()

    def set_after_hours(self, start_hour: int, end_hour: int):
        """Set after-hours window (e.g., 22, 6 = 10 PM to 6 AM)"""
        self.after_hours = (start_hour, end_hour)
        self.save_config()

    def _is_after_hours(self) -> bool:
        """Check if current time is within after-hours window"""
        now = datetime.now()
        current_hour = now.hour
        start, end = self.after_hours

        if start < end:
            return start <= current_hour < end
        else:
            # Wraps around midnight (e.g., 22 to 6)
            return current_hour >= start or current_hour < end

    def save_config(self):
        """Save zones and config to JSON file"""
        data = {
            "zones": {zone_id: asdict(zone) for zone_id, zone in self.zones.items()},
            "restricted_objects": self.restricted_objects,
            "after_hours": list(self.after_hours),
        }
        os.makedirs(os.path.dirname(self.config_file) or ".", exist_ok=True)
        with open(self.config_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_config(self):
        """Load zones and config from JSON file"""
        if not os.path.exists(self.config_file):
            return

        try:
            with open(self.config_file, "r") as f:
                data = json.load(f)

            # Load zones
            for zone_id, zone_data in data.get("zones", {}).items():
                zone_data["coords"] = [tuple(c) for c in zone_data["coords"]]
                zone_data["color"] = tuple(zone_data["color"])
                self.zones[zone_id] = Zone(**zone_data)

            # Load restricted objects
            self.restricted_objects = data.get("restricted_objects", self.restricted_objects)

            # Load after-hours
            after_hours = data.get("after_hours", list(self.after_hours))
            self.after_hours = tuple(after_hours)
        except Exception as e:
            print(f"Error loading config: {e}")


class AlertLogger:
    """Log and manage security alerts"""

    def __init__(self, log_file: str = "security_alerts.log"):
        self.log_file = log_file
        self.alert_history: List[dict] = []

    def log_alert(self, alert_type: AlertType, zone: Zone, detection_label: str,
                  object_name: str = "", confidence: float = 0.0):
        """Log a security alert"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "alert_type": alert_type.value,
            "zone": zone.name if zone else "None",
            "detection_label": detection_label,
            "object_name": object_name,
            "confidence": confidence,
        }
        self.alert_history.append(entry)

        # Write to file
        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(f"{entry['timestamp']} | {alert_type.value} | {zone.name if zone else 'None'} | "
                   f"{detection_label} | {object_name} | {confidence:.2f}\n")

    def get_recent_alerts(self, limit: int = 50) -> List[dict]:
        """Get recent alerts"""
        return self.alert_history[-limit:]
