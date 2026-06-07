import cv2
import numpy as np

LANES = {
    "lane_1": [(100, 720), (320, 420), (520, 420), (420, 720)],
    "lane_2": [(420, 720), (520, 420), (720, 420), (760, 720)],
}

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def assign_lane(point):
    for lane_name, polygon in LANES.items():
        if point_in_polygon(point, polygon):
            return lane_name
    return None

def draw_lanes(frame):
    for lane_name, polygon in LANES.items():
        pts = np.array(polygon, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
        x, y = pts[0]
        cv2.putText(frame, lane_name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)