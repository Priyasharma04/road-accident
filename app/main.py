import cv2
import os
from collections import defaultdict
from ultralytics import YOLO

from trajectory.history import track_history
from trajectory.velocity import get_velocity
from trajectory.predictor import predict_future
from trajectory.ttc import calculate_distance, calculate_ttc
from risk.severity import get_severity


MODEL_PATH = "weights/yolo11m.pt"
VIDEO_PATH = "videos/test.mp4"
OUTPUT_PATH = "outputs/result.mp4"

PREDICTION_SECONDS = 2.0

WARNING_DISTANCE = 160
DANGER_DISTANCE = 90
STATIONARY_SPEED = 8

PAIR_HISTORY_FRAMES = 5
PAIR_CONFIRM_COUNT = 3

PERSON = 0
VEHICLES = {1, 2, 3, 5, 7}

pair_history = defaultdict(list)


def bbox_overlap_or_close(box1, box2, margin=30):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    return not (
        x2 + margin < a1 or
        a2 + margin < x1 or
        y2 + margin < b1 or
        b2 + margin < y1
    )


def upgrade_status(old_status, new_status):
    priority = {
        "safe": 0,
        "warning": 1,
        "danger": 2
    }

    if priority[new_status] > priority[old_status]:
        return new_status

    return old_status


def get_speed(velocity):
    vx, vy = velocity
    return (vx ** 2 + vy ** 2) ** 0.5


def confirm_risk(pair_key, instant_status):
    pair_history[pair_key].append(instant_status)

    if len(pair_history[pair_key]) > PAIR_HISTORY_FRAMES:
        pair_history[pair_key].pop(0)

    danger_count = pair_history[pair_key].count("danger")
    warning_count = pair_history[pair_key].count("warning")

    if danger_count >= PAIR_CONFIRM_COUNT:
        return "danger"

    if danger_count + warning_count >= PAIR_CONFIRM_COUNT:
        return "warning"

    return "safe"


os.makedirs("outputs", exist_ok=True)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Video not found:", VIDEO_PATH)
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if fps == 0:
    fps = 30

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)


while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        verbose=False
    )

    objects = {}

    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes

        for box in boxes:
            if box.id is None:
                continue

            cls = int(box.cls[0])

            if cls != PERSON and cls not in VEHICLES:
                continue

            track_id = int(box.id[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = y2
            point = (cx, cy)

            track_history[track_id].append(point)

            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

            points = track_history[track_id]

            for k in range(1, len(points)):
                cv2.line(
                    frame,
                    points[k - 1],
                    points[k],
                    (255, 0, 0),
                    2
                )

            vx, vy = get_velocity(points, fps=fps)

            future_x, future_y = predict_future(
                cx,
                cy,
                vx,
                vy,
                seconds=PREDICTION_SECONDS
            )

            future_point = (future_x, future_y)

            cv2.circle(
                frame,
                future_point,
                5,
                (0, 255, 255),
                -1
            )

            cv2.line(
                frame,
                point,
                future_point,
                (255, 255, 0),
                2
            )

            objects[track_id] = {
                "class": cls,
                "point": point,
                "future": future_point,
                "velocity": (vx, vy),
                "bbox": (x1, y1, x2, y2),
                "status": "safe",
                "ttc": None
            }

    ids = list(objects.keys())

    danger_pairs = []
    warning_pairs = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):

            id1 = ids[i]
            id2 = ids[j]

            obj1 = objects[id1]
            obj2 = objects[id2]

            speed1 = get_speed(obj1["velocity"])
            speed2 = get_speed(obj2["velocity"])

            if speed1 < STATIONARY_SPEED and speed2 < STATIONARY_SPEED:
                continue

            person_involved = (
                obj1["class"] == PERSON
                or obj2["class"] == PERSON
            )

            both_person = (
                obj1["class"] == PERSON
                and obj2["class"] == PERSON
            )

            if both_person:
                continue

            current_distance = calculate_distance(
                obj1["point"],
                obj2["point"]
            )

            future_distance = calculate_distance(
                obj1["future"],
                obj2["future"]
            )

            ttc = float("inf")

            if future_distance < current_distance:
                ttc = calculate_ttc(
                    current_distance,
                    future_distance,
                    prediction_seconds=PREDICTION_SECONDS
                )

            status = get_severity(ttc)

            if person_involved and current_distance < WARNING_DISTANCE + 40:
                status = upgrade_status(status, "warning")

            if person_involved and current_distance < DANGER_DISTANCE + 30:
                status = upgrade_status(status, "danger")

            near_overlap = bbox_overlap_or_close(
                obj1["bbox"],
                obj2["bbox"],
                margin=35
            )

            actual_overlap = bbox_overlap_or_close(
                obj1["bbox"],
                obj2["bbox"],
                margin=0
            )

            moving_pair = (
                speed1 >= STATIONARY_SPEED
                or speed2 >= STATIONARY_SPEED
            )

            if near_overlap and moving_pair:
                status = upgrade_status(status, "warning")

            if actual_overlap and moving_pair:
                status = upgrade_status(status, "danger")

            if current_distance < 70 and moving_pair:
                status = upgrade_status(status, "danger")

            if (
                not person_involved
                and not near_overlap
                and current_distance > WARNING_DISTANCE
            ):
                status = "safe"

            pair_key = tuple(sorted((id1, id2)))
            status = confirm_risk(pair_key, status)

            if status == "danger":
                obj1["status"] = upgrade_status(obj1["status"], "danger")
                obj2["status"] = upgrade_status(obj2["status"], "danger")

                obj1["ttc"] = ttc if ttc != float("inf") else None
                obj2["ttc"] = ttc if ttc != float("inf") else None

                danger_pairs.append(f"{id1}-{id2}")

            elif status == "warning":
                obj1["status"] = upgrade_status(obj1["status"], "warning")
                obj2["status"] = upgrade_status(obj2["status"], "warning")

                obj1["ttc"] = ttc if ttc != float("inf") else None
                obj2["ttc"] = ttc if ttc != float("inf") else None

                warning_pairs.append(f"{id1}-{id2}")

    y_text = 50

    for pair in danger_pairs:
        cv2.putText(
            frame,
            f"DANGER {pair}",
            (40, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )
        y_text += 40

    for pair in warning_pairs:
        cv2.putText(
            frame,
            f"WARNING {pair}",
            (40, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )
        y_text += 40

    for track_id, obj in objects.items():

        x1, y1, x2, y2 = obj["bbox"]
        status = obj["status"]

        if status == "danger":
            color = (0, 0, 255)
        elif status == "warning":
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        label = f"ID:{track_id} {status}"

        if obj["ttc"] is not None:
            label += f" TTC:{obj['ttc']:.1f}s"

        cv2.putText(
            frame,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        cv2.circle(
            frame,
            obj["point"],
            4,
            color,
            -1
        )

    out.write(frame)

    cv2.imshow("Road Accident Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

print("Done. Output saved at:", OUTPUT_PATH)