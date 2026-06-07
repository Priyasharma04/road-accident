import cv2
import math
from ultralytics import YOLO

from trajectory.history import track_history
from trajectory.velocity import get_velocity
from trajectory.predictor import predict_future

# =====================================
# CONFIG
# =====================================

WARNING_TTC = 5
DANGER_TTC = 2

PERSON = 0

VEHICLES = {
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7    # truck
}

# =====================================
# GEOMETRY
# =====================================

def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0])**2 +
        (p1[1] - p2[1])**2
    )


def ccw(A, B, C):

    return (
        (C[1]-A[1])*(B[0]-A[0])
        >
        (B[1]-A[1])*(C[0]-A[0])
    )


def intersect(A, B, C, D):

    return (
        ccw(A, C, D) != ccw(B, C, D)
        and
        ccw(A, B, C) != ccw(A, B, D)
    )

# =====================================
# MODEL
# =====================================

model = YOLO("weights/yolo11m.pt")

# =====================================
# VIDEO
# =====================================

cap = cv2.VideoCapture("videos/test.mp4")

if not cap.isOpened():
    print("Video not found")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "outputs/result.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# =====================================
# LOOP
# =====================================

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

    if len(results) == 0:
        continue

    boxes = results[0].boxes

    objects = {}

    # =====================================
    # TRACK OBJECTS
    # =====================================

    for box in boxes:

        if box.id is None:
            continue

        cls = int(box.cls[0])

        if cls != PERSON and cls not in VEHICLES:
            continue

        track_id = int(box.id)

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        track_history[track_id].append(
            (cx, cy)
        )

        if len(track_history[track_id]) > 30:
            track_history[track_id].pop(0)

        points = track_history[track_id]

        for i in range(1, len(points)):

            cv2.line(
                frame,
                points[i - 1],
                points[i],
                (255, 0, 0),
                2
            )

        vx, vy = get_velocity(points)

        future_x, future_y = predict_future(
            cx,
            cy,
            vx,
            vy,
            frames=20
        )

        cv2.circle(
            frame,
            (future_x, future_y),
            5,
            (0, 255, 255),
            -1
        )

        cv2.line(
            frame,
            (cx, cy),
            (future_x, future_y),
            (255, 255, 0),
            2
        )

        objects[track_id] = {
            "class": cls,
            "center": (cx, cy),
            "future": (future_x, future_y),
            "velocity": (vx, vy),
            "bbox": (x1, y1, x2, y2),
            "status": "safe"
        }

    # =====================================
    # COLLISION PREDICTION
    # =====================================

    ids = list(objects.keys())

    for i in range(len(ids)):

        for j in range(i + 1, len(ids)):

            obj1 = objects[ids[i]]
            obj2 = objects[ids[j]]

            p1 = obj1["center"]
            p2 = obj1["future"]

            p3 = obj2["center"]
            p4 = obj2["future"]

            # trajectories do not cross
            if not intersect(
                p1,
                p2,
                p3,
                p4
            ):
                continue

            current_distance = distance(
                p1,
                p3
            )

            vx1, vy1 = obj1["velocity"]
            vx2, vy2 = obj2["velocity"]

            relative_speed = math.sqrt(
                (vx1-vx2)**2 +
                (vy1-vy2)**2
            )

            if relative_speed < 1:
                continue

            ttc = current_distance / relative_speed

            if ttc < DANGER_TTC:

                obj1["status"] = "danger"
                obj2["status"] = "danger"

            elif ttc < WARNING_TTC:

                if obj1["status"] != "danger":
                    obj1["status"] = "warning"

                if obj2["status"] != "danger":
                    obj2["status"] = "warning"

    # =====================================
    # DRAW
    # =====================================

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

        cv2.putText(
            frame,
            f"ID:{track_id} {status}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    out.write(frame)

    cv2.imshow(
        "Road Accident Prediction",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()

cv2.destroyAllWindows()