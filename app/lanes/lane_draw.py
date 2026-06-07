import cv2


def draw_lanes(frame):

    h, w = frame.shape[:2]

    lane_width = w // 4

    for i in range(1, 4):

        x = lane_width * i

        cv2.line(
            frame,
            (x, 0),
            (x, h),
            (255, 255, 255),
            2
        )

    return frame