def get_velocity(history):

    if len(history) < 5:
        return (0, 0)

    x1, y1 = history[-5]
    x2, y2 = history[-1]

    vx = (x2 - x1) / 4
    vy = (y2 - y1) / 4

    return (vx, vy)