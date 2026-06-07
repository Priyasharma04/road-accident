def get_velocity(history, fps=30, window=5):
    if len(history) < window:
        return 0.0, 0.0

    x1, y1 = history[-window]
    x2, y2 = history[-1]

    dt = (window - 1) / fps

    vx = (x2 - x1) / dt
    vy = (y2 - y1) / dt

    return vx, vy