def predict_future(x, y, vx, vy, seconds=2.0):
    future_x = int(x + vx * seconds)
    future_y = int(y + vy * seconds)
    return future_x, future_y