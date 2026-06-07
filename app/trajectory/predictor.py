def predict_future(x, y, vx, vy, frames=20):

    future_x = int(x + vx * frames)
    future_y = int(y + vy * frames)

    return future_x, future_y