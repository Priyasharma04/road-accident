import math

def calculate_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_ttc(current_distance, future_distance, prediction_seconds=2.0):
    closing_distance = current_distance - future_distance

    if closing_distance <= 1:
        return float("inf")

    closing_speed = closing_distance / prediction_seconds
    return current_distance / closing_speed