import math


def calculate_distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def calculate_ttc(
        current_distance,
        relative_speed
):

    if relative_speed <= 0:
        return float("inf")

    return current_distance / relative_speed