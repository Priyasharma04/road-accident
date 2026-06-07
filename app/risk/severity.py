def get_severity(ttc):

    if ttc <= 2:
        return "danger"

    elif ttc <= 5:
        return "warning"

    return "safe"