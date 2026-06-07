def get_risk_score(ttc):

    if ttc <= 2:
        return 90

    elif ttc <= 5:
        return 60

    return 10