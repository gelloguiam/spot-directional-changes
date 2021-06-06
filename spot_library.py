import math

def get_angle(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    x3 = x1
    y3 = y2

    p3 = (x3, y3)

    a = get_distance(p1, p2)
    b = get_distance(p1, p3)
    c = get_distance(p2, p3)

    if b == 0 or c == 0:
        # print("ZERO:", p1, p2, p3, a, b, c)
        return 0

    z = (c**2 - a**2 - b**2)/(-2*b*a)
    angle_rad = math.acos(z)
    angle_deg = math.degrees(angle_rad)

    return angle_deg
    # print(p1, p2, p3, a, b, c, angle_deg)


def get_distance(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2

    d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return d


def get_slope(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2

    if (x1 == x2) return 0
    return (y2-y1)/(x2-x1)