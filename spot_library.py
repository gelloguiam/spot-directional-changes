import math
import numpy as np

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

    if (x1 == x2): return 0
    return (y2-y1)/(x2-x1)

# def get_direction(p1, p2):
#     (x1, y1) = p1
#     (x2, y2) = p2

#     dX = x2 - x1
#     dY = y2 - y1

#     direction = ""

#     if(dY < 0):
#         direction += "N"
#     elif(dY > 0):
#         direction += "S"

#     if(dX < 0):
#         direction += "W"
#     elif(dX > 0):
#         direction += "E"
    
#     return direction

def get_direction(prevPoint, currentPoint):
    dX = currentPoint[0] - prevPoint[0]
    dY = prevPoint[1] - currentPoint[1]
    (dirX, dirY) = ("", "")
    if np.abs(dX) > 3:
        #The sign function returns -1 if dX < 0, 0 if dX==0, 1 if dX > 0. nan is returned for nan
        dirX = "E" if np.sign(dX) == 1 else "W"
    if np.abs(dY) > 3:
        dirY = "N" if np.sign(dY) == 1 else "S"
    if dirX != "" and dirY != "":
        direction = "{}{}".format(dirY, dirX)
    else:
        direction = dirX if dirX != "" else dirY
    return direction
